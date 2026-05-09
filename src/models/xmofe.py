"""Top-level X-MoFE model.

Wires the component layers from :mod:`src.models` into a single
``nn.Module`` whose ``forward`` returns predictions plus all three
explanation outputs. Per-modality input dims are passed at construction so
the same architecture serves MOSEI (text 768 / audio 74 / visual 713) and
MELD/CH-SIMS (768/768/768).

Architecture (per spec §12)::

    Z_m = Linear_m(LayerNorm(X_m))      for m ∈ {T, A, V}
    z_m, α_m = AttnPool(Z_m, length_m)
    r       = softmax(MLP_R([z_T; z_A; z_V; q?]))            # E_modality
    c_TA, c_TV, c_AV       = pooled cross-attentions          # pairwise
    c_TAV  = pooled tri-modal cross-attention (optional)
    λ      = softmax(MLP_I([c_TA; c_TV; c_AV; c_TAV]))        # E_interaction
    u      = r_T·z_T + r_A·z_A + r_V·z_V                      # unimodal evidence
    i      = λ_TA·c_TA + λ_TV·c_TV + λ_AV·c_AV + λ_TAV·c_TAV  # interaction evidence
    h      = LayerNorm(u + i) → FFN
    ŷ      = Head(h)
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPool
from src.models.explanation_heads import XMoFEOutput
from src.models.fusion_layers import HybridFusion
from src.models.interaction import (
    CrossModalInteraction,
    InteractionEstimator,
    TriModalInteraction,
)
from src.models.prediction_heads import PredictionHead
from src.models.projections import ModalityProjection
from src.models.reliability import ReliabilityEstimator


def _lazy_text_encoder_module():
    """Import ``TextEncoderModule`` only when fine-tuning is requested.

    Keeps the frozen-feature default path free of any ``transformers``
    runtime dependency in the model graph.
    """
    from src.encoders.text_encoder_module import TextEncoderModule
    return TextEncoderModule


PAIRWISE_INTERACTIONS = ("text_audio", "text_visual", "audio_visual")
TRIMODAL_INTERACTION = "trimodal"


class XMoFE(nn.Module):
    """Faithful and reliability-aware multimodal fusion.

    Args:
        text_dim, audio_dim, visual_dim: Per-modality input feature
            dimensions. Different per dataset (see module docstring).
        shared_dim: Shared projection dim ``d`` (spec §11.4 default 256).
        attention_heads: Heads in cross-modal attention blocks.
        dropout: Dropout rate applied throughout.
        use_trimodal: If True, include ``c_TAV`` and a 4-way ``λ``.
        use_reliability_gate: If False, replaces the learned reliability
            estimator with a fixed uniform ``r = [1/3, 1/3, 1/3]``. Used by
            the ``xmofe_no_reliability`` ablation.
        use_interaction_block: If False, drops the cross-modal interaction
            blocks and the interaction estimator entirely. The interaction
            term ``i`` becomes 0 and only the unimodal evidence ``u`` flows
            into the fusion. Used by the ``xmofe_no_interaction`` ablation.
        task: ``"regression"`` or ``"classification"``.
        num_classes: Output dim for classification (ignored for regression).
        num_quality_features_per_modality: Size of optional quality vector
            per modality (e.g. ``{snr, face_conf, blur}`` → 3). 0 disables.
        ffn_multiplier: Inner-dim factor in the fusion FFN.
        reliability_mlp_hidden, interaction_mlp_hidden: Hidden sizes.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        shared_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.2,
        use_trimodal: bool = True,
        use_reliability_gate: bool = True,
        use_interaction_block: bool = True,
        task: str = "regression",
        num_classes: int = 1,
        num_quality_features_per_modality: int = 0,
        ffn_multiplier: int = 4,
        reliability_mlp_hidden: int = 256,
        interaction_mlp_hidden: int = 256,
        condition_interaction_on_reliability: bool = False,
        text_encoder_finetune: bool = False,
        text_encoder_name: str = "answerdotai/ModernBERT-base",
        text_encoder_max_length: int = 128,
        text_encoder_trainable_layers: str = "last_n",
        text_encoder_last_n: int = 4,
    ) -> None:
        super().__init__()

        self.shared_dim = shared_dim
        self.use_trimodal = use_trimodal
        self.use_reliability_gate = use_reliability_gate
        self.use_interaction_block = use_interaction_block
        self.condition_interaction_on_reliability = condition_interaction_on_reliability
        self.text_encoder_finetune = text_encoder_finetune
        self.task = task

        # Optional in-graph text encoder. When ``text_encoder_finetune`` is
        # True, the model takes raw transcripts at forward-time and runs
        # them through this encoder; otherwise it expects cached text
        # features in the ``text`` argument as before. Loaded lazily so the
        # default frozen-feature path stays free of HF runtime overhead.
        if text_encoder_finetune:
            TextEncoderModule = _lazy_text_encoder_module()
            self.text_encoder = TextEncoderModule(
                model_name=text_encoder_name,
                max_length=text_encoder_max_length,
                trainable_layers=text_encoder_trainable_layers,
                last_n=text_encoder_last_n,
            )
            # Override the input text_dim so projections match the encoder's
            # actual hidden size (e.g. ModernBERT-base = 768).
            text_dim = self.text_encoder.feature_dim
        else:
            self.text_encoder = None

        # Projections
        self.proj_text = ModalityProjection(text_dim, shared_dim, dropout)
        self.proj_audio = ModalityProjection(audio_dim, shared_dim, dropout)
        self.proj_visual = ModalityProjection(visual_dim, shared_dim, dropout)

        # Per-modality attention pooling (own learnable query per modality)
        self.pool_text = AttentionPool(shared_dim)
        self.pool_audio = AttentionPool(shared_dim)
        self.pool_visual = AttentionPool(shared_dim)

        # Cross-modal interactions (skipped entirely when use_interaction_block=False)
        if use_interaction_block:
            self.cross_ta = CrossModalInteraction(shared_dim, attention_heads, dropout)
            self.cross_tv = CrossModalInteraction(shared_dim, attention_heads, dropout)
            self.cross_av = CrossModalInteraction(shared_dim, attention_heads, dropout)
            self.cross_tav: TriModalInteraction | None = (
                TriModalInteraction(shared_dim, attention_heads, dropout) if use_trimodal else None
            )
            num_interactions = 4 if use_trimodal else 3
            self.interaction_estimator: InteractionEstimator | None = InteractionEstimator(
                shared_dim,
                num_interactions,
                mlp_hidden=interaction_mlp_hidden,
                dropout=dropout,
                condition_on_reliability=condition_interaction_on_reliability,
            )
            self.interaction_names = (
                (*PAIRWISE_INTERACTIONS, TRIMODAL_INTERACTION) if use_trimodal else PAIRWISE_INTERACTIONS
            )
        else:
            self.cross_ta = None
            self.cross_tv = None
            self.cross_av = None
            self.cross_tav = None
            num_interactions = 0
            self.interaction_estimator = None
            self.interaction_names = ()
        self.num_interactions = num_interactions

        # Reliability estimator (skipped when use_reliability_gate=False)
        self.reliability: ReliabilityEstimator | None = (
            ReliabilityEstimator(
                shared_dim,
                mlp_hidden=reliability_mlp_hidden,
                dropout=dropout,
                num_quality_features_per_modality=num_quality_features_per_modality,
            ) if use_reliability_gate else None
        )

        # Fusion + prediction
        self.fusion = HybridFusion(shared_dim, dropout=dropout, ffn_multiplier=ffn_multiplier)
        self.head = PredictionHead(shared_dim, task=task, num_classes=num_classes, dropout=dropout)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        task: str,
        num_classes: int = 1,
    ) -> "XMoFE":
        """Instantiate from a parsed ``configs/models/xmofe.yaml`` dict.

        Per-modality input dims, task, and num_classes come from the dataset
        manifest at construction time; everything else is read from config.
        """
        rel = config.get("reliability") or {}
        inter = config.get("interaction") or {}
        text_cfg = config.get("text") or {}
        return cls(
            text_dim=text_dim,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            shared_dim=config.get("shared_dim", 256),
            attention_heads=config.get("attention_heads", 4),
            dropout=config.get("dropout", 0.2),
            use_trimodal=inter.get("use_trimodal", True),
            task=task,
            num_classes=num_classes,
            num_quality_features_per_modality=(
                rel.get("num_quality_features", 0) if rel.get("use_quality_features", False) else 0
            ),
            ffn_multiplier=config.get("ffn_multiplier", 4),
            reliability_mlp_hidden=rel.get("mlp_hidden", 256),
            interaction_mlp_hidden=inter.get("mlp_hidden", 256),
            condition_interaction_on_reliability=inter.get("condition_on_reliability", False),
            text_encoder_finetune=text_cfg.get("finetune", False),
            text_encoder_name=text_cfg.get("encoder_name", "answerdotai/ModernBERT-base"),
            text_encoder_max_length=text_cfg.get("max_length", 128),
            text_encoder_trainable_layers=text_cfg.get("trainable_layers", "last_n"),
            text_encoder_last_n=text_cfg.get("last_n", 4),
        )

    def forward(
        self,
        text: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        visual: torch.Tensor | None = None,
        text_length: torch.Tensor | None = None,
        audio_length: torch.Tensor | None = None,
        visual_length: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
        transcripts: list[str] | None = None,
        return_intermediates: bool = True,
    ) -> XMoFEOutput:
        # When the model owns a trainable text encoder and transcripts are
        # provided, encode them in-graph and override the cached text tensor
        # + length so all downstream stages see the freshly-encoded features.
        # Backward then flows through the encoder. When no transcripts are
        # given, fall back to the cached path (`text`, `text_length`).
        if self.text_encoder is not None and transcripts is not None:
            text, text_length = self.text_encoder(transcripts)
        if text is None or audio is None or visual is None:
            raise ValueError(
                "XMoFE.forward needs all three modality tensors. Either pass "
                "cached features or pass transcripts (with text_encoder enabled)."
            )

        # 1. Project to shared dim
        Z_t = self.proj_text(text)
        Z_a = self.proj_audio(audio)
        Z_v = self.proj_visual(visual)

        # 2. Attention pool → modality summaries + temporal explanations
        z_t, alpha_t = self.pool_text(Z_t, text_length)
        z_a, alpha_a = self.pool_audio(Z_a, audio_length)
        z_v, alpha_v = self.pool_visual(Z_v, visual_length)

        # 3. Reliability scores → modality-level explanation
        if self.reliability is not None:
            r = self.reliability(z_t, z_a, z_v, quality=quality)    # (B, 3)
        else:
            # xmofe_no_reliability ablation: fixed uniform 1/3 weighting
            r = z_t.new_full((z_t.size(0), 3), 1.0 / 3.0)
        r_t, r_a, r_v = r[:, 0:1], r[:, 1:2], r[:, 2:3]

        # 4. Cross-modal interactions
        if self.use_interaction_block:
            c_ta = self.cross_ta(Z_t, Z_a, text_length, audio_length)
            c_tv = self.cross_tv(Z_t, Z_v, text_length, visual_length)
            c_av = self.cross_av(Z_a, Z_v, audio_length, visual_length)
            if self.cross_tav is not None:
                c_tav = self.cross_tav(Z_t, Z_a, Z_v, text_length, audio_length, visual_length)
                interaction_vecs: tuple[torch.Tensor, ...] = (c_ta, c_tv, c_av, c_tav)
            else:
                interaction_vecs = (c_ta, c_tv, c_av)
            # 5. Interaction-contribution weights → interaction-level explanation.
            # When the estimator is conditioned on reliability, feed r so the
            # interaction weights can adapt to per-sample modality reliability.
            if self.condition_interaction_on_reliability:
                lam = self.interaction_estimator(*interaction_vecs, reliability=r)
            else:
                lam = self.interaction_estimator(*interaction_vecs)     # (B, K)
            i = sum(lam[:, k:k + 1] * v for k, v in enumerate(interaction_vecs))
        else:
            # xmofe_no_interaction ablation: drop cross-modal evidence
            interaction_vecs = ()
            lam = z_t.new_full((z_t.size(0), 1), 1.0)               # placeholder
            i = z_t.new_zeros(z_t.shape)

        # 6. Hybrid fusion
        u = r_t * z_t + r_a * z_a + r_v * z_v                       # (B, D)
        h_fused = self.fusion(u, i)

        # 7. Prediction
        prediction = self.head(h_fused)

        return XMoFEOutput(
            prediction=prediction,
            reliability=r,
            interactions=lam,
            temporal_attention={"text": alpha_t, "audio": alpha_a, "visual": alpha_v},
            interaction_names=self.interaction_names if self.interaction_names else ("none",),
            pooled_modalities=(
                {"text": z_t, "audio": z_a, "visual": z_v} if return_intermediates else None
            ),
            interaction_vectors=(
                dict(zip(self.interaction_names, interaction_vecs))
                if (return_intermediates and self.interaction_names) else None
            ),
            fused=h_fused if return_intermediates else None,
        )
