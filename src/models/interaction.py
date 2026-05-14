"""Cross-modal interaction blocks and the interaction-contribution estimator.

Three pieces:

* :class:`CrossModalBlock` — multi-head cross-attention with pre-LN and
  residual. Q comes from one modality, K/V from another (or two concatenated).
* :class:`CrossModalInteraction` — wraps a block + attention pool to produce a
  single interaction vector ``c_XY ∈ R^D`` per sample.
* :class:`TriModalInteraction` — same idea, but K/V is the concatenation of two
  modalities so the query attends to both at once (yields ``c_TAV``).
* :class:`InteractionEstimator` — MLP + softmax over the interaction vectors,
  producing the contribution weights ``λ`` that double as the
  interaction-level explanation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention_pooling import AttentionPool, make_padding_mask


class CrossModalBlock(nn.Module):
    """Single cross-attention layer with pre-LN and residual on the query side."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(
            self.norm_q(q),
            kv_n, kv_n,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        return q + self.dropout(attn_out)


class CrossModalInteraction(nn.Module):
    """``Q`` (length ``L_q``) attends to ``KV`` (length ``L_kv``) → pooled to (B, D)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = CrossModalBlock(dim, num_heads, dropout)
        self.pool = AttentionPool(dim)

    def forward(
        self,
        q: torch.Tensor, kv: torch.Tensor,
        q_lengths: torch.Tensor, kv_lengths: torch.Tensor,
    ) -> torch.Tensor:
        kv_padding_mask = make_padding_mask(kv_lengths, kv.size(1))
        # Substitute an all-False mask for samples where KV is entirely missing,
        # so MultiheadAttention doesn't NaN. The pooled output for these rows is
        # essentially noise; downstream interaction weights (λ) will learn to
        # suppress them, and AttentionPool's missing-token covers any rows with
        # missing query.
        all_masked = kv_lengths == 0
        if all_masked.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked] = False

        attended = self.block(q, kv, kv_padding_mask)
        c, _ = self.pool(attended, q_lengths)
        return c


class TriModalInteraction(nn.Module):
    """``Q`` attends to ``[KV1; KV2]`` along the sequence axis → pooled to (B, D)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = CrossModalBlock(dim, num_heads, dropout)
        self.pool = AttentionPool(dim)

    def forward(
        self,
        q: torch.Tensor,
        kv1: torch.Tensor, kv2: torch.Tensor,
        q_lengths: torch.Tensor,
        kv1_lengths: torch.Tensor, kv2_lengths: torch.Tensor,
    ) -> torch.Tensor:
        kv = torch.cat([kv1, kv2], dim=1)                          # (B, L1+L2, D)
        mask1 = make_padding_mask(kv1_lengths, kv1.size(1))
        mask2 = make_padding_mask(kv2_lengths, kv2.size(1))
        kv_padding_mask = torch.cat([mask1, mask2], dim=1)         # (B, L1+L2)

        all_masked = (kv1_lengths == 0) & (kv2_lengths == 0)
        if all_masked.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked] = False

        attended = self.block(q, kv, kv_padding_mask)
        c, _ = self.pool(attended, q_lengths)
        return c


VALID_GATING = ("softmax", "sigmoid")


class InteractionEstimator(nn.Module):
    """Interaction-level explanation weights ``λ``.

    Two gating modes:

    * **``softmax``** (default, original behaviour): ``λ = softmax(MLP_I(...))``.
      Sums to 1 across interactions. Forces a zero-sum competition between
      the pairwise (``c_TA``, ``c_TV``, ``c_AV``) and trimodal (``c_TAV``)
      pathways. In practice this **collapses to ~99.95% on the trimodal
      term** during X-MoFE training — the model effectively bypasses the
      pairwise interactions entirely.

    * **``sigmoid``**: ``λ = σ(MLP_I(...))``. Each interaction gets an
      independent gate in ``[0, 1]``; no zero-sum constraint. The model
      can use multiple pathways together at full strength without the
      softmax-induced collapse. Recommended replacement for production
      training; preserves softmax for ablation studies.

    When ``condition_on_reliability=True``, the per-modality reliability
    ``r ∈ R^3`` is concatenated to the MLP input before gating so ``λ``
    can adapt to per-sample modality reliability.
    """

    NUM_RELIABILITY_INPUTS = 3   # text, audio, visual

    def __init__(
        self,
        shared_dim: int,
        num_interactions: int,
        mlp_hidden: int = 256,
        dropout: float = 0.2,
        condition_on_reliability: bool = False,
        gating: str = "softmax",
    ) -> None:
        super().__init__()
        if gating not in VALID_GATING:
            raise ValueError(
                f"gating must be one of {VALID_GATING}; got {gating!r}"
            )
        self.num_interactions = num_interactions
        self.condition_on_reliability = condition_on_reliability
        self.gating = gating
        input_dim = shared_dim * num_interactions
        if condition_on_reliability:
            input_dim += self.NUM_RELIABILITY_INPUTS
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_interactions),
        )

    def forward(
        self,
        *interaction_vectors: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(interaction_vectors) != self.num_interactions:
            raise ValueError(
                f"expected {self.num_interactions} interaction vectors, "
                f"got {len(interaction_vectors)}"
            )
        x = torch.cat(interaction_vectors, dim=-1)
        if self.condition_on_reliability:
            if reliability is None:
                raise ValueError(
                    "condition_on_reliability=True but no reliability tensor was passed; "
                    "call forward(... , reliability=r)"
                )
            if reliability.shape[-1] != self.NUM_RELIABILITY_INPUTS:
                raise ValueError(
                    f"reliability must have last dim {self.NUM_RELIABILITY_INPUTS}; "
                    f"got shape {tuple(reliability.shape)}"
                )
            x = torch.cat([x, reliability], dim=-1)
        logits = self.mlp(x)
        if self.gating == "softmax":
            return F.softmax(logits, dim=-1)
        # sigmoid — each interaction independent in [0, 1]
        return torch.sigmoid(logits)
