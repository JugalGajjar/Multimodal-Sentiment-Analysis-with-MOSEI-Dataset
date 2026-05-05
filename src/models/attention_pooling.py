"""Single-query attention pooling that doubles as a temporal explanation.

The attention weights ``α`` returned alongside the pooled vector ``z`` are
exactly the per-position importance scores that X-MoFE exposes as the
temporal explanation. A single attention head is used deliberately: it
yields one clean weight per position rather than a per-head average that's
harder to interpret faithfully.

When an entire modality is missing for a sample (``length == 0``), naive
softmax over an all-padded sequence produces NaN. This module substitutes a
learned "missing modality" embedding for those rows and zeros their
attention weights, keeping the architecture differentiable in robustness
experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Return a (B, max_length) bool tensor: True at padded (invalid) positions.

    Matches the convention expected by ``nn.MultiheadAttention``'s
    ``key_padding_mask`` argument.
    """
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    return ~valid


class AttentionPool(nn.Module):
    """Pool a (B, L, D) sequence to (B, D) with a learned single-head query.

    Returns:
        z: (B, D) pooled embedding.
        alpha: (B, L) per-position attention weights — these are explanation
               scores callers can downstream-interpret.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)
        self.missing_token = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.missing_token, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, d = x.shape

        padding_mask = make_padding_mask(lengths, l)              # (B, L)
        is_missing = lengths == 0                                  # (B,)

        # Avoid all-True padding masks, which would NaN the softmax. For
        # entirely-missing rows we run attention against the (zero) padding
        # but discard the result via ``missing_token`` afterwards.
        safe_padding_mask = padding_mask.clone()
        safe_padding_mask[is_missing] = False

        q = self.query.expand(b, -1, -1)                           # (B, 1, D)
        out, attn = self.attn(
            q, x, x,
            key_padding_mask=safe_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        z = out.squeeze(1)                                         # (B, D)
        alpha = attn.squeeze(1) if attn.dim() == 3 else attn       # (B, L)

        if is_missing.any():
            z = torch.where(is_missing.unsqueeze(1), self.missing_token.expand(b, d), z)
            alpha = alpha.masked_fill(is_missing.unsqueeze(1), 0.0)

        return z, alpha
