"""Stratified subsampling for compute-bounded VLM evaluation.

Spec §18.5: "If full test-set inference is too expensive, use a fixed
stratified subset: 500 to 1000 test samples per dataset."

The implementation preserves per-class proportions (rounded up to ≥1
per class) and uses a configurable seed so re-runs are reproducible.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


def stratified_subsample(
    samples: Sequence[T],
    label_fn: Callable[[T], object],
    n_total: int,
    seed: int = 42,
) -> list[T]:
    """Sample ≤ ``n_total`` items preserving the label distribution of ``samples``.

    Each label class is allocated ``round(n_total * |class| / |total|)``
    samples, with a minimum of 1 per class so rare classes still appear.
    If ``n_total`` ≥ total, returns all samples (in original order).
    """
    items = list(samples)
    total = len(items)
    if total == 0:
        return []
    if n_total >= total:
        return items

    by_label: dict[object, list[T]] = defaultdict(list)
    for item in items:
        by_label[label_fn(item)].append(item)

    rng = random.Random(seed)

    quotas: dict[object, int] = {}
    for label, group in by_label.items():
        quota = max(1, round(n_total * len(group) / total))
        quotas[label] = min(quota, len(group))

    # If our minimum-1 rule pushes us over n_total, trim from the largest
    # classes first so rare classes keep their representation.
    overshoot = sum(quotas.values()) - n_total
    if overshoot > 0:
        ordered = sorted(quotas.items(), key=lambda kv: -len(by_label[kv[0]]))
        for label, _ in ordered:
            if overshoot <= 0:
                break
            slack = quotas[label] - 1
            take = min(slack, overshoot)
            quotas[label] -= take
            overshoot -= take

    subset: list[T] = []
    for label, group in by_label.items():
        rng.shuffle(group)
        subset.extend(group[: quotas[label]])
    return subset
