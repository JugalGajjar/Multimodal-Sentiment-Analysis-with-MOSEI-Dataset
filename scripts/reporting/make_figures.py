"""Build Figure 2 (reliability behavior under corruption) from spec §20.8.

Reads ``results/robustness_*.json``, plots a grouped bar chart of mean
reliability per modality across (clean, single-missing, single-noisy)
conditions. Saves to ``paper/emnlp2026/figures/`` as both PDF (paper) and
PNG (preview) at 300 dpi.

Figure 1 (architecture diagram) and Figure 3 (qualitative case study) are
authoring tasks — not aggregation — so they're not generated here.

Examples
--------
    python scripts/reporting/make_figures.py
    python scripts/reporting/make_figures.py --dataset mosei --variant xmofe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("make_figures")


def _select_record(records: list[dict], dataset: str, variant: str | None) -> dict | None:
    matches = [r for r in records if r.get("experiment") == dataset]
    if variant is not None:
        matches = [r for r in matches if r.get("variant") == variant]
    return matches[0] if matches else None


def _reliability_for(condition: dict) -> tuple[float, float, float] | None:
    """Pull (r_T, r_A, r_V) from one robustness condition's metrics block."""
    keys = ("reliability_mean_text", "reliability_mean_audio", "reliability_mean_visual")
    if not all(k in condition for k in keys):
        return None
    return tuple(float(condition[k]) for k in keys)  # type: ignore[return-value]


def make_reliability_under_corruption(
    record: dict,
    output_dir: Path,
    dataset: str,
    variant: str,
) -> None:
    """Bar chart: mean reliability across conditions × modalities."""
    conditions: list[tuple[str, tuple[float, float, float]]] = []
    missing = record.get("missing_modality") or {}
    noisy = record.get("noisy_modality") or {}

    show_keys = [
        ("clean", missing.get("clean")),
        ("text missing", missing.get("text_missing")),
        ("audio missing", missing.get("audio_missing")),
        ("visual missing", missing.get("visual_missing")),
    ]
    for key, m in noisy.items():
        show_keys.append((f"noisy {key}", m))

    for label, m in show_keys:
        if m is None:
            continue
        rels = _reliability_for(m)
        if rels is None:
            continue
        conditions.append((label, rels))

    if not conditions:
        log.warning("no usable reliability values for %s/%s; skipping figure", dataset, variant)
        return

    labels = [c[0] for c in conditions]
    r_text = np.array([c[1][0] for c in conditions])
    r_audio = np.array([c[1][1] for c in conditions])
    r_visual = np.array([c[1][2] for c in conditions])

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.0), 4.0))
    width = 0.27
    x = np.arange(len(conditions))
    ax.bar(x - width, r_text, width, label="text",     color="#1f77b4")
    ax.bar(x,         r_audio, width, label="audio",   color="#ff7f0e")
    ax.bar(x + width, r_visual, width, label="visual", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean reliability r_m")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Reliability under corruption — {dataset.upper()} / {variant}")
    ax.legend(loc="upper right", ncol=3, frameon=False)
    ax.axhline(1.0 / 3.0, color="gray", linestyle=":", linewidth=0.7)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"reliability_under_corruption_{dataset}_{variant}.pdf"
    png_path = output_dir / f"reliability_under_corruption_{dataset}_{variant}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    log.info("wrote %s and %s", pdf_path, png_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collected", type=Path, default=REPO_ROOT / "results" / "collected.json")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "paper" / "emnlp2026" / "figures")
    args = parser.parse_args()

    if not args.collected.exists():
        log.error("collected.json not found at %s. Run scripts/reporting/collect_results.py first.",
                  args.collected)
        sys.exit(2)
    bundle = json.loads(args.collected.read_text(encoding="utf-8"))
    records = bundle.get("robustness", [])
    if not records:
        log.warning("no robustness records to plot")
        return

    datasets = sorted({r.get("experiment") for r in records if r.get("experiment")}) \
        if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        rec = _select_record(records, dataset, args.variant)
        if rec is None:
            log.warning("no record for dataset=%s variant=%s", dataset, args.variant)
            continue
        variant = rec.get("variant") or "xmofe"
        make_reliability_under_corruption(rec, args.output_dir, dataset, variant)


if __name__ == "__main__":
    main()
