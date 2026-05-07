"""Build Table 6 (explanation faithfulness) per spec §20.6.

Aggregates the explanation-evaluation JSON files into a per-modality
faithfulness table: deletion AULC, insertion AULC, sufficiency,
comprehensiveness, plus the modality-level reliability-sensitivity
correlation.

Examples
--------
    python scripts/reporting/make_explanation_tables.py
    python scripts/reporting/make_explanation_tables.py --variant xmofe --dataset mosei
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.reporting import VALID_FORMATS, write_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("make_explanation_tables")

MODALITIES = ("text", "audio", "visual")


def _build_rows(record: dict) -> list[dict]:
    """One row per modality + a final row holding overall faithfulness."""
    rows: list[dict] = []
    for modality in MODALITIES:
        d = (record.get("deletion") or {}).get(modality, {}) or {}
        i = (record.get("insertion") or {}).get(modality, {}) or {}
        s = (record.get("sufficiency") or {}).get(modality, {}) or {}
        c = (record.get("comprehensiveness") or {}).get(modality, {}) or {}
        rows.append({
            "modality": modality,
            "deletion_aulc": d.get("aulc"),
            "insertion_aulc": i.get("aulc"),
            "sufficiency": s.get("sufficiency"),
            "comprehensiveness": c.get("comprehensiveness"),
        })

    mf = record.get("modality_faithfulness") or {}
    rows.append({
        "modality": "modality-level",
        "deletion_aulc": None,
        "insertion_aulc": None,
        "sufficiency": None,
        "comprehensiveness": None,
        "spearman_r_to_s": mf.get("spearman"),
        "kl_s_to_r": mf.get("kl_s_to_r_mean"),
    })
    return rows


def _select_record(records: list[dict], dataset: str, variant: str | None) -> dict | None:
    matches = [r for r in records if r.get("experiment") == dataset]
    if variant is not None:
        matches = [r for r in matches if r.get("variant") == variant]
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collected", type=Path, default=REPO_ROOT / "results" / "collected.json")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--format", choices=VALID_FORMATS, default="tex")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "paper" / "emnlp2026" / "tables")
    args = parser.parse_args()

    if not args.collected.exists():
        log.error("collected.json not found at %s. Run scripts/reporting/collect_results.py first.",
                  args.collected)
        sys.exit(2)
    bundle = json.loads(args.collected.read_text(encoding="utf-8"))
    records = bundle.get("explanations", [])

    datasets = sorted({r.get("experiment") for r in records if r.get("experiment")}) if args.dataset == "all" \
        else [args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        rec = _select_record(records, dataset, args.variant)
        if rec is None:
            log.warning("no explanation record for dataset=%s variant=%s", dataset, args.variant)
            continue
        rows = _build_rows(rec)
        columns = [
            "modality", "deletion_aulc", "insertion_aulc",
            "sufficiency", "comprehensiveness", "spearman_r_to_s", "kl_s_to_r",
        ]
        headers = [
            "Modality", "Deletion AUC", "Insertion AUC",
            "Sufficiency", "Comprehensiveness", "ρ(r,s)", "KL(s‖r)",
        ]
        out_path = args.output_dir / f"explanation_faithfulness_{dataset}.{args.format}"
        write_table(
            out_path, rows, columns, headers,
            caption=f"Explanation faithfulness on {dataset.upper()}.",
            label=f"tab:explanation_{dataset}",
            fmt=args.format,
        )
        log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
