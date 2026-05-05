"""Run all three modality extractors in sequence (text → audio → visual).

A thin convenience wrapper around the per-modality scripts. CLI flags are
passed through to each subscript verbatim. Failures abort the chain so the
exit code reflects the first script that failed; rerun with the same args to
resume (the per-script ``--skip-existing`` default makes already-fresh
caches no-ops, so this script is idempotent).

Examples
--------
    python scripts/features/extract_all_features.py
    python scripts/features/extract_all_features.py --dataset meld --device cpu
    python scripts/features/extract_all_features.py --dataset all --no-skip-existing
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "features"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_all")

PIPELINE = (
    ("text", "extract_text_features.py"),
    ("audio", "extract_audio_features.py"),
    ("visual", "extract_visual_features.py"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", help="Forwarded to each subscript.")
    parser.add_argument("--device", default="auto", help="Forwarded to each subscript.")
    parser.add_argument("--batch-size", type=int, default=None, help="Forwarded to each subscript.")
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip splits whose cache is already fresh (default).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-extract every cache, ignoring freshness checks.",
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _ in PIPELINE],
        action="append",
        help="Run only the named modality extractors (repeatable).",
    )
    args = parser.parse_args()

    forwarded = ["--dataset", args.dataset, "--device", args.device]
    if args.batch_size is not None:
        forwarded += ["--batch-size", str(args.batch_size)]
    forwarded.append("--skip-existing" if args.skip_existing else "--no-skip-existing")

    selected = set(args.only) if args.only else {name for name, _ in PIPELINE}
    pipeline = [(n, s) for n, s in PIPELINE if n in selected]

    log.info(
        "running extractors: %s",
        ", ".join(name for name, _ in pipeline),
    )

    overall_start = time.time()
    for modality, script in pipeline:
        path = SCRIPTS_DIR / script
        cmd = [sys.executable, str(path), *forwarded]
        log.info("[%s] $ %s", modality, " ".join(cmd))
        start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start
        if result.returncode != 0:
            log.error("[%s] failed (exit %d) after %.1fs", modality, result.returncode, elapsed)
            sys.exit(result.returncode)
        log.info("[%s] done in %.1fs", modality, elapsed)

    log.info("all extractors finished in %.1fs", time.time() - overall_start)


if __name__ == "__main__":
    main()
