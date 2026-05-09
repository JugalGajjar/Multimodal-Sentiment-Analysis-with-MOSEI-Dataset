"""One-shot patch: enrich existing ``<split>.pt`` manifests with raw text.

The Phase-2 manifests (produced by ``merge_cached_features.py``) carry only
the per-modality feature cache references and the rich-label dicts — they
*don't* carry the original transcripts. End-to-end fine-tuning of the text
encoder needs raw text at training time, so this script reads the source
``data/interim/<dataset>/metadata.jsonl`` files and adds three fields to
each manifest in place:

    transcripts:        list[str]   # one per sample, in sample_id order
    speaker_ids:        list[str|None]
    dialogue_ids:       list[str|None]
    utterance_indices:  list[int|None]

The first is needed by Phase 1 (text fine-tuning); the next three are
needed by Phase 3 (MELD dialogue context). Including all of them now means
we patch once instead of twice.

Idempotent: re-running on already-patched manifests is a no-op (writes the
same content). Safe to interleave with other work.

Usage
-----
    python scripts/data/patch_manifests_with_text.py
    python scripts/data/patch_manifests_with_text.py --dataset meld
    python scripts/data/patch_manifests_with_text.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patch_manifests")

DATASETS = ("ch_sims", "meld", "mosei")
SPLITS = ("train", "val", "test")
EXTRA_FIELDS = ("transcripts", "speaker_ids", "dialogue_ids", "utterance_indices")


def load_metadata_index(path: Path) -> dict[str, dict]:
    """Build a sample_id → metadata-row index from a JSONL file."""
    index: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[row["sample_id"]] = row
    return index


def patch_manifest(
    manifest_path: Path,
    metadata_index: dict[str, dict],
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Patch one ``<split>.pt`` manifest in place. Returns counts for reporting."""
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    sample_ids: list[str] = list(manifest["sample_ids"])
    n = len(sample_ids)

    transcripts: list[str] = []
    speakers: list[str | None] = []
    dialogues: list[str | None] = []
    utt_idxs: list[int | None] = []

    missing_in_metadata: list[str] = []
    missing_transcript: list[str] = []
    for sid in sample_ids:
        row = metadata_index.get(sid)
        if row is None:
            missing_in_metadata.append(sid)
            transcripts.append("")
            speakers.append(None)
            dialogues.append(None)
            utt_idxs.append(None)
            continue
        transcript = row.get("transcript") or ""
        if not transcript:
            missing_transcript.append(sid)
        transcripts.append(transcript)
        speakers.append(row.get("speaker_id"))
        dialogues.append(row.get("dialogue_id"))
        utt_idxs.append(row.get("utterance_index"))

    # Don't bother re-writing if every value already matches what's there
    # — supports idempotent re-runs.
    already_patched = all(
        manifest.get(field) is not None for field in EXTRA_FIELDS
    )
    will_change = not already_patched or any(
        manifest.get(field) != val
        for field, val in zip(
            EXTRA_FIELDS,
            (transcripts, speakers, dialogues, utt_idxs),
        )
    )

    if not will_change:
        log.info("  %s already patched — skipping write", manifest_path.name)
    elif dry_run:
        log.info(
            "  %s dry-run: would add transcripts (%d), speakers (%d non-null), "
            "dialogues (%d non-null), utt_indices (%d non-null)",
            manifest_path.name,
            sum(1 for t in transcripts if t),
            sum(1 for s in speakers if s is not None),
            sum(1 for d in dialogues if d is not None),
            sum(1 for u in utt_idxs if u is not None),
        )
    else:
        manifest["transcripts"] = transcripts
        manifest["speaker_ids"] = speakers
        manifest["dialogue_ids"] = dialogues
        manifest["utterance_indices"] = utt_idxs
        torch.save(manifest, manifest_path)
        log.info(
            "  %s patched: %d samples, %d non-empty transcripts",
            manifest_path.name, n, sum(1 for t in transcripts if t),
        )

    return {
        "n_samples": n,
        "missing_in_metadata": len(missing_in_metadata),
        "missing_transcript": len(missing_transcript),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--interim-root", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = DATASETS if args.dataset == "all" else (args.dataset,)
    rc = 0

    for ds in targets:
        meta_path = args.interim_root / ds / "metadata.jsonl"
        if not meta_path.exists():
            log.error("metadata not found: %s — skipping %s", meta_path, ds)
            rc = 2
            continue
        log.info("dataset=%s  metadata=%s", ds, meta_path)
        index = load_metadata_index(meta_path)
        log.info("  %d samples in metadata", len(index))

        for split in SPLITS:
            manifest_path = args.processed_root / ds / f"{split}.pt"
            if not manifest_path.exists():
                log.warning("  manifest missing: %s — skipping", manifest_path)
                continue
            stats = patch_manifest(manifest_path, index, dry_run=args.dry_run)
            if stats["missing_in_metadata"]:
                log.warning(
                    "    %d/%d sample_ids not found in metadata — empty placeholders inserted",
                    stats["missing_in_metadata"], stats["n_samples"],
                )
            if stats["missing_transcript"]:
                log.info(
                    "    %d/%d samples have empty transcripts (informational)",
                    stats["missing_transcript"], stats["n_samples"],
                )

    return rc


if __name__ == "__main__":
    sys.exit(main())
