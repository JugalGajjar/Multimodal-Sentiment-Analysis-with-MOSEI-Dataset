"""Download raw assets for the X-MoFE dataset suite.

CMU-MOSEI and CH-SIMS are pulled from Google Drive folders via ``gdown``;
MELD is fetched from its public mirror and extracted in-place.

Examples
--------
    python scripts/data/download_datasets.py --dataset all
    python scripts/data/download_datasets.py --dataset meld
    python scripts/data/download_datasets.py --dataset mosei --force

Downloads are skipped when the destination already looks populated unless
``--force`` is passed. The script does not modify or repackage any files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "datasets"

DATASETS = ("mosei", "meld", "ch_sims")


def load_config(dataset: str) -> dict:
    cfg_path = CONFIG_DIR / f"{dataset}.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_populated(directory: Path) -> bool:
    if not directory.exists():
        return False
    for entry in directory.iterdir():
        if entry.name == ".gitkeep":
            continue
        return True
    return False


def require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(
            f"Required binary {binary!r} not found on PATH. "
            f"Install it (e.g. `pip install gdown` or `brew install wget`)."
        )
    return path


def download_gdrive_folder(folder_id: str, dest: Path) -> None:
    require("gdown")
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["gdown", "--folder", "--id", folder_id, "-O", str(dest)]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def download_mosei(force: bool) -> None:
    cfg = load_config("mosei")
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"]
    if is_populated(raw_dir) and not force:
        print(f"[mosei] {raw_dir} already populated — skip (use --force to redownload).")
        return
    folder_id = cfg["download"]["folder_id"]
    print(f"[mosei] downloading Google Drive folder {folder_id} -> {raw_dir}")
    download_gdrive_folder(folder_id, raw_dir)


def download_meld(force: bool) -> None:
    cfg = load_config("meld")
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"]
    raw_root = REPO_ROOT / cfg["paths"]["raw_root"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / cfg["download"]["archive"]
    url = cfg["download"]["url"]

    if archive.exists() and not force:
        print(f"[meld] archive already at {archive} — skipping download.")
    else:
        require("wget")
        cmd = ["wget", "-O", str(archive), url]
        print(f"$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    require("tar")
    if not raw_root.exists() or force:
        cmd = ["tar", "-xzf", str(archive), "-C", str(raw_dir)]
        print(f"$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    else:
        print(f"[meld] {raw_root} already extracted — skipping outer untar.")

    # MELD.Raw.tar.gz is a tarball of tarballs: dev.tar.gz / test.tar.gz /
    # train.tar.gz contain the actual mp4 clips. Extract whichever inner
    # archives are still missing their split directory.
    inner_archives = [
        ("train.tar.gz", "train_splits"),
        ("dev.tar.gz", "dev_splits_complete"),
        ("test.tar.gz", "output_repeated_splits_test"),
    ]
    for archive_name, split_dir in inner_archives:
        archive_path = raw_root / archive_name
        split_path = raw_root / split_dir
        if split_path.exists() and any(split_path.iterdir()) and not force:
            print(f"[meld] {split_dir} already populated — skip.")
            continue
        if not archive_path.exists():
            print(f"[meld] WARNING: nested archive missing: {archive_path}")
            continue
        cmd = ["tar", "-xzf", str(archive_path), "-C", str(raw_root)]
        print(f"$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def download_ch_sims(force: bool) -> None:
    cfg = load_config("ch_sims")
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"]
    if is_populated(raw_dir) and not force:
        print(f"[ch_sims] {raw_dir} already populated — skip (use --force to redownload).")
        return
    folder_id = cfg["download"]["folder_id"]
    print(f"[ch_sims] downloading Google Drive folder {folder_id} -> {raw_dir}")
    download_gdrive_folder(folder_id, raw_dir)


DOWNLOADERS = {
    "mosei": download_mosei,
    "meld": download_meld,
    "ch_sims": download_ch_sims,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="all",
        help="Dataset to download (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the target directory already looks populated.",
    )
    args = parser.parse_args()

    targets = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in targets:
        DOWNLOADERS[name](args.force)
    print("Done.")


if __name__ == "__main__":
    main()
