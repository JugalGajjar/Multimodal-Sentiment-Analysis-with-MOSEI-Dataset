# Datasets

X-MoFE evaluates on three datasets: **CMU-MOSEI**, **MELD**, and **CH-SIMS**.
This doc walks through download → preparation → validation. After this phase
you will have, for each dataset:

```
data/raw/<dataset>/        # raw mp4s, label files, etc. (gitignored)
data/interim/<dataset>/
    metadata.jsonl         # one Sample record per utterance/segment
    splits.json            # {"train": [...], "val": [...], "test": [...]}
```

The `Sample` schema is defined in [`src/data/schemas.py`](../src/data/schemas.py).

---

## Quick start

```bash
# 1. Download all three datasets (uses gdown / wget under the hood)
python scripts/data/download_datasets.py --dataset all

# 2. Prepare standardized metadata + splits
python scripts/data/prepare_mosei.py
python scripts/data/prepare_meld.py
python scripts/data/prepare_ch_sims.py

# 3. Validate
python scripts/data/validate_dataset.py --dataset all --check-files
```

Each `prepare_*.py` is idempotent — it overwrites `metadata.jsonl` /
`splits.json` on every run. `download_datasets.py` skips downloads that are
already present unless you pass `--force`.

---

## CMU-MOSEI

### Source

- Mirror on Google Drive (uploaded during the preprint work): <https://drive.google.com/drive/folders/1sNPjCssAiUHVNgqnMiSEKX8P2E5GhQct>
- Configured in [`configs/datasets/mosei.yaml`](../configs/datasets/mosei.yaml)

The Drive folder contains the CMU-MultimodalSDK Computational Sequence Data
(`*.csd`) files used to enumerate segments and labels. Raw `mp4` videos are
**not** included; if you have them locally, point the config's `video.root`
at their directory and `prepare_mosei.py` will populate `Sample.video_path`
with start/end times so VideoMAEv2 can slice clips during Phase 2.

### Download

```bash
pip install gdown
python scripts/data/download_datasets.py --dataset mosei
```

This drops the CSD files under `data/raw/mosei/`.

### Prepare

```bash
pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
python scripts/data/prepare_mosei.py
```

`prepare_mosei.py` uses the SDK's `cmu_mosei_std_folds` to map every
segment id to one of `train` / `val` / `test`, decodes timestamped words
into the `transcript` field, and emits per-segment 7-dim labels
(sentiment regression + 6 emotion intensities).

### Labels

| Field                            | Type    | Range / values                                |
|----------------------------------|---------|------------------------------------------------|
| `primary_label`                  | float   | sentiment regression in `[-3, 3]`              |
| `labels.sentiment_regression`    | float   | same as primary                                |
| `labels.sentiment_binary`        | int     | 0 (≤ 0) or 1 (> 0)                             |
| `labels.sentiment_7class`        | int     | rounded sentiment shifted to `0..6`            |
| `labels.emotions.{happy,sad,...}`| float   | per-emotion intensity                          |

---

## MELD

### Source

- Public mirror at the University of Michigan: <https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz>
- Configured in [`configs/datasets/meld.yaml`](../configs/datasets/meld.yaml)

The tarball expands into `MELD.Raw/` with one mp4 per utterance and three
top-level CSVs (`train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`).

### Download

```bash
python scripts/data/download_datasets.py --dataset meld
```

This runs `wget` followed by `tar -xzf` so you end up with
`data/raw/meld/MELD.Raw/`.

### Prepare

```bash
python scripts/data/prepare_meld.py
```

A handful of clips are missing from MELD's release (a known issue). By
default they're dropped; pass `--skip-missing-clips=false` to keep the rows
with `video_path=None`.

### Labels

| Field                       | Type | Values                                              |
|-----------------------------|------|-----------------------------------------------------|
| `primary_label`             | int  | emotion id (0..6)                                   |
| `labels.emotion`            | str  | `neutral, joy, sadness, anger, surprise, fear, disgust` |
| `labels.emotion_id`         | int  | as above                                            |
| `labels.sentiment`          | str  | `neutral, positive, negative`                       |
| `labels.sentiment_id`       | int  | as above                                            |
| `speaker_id`, `dialogue_id` | str  | populated for conversation-aware modeling           |

---

## CH-SIMS

### Source

- Google Drive folder: <https://drive.google.com/drive/folders/1wFvGS0ebKRvT3q6Xolot-sDtCNfz7HRA>
- Configured in [`configs/datasets/ch_sims.yaml`](../configs/datasets/ch_sims.yaml)

CH-SIMS provides per-clip *multimodal* and *unimodal* sentiment labels —
this is what makes it the linchpin for X-MoFE's reliability supervision.

### Download

```bash
python scripts/data/download_datasets.py --dataset ch_sims
```

Expected layout afterwards:

```
data/raw/ch_sims/
    label.csv
    Raw/<video_id>/<clip_id>.mp4
```

### Prepare

```bash
python scripts/data/prepare_ch_sims.py
```

The CH-SIMS `mode` column (`train` / `valid` / `test`) is mapped to our
`train` / `val` / `test` split names.

### Labels

| Field                          | Type  | Values                                |
|--------------------------------|-------|----------------------------------------|
| `primary_label`                | float | multimodal sentiment in `[-1, 1]`     |
| `labels.sentiment_M`           | float | multimodal label                      |
| `labels.sentiment_T`           | float | text-only annotation                  |
| `labels.sentiment_A`           | float | audio-only annotation                 |
| `labels.sentiment_V`           | float | visual-only annotation                |
| `labels.annotation`            | str   | coarse class (`Negative`, ..., `Positive`) |

The unimodal labels feed `L_reliability` during X-MoFE training.

---

## Validation

```bash
python scripts/data/validate_dataset.py --dataset all --check-files
```

Reports per dataset:

- per-split sample counts
- transcript length and clip duration stats
- label distribution (binned for regression, raw counts for classification)
- duplicate / split-mismatched sample ids
- missing media files (with `--check-files`)

Non-zero exit code means at least one error was detected.

---

## Adding a new dataset

1. Add a YAML config under `configs/datasets/`.
2. Append the dataset name to `VALID_DATASETS` in `src/data/schemas.py`.
3. Write `scripts/data/prepare_<dataset>.py` that emits records matching the
   `Sample` schema.
4. Document download instructions here.
