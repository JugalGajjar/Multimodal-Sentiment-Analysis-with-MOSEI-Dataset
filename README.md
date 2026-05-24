# X-MoFE: Faithful and Reliability-Aware Multimodal Fusion for Emotion and Sentiment Understanding

X-MoFE is an explainable multimodal fusion architecture for emotion and sentiment understanding. It learns
which modality to trust on a per-sample basis, which cross-modal interactions drive a prediction, and whether
those explanations are faithful to the model's actual sensitivity.

> **Status:** the repository skeleton is in place; implementation is being built phase by phase. The
> preliminary preprint code is preserved under [`legacy/`](legacy/).

---

<!-- ## Key ideas

- **Frozen modern unimodal encoders** — text: ModernBERT-base, audio: WavLM Base+, visual: VideoMAEv2-base. The learnable contribution is the fusion architecture, not encoder fine-tuning.
- **Sample-wise modality reliability** — a reliability estimator predicts how much each modality should be trusted for a given sample.
- **Cross-modal interaction attribution** — pairwise (T-A, T-V, A-V) and tri-modal interactions are weighted explicitly.
- **Three-level explanations** — modality, temporal, and interaction outputs, each tied to the prediction.
- **Faithfulness-centered training** — auxiliary losses align reliability scores with prediction sensitivity, encourage stable explanations under perturbation, and control reliability entropy.

---

## Datasets

| Dataset    | Role                                                                 |
|------------|----------------------------------------------------------------------|
| CMU-MOSEI  | Large-scale multimodal sentiment / emotion benchmark                 |
| MELD       | Conversational emotion recognition                                   |
| CH-SIMS    | Independent unimodal annotations — used for reliability supervision  |

## Baselines

- **Multimodal fusion**: MulT, Self-MM, MISA, Dynamic Fusion Graph
- **VLM / MLLM (zero-/few-shot, inference only)**: Qwen2.5-VL, LLaVA-OneVision

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt              # pip dependencies
├── environment.yml               # conda environment
├── pyproject.toml                # package metadata + ruff/black/mypy/pytest config
├── setup.py                      # editable-install shim
│
├── configs/                      # YAML configs
│   ├── datasets/
│   ├── encoders/
│   ├── models/
│   ├── experiments/
│   └── vlms/
│
├── data/                         # contents gitignored
│   ├── raw/                      # raw downloads
│   ├── interim/                  # intermediate artifacts
│   ├── processed/                # cached encoder features
│   └── external/                 # third-party assets
│
├── src/
│   ├── data/                     # dataset loading and split utilities
│   ├── encoders/                 # frozen ModernBERT / WavLM / VideoMAEv2 wrappers
│   ├── models/
│   │   ├── baselines/            # MulT, Self-MM, MISA, Dynamic Fusion Graph
│   │   └── ablations/            # X-MoFE component-removal variants
│   ├── losses/                   # task, reliability, faithfulness, stability, entropy
│   ├── training/                 # trainer, evaluator, checkpointing
│   ├── evaluation/               # explanation metrics, deletion/insertion, reliability alignment
│   ├── robustness/               # missing- and noisy-modality protocols
│   ├── explainability/           # explanation outputs and analysis
│   ├── vlms/                     # Qwen2.5-VL / LLaVA-OneVision inference helpers
│   └── utils/                    # logging, seeding, IO
│
├── scripts/
│   ├── data/                     # dataset preparation
│   ├── features/                 # feature extraction (ModernBERT/WavLM/VideoMAEv2)
│   ├── train/                    # training entry points
│   ├── evaluate/                 # evaluation runs
│   ├── vlms/                     # VLM inference runners
│   └── reporting/                # table/figure generation
│
├── notebooks/                    # exploratory analysis
├── tests/                        # unit tests
├── docs/                         # internal docs
│
└── legacy/                       # preserved preliminary preprint pipeline
    ├── main.py
    ├── config.py
    ├── src/                      # old data / models / training / utils
    ├── scripts/                  # old train_unimodal / train_multimodal / evaluate
    └── data/processed/           # old BERT-CLS / averaged-COVAREP / averaged-OpenFace caches
```

---

## Setup

### pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### conda

```bash
conda env create -f environment.yml
conda activate xmofe
pip install -e .
```

The `xmofe` package is installed in editable mode so imports under `src/` resolve as you develop.

---

## Compute strategy

The project is intentionally compute-aware. Available hardware:

- M4 Pro MacBook Pro, 48 GB unified memory
- Google Colab free tier
- Kaggle P100, ~30 hours / week

Strategy: freeze all encoders, cache features once per dataset, train only fusion + explanation modules, run 3 seeds for the final X-MoFE and key controlled variants (1 seed elsewhere), and use VLMs as inference-only baselines on stratified subsets.

--- -->

## Legacy work

The preliminary preprint — transformer cross-attention fusion on CMU-MOSEI with BERT-CLS text embeddings, time-averaged COVAREP audio features, and time-averaged OpenFace visual features — is preserved under [`legacy/`](legacy/), including the original processed feature pickles. The legacy paper preprint can be found at [`arXiv:2505.06110`](https://arxiv.org/abs/2505.06110).

---

## License

Released under the [MIT License](LICENSE).
