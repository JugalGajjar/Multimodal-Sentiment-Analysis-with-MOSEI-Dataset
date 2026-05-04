# X-MoFE: Faithful and Reliability-Aware Multimodal Fusion for Emotion and Sentiment Understanding

X-MoFE is an explainable multimodal fusion architecture for emotion and sentiment understanding. It learns
which modality to trust on a per-sample basis, which cross-modal interactions drive a prediction, and whether
those explanations are faithful to the model's actual sensitivity.

> Project status: in active development. This repository is being upgraded from a preliminary preprint
> ([`legacy/`](legacy/), [`paper/old/`](paper/old/)) into a new method paper. See
> [`X-MoFE_Research_Project_Specification.md`](X-MoFE_Research_Project_Specification.md) for the full
> research plan.

---

## Key ideas

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

- **Multimodal fusion**: MulT, Self-MM, MISA, Dynamic Fusion Graph (MAG-BERT as backup)
- **VLM / MLLM (zero-/few-shot, inference only)**: Qwen2.5-VL, LLaVA-OneVision

---

## Repository layout

```
.
├── configs/         # Hydra/YAML configs (datasets, encoders, models, experiments, vlms)
├── data/            # raw / interim / processed / external feature caches (gitignored)
├── src/
│   ├── data/             # dataset loading
│   ├── encoders/         # frozen ModernBERT / WavLM / VideoMAEv2 wrappers
│   ├── models/           # X-MoFE + baselines + ablations
│   ├── losses/           # task, reliability, faithfulness, stability, entropy
│   ├── training/         # trainer, evaluator, checkpointing
│   ├── evaluation/       # explanation metrics, deletion/insertion, reliability alignment
│   ├── robustness/       # missing- and noisy-modality protocols
│   ├── explainability/   # explanation outputs and analysis
│   ├── vlms/             # Qwen / LLaVA-OneVision inference helpers
│   └── utils/            # logging, seeding, IO
├── scripts/         # data prep, feature extraction, training, evaluation, reporting
├── experiments/     # per-experiment outputs (gitignored)
├── results/         # collected result tables (gitignored)
├── checkpoints/     # model checkpoints (gitignored)
├── logs/            # training/eval logs (gitignored)
├── notebooks/       # exploratory analysis
├── tests/           # unit tests
├── docs/            # internal docs
├── paper/
│   ├── emnlp2026/   # in-progress paper sources
│   └── old/         # preserved preprint sources (gitignored)
└── legacy/          # preserved code from the preliminary preprint pipeline
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

The `xmofe` package is installed in editable mode so `import src.models.xmofe` resolves while you iterate.

---

## Compute strategy

The project is intentionally compute-aware. Available hardware:

- M4 Pro MacBook Pro, 48 GB unified memory
- Google Colab free tier
- Kaggle P100, ~30 hours / week

Strategy: freeze all encoders, cache features once per dataset, train only fusion + explanation modules, run 3 seeds for the final X-MoFE and key controlled variants (1 seed elsewhere), and use VLMs as inference-only baselines on stratified subsets.

---

## Legacy work

The preliminary preprint (transformer cross-attention fusion on CMU-MOSEI with BERT-CLS / averaged COVAREP / averaged OpenFace features) is preserved under [`legacy/`](legacy/) for reproducibility and as a starting reference. The accompanying paper sources live in [`paper/old/`](paper/old/) (gitignored locally).

---

## License

Released under the [MIT License](LICENSE).
