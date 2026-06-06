# Reproducibility

This repository contains the EviRank model implementation without raw data or trained checkpoints.

## Main Configuration

The standard paper configuration is:

- Encoder: SPECTER2-style encoder with proximity adapter.
- Evidence papers per reviewer: top `L=5` in the locked paper runs.
- Attention temperature: `0.2`.
- Reranker: dual-head MLP, hidden size `128`, dropout `0.1`.
- Training: query-level 5-fold out-of-fold evaluation.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Epochs: `200`.
- Negative sampling: mixed lexical hard negatives and random negatives.
- Score fusion: reranker coefficient `0.2`, Stage-1 anchor coefficient `0.8`.
- Feature-group scaling after per-query z-score normalization:
  `interaction=1.0, profile=1.0, dense=0.75, lexical=1.0`.
- Vector-block scaling after per-query z-score normalization:
  `profile=1.0, attn=1.0, hadamard=1.25, absdiff=1.0`.
- Seed: `42`.

See `configs/evirank_locked_standard.json` for a machine-readable summary.

## Reproducing a Run

1. Prepare the processed data files described in `docs/DATA_FORMAT.md`.
2. Install dependencies from `requirements.txt`.
3. Run:

```bash
bash scripts/run_evirank_cv.sh <dataset>
```

Outputs are written to `output/artifacts/<dataset>_evirank_oof_<timestamp>/`.

## Feature-Knockout Sensitivity

The paper reports frozen feature-knockout sensitivity rather than retrained
ablation. This keeps the trained full-model checkpoints fixed and masks one
feature group at inference time:

```bash
CKPT_DIR=output/artifacts/<dataset>_evirank_oof_<timestamp>/checkpoints \
bash scripts/run_feature_knockout.sh <dataset>
```

The supported feature groups are:

- `explicit_interaction`
- `profile_context`
- `dense_semantic_stats`
- `lexical_matching`

Paper-aligned lightweight summaries are stored in:

- `results/evirank_results_only.csv`
- `results/evirank_feature_knockout_sensitivity.csv`
