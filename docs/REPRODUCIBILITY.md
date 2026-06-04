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

