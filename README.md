# EviRank

**EviRank: Evidence-Grounded Reviewer Recommendation via Dynamic Query-Specific Profiles**

- **Venue:** IEEE International Conference on Data Mining (ICDM 2026)
- **Authors:** Reyncher
- **Status:** Accepted
- **Code and supplementary materials:** this repository
- **Contact:** [xhw99493133@gmail.com](mailto:xhw99493133@gmail.com)

EviRank is an evidence-grounded retrieve-and-rerank model for reviewer
recommendation. This repository contains the implementation, evaluation tools,
locked configuration, lightweight results, and supplementary analyses reported
with the paper.

## Supplementary Materials

- [Calibration sensitivity](supplementary/S3_calibration_sensitivity.md)
- [Statistical significance](supplementary/S4_significance_tests.md)
- [Runtime and scalability](supplementary/S5_runtime_scalability.md)
- [ConfusEval annotation protocol](supplementary/S1_confuseval_annotation_protocol.md)
- [Annotation and adjudication examples](supplementary/S2_annotation_examples.md)
- [Exposure and deployment diagnostics](supplementary/S6_exposure_diagnostics.md)
- [Baseline comparability](supplementary/S7_baseline_comparability.md)
- [Complete supplementary index](supplementary/README.md)

## What Is Included

- `r2reviewer/prx_eval.py`: Stage-1 semantic retrieval with SPECTER2-style paper embeddings.
- `r2reviewer/stage1_encoder.py`: text encoding and reviewer-level aggregation utilities used by Stage 1.
- `r2reviewer/reranker_features.py`: query-specific evidence selection and feature construction.
- `r2reviewer/pairwise_reranker.py`: dual-head MLP reranker.
- `r2reviewer/cv_train_reranker.py`: query-level cross-validation training and out-of-fold inference.
- `r2reviewer/eval_feature_knockout_oof.py`: frozen feature-knockout sensitivity evaluation for saved full-model checkpoints.
- `r2reviewer/rerank_pairwise_mlp.py`: inference utility for applying a trained reranker to a candidate pool.
- `tools/eval_paper_metrics.py`: standard reviewer-ranking metrics.
- `tools/eval_confuseval.py`: hard-negative diagnostic metrics for ConfusEval-style judged pools.
- `tools/preprocess_paper_data.py`: conversion utility for qrel-style reviewer-ranking data.
- `examples/`: tiny synthetic files for checking data schemas and evaluation scripts.
- `results/`: lightweight EviRank-only result summaries and diagnostic statistics.
- `supplementary/`: paper-aligned calibration, significance, runtime,
  annotation, comparability, and exposure details.
- `MODEL_CARD.md`: intended use, limitations, and ethical notes.
- `RELEASE_CHECKLIST.md`: pre-upload safety checklist.

## Expected Data Layout

Place benchmark files under `data/` or provide paths explicitly. The processed EviRank format is:

```text
data/processed/<dataset>/
  specter_reviewers.json
  queries_for_inference.json
  gold_standard.json
```

See `docs/DATA_FORMAT.md` for schemas.

## Installation

The reported runs used Python 3.12.3, PyTorch 2.10.0 with CUDA 12.8,
Transformers 4.51.3, and one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB).

```bash
conda create -n evirank python=3.12 -y
conda activate evirank
pip install -r requirements.txt
```

The Stage-1 encoder uses HuggingFace models. For offline review, pre-download the required SPECTER2 model and adapter, or set the model paths explicitly in the run scripts.

## Run EviRank

Run one or more datasets after placing the required data files:

```bash
bash scripts/run_evirank_cv.sh kdd
bash scripts/run_evirank_cv.sh nips kdd sigir scirepeval
```

The script first computes Stage-1 semantic scores and then trains the EviRank reranker with query-level out-of-fold evaluation.

The default script arguments match the locked EviRank configuration used in the
paper: top-5 evidence papers, Stage-1 top-3 score anchoring with reranker weight
0.2, feature-group weights `interaction=1.0,profile=1.0,dense=0.75,lexical=1.0`,
and vector-block weights `profile=1.0,attn=1.0,hadamard=1.25,absdiff=1.0`.
These values are also recorded in `configs/evirank_locked_standard.json`.

## Run Frozen Feature-Knockout Diagnostics

After training a full EviRank model and saving fold checkpoints, run:

```bash
CKPT_DIR=output/artifacts/kdd_evirank_oof_<timestamp>/checkpoints \
bash scripts/run_feature_knockout.sh kdd
```

This keeps the trained full reranker fixed and masks one feature group at
inference time. It is a sensitivity diagnostic, not a retrained ablation.

## Evaluate a Saved Run

```bash
python tools/eval_paper_metrics.py \
  --runs_dir output/artifacts/<run_dir> \
  --dataset kdd \
  --gold_json data/processed/kdd/gold_standard.json \
  --map_k 5,10 \
  --p_k 5,10
```

For a ConfusEval-style hard-negative judged pool:

```bash
python tools/eval_confuseval.py \
  --benchmark_dir data/confuseval \
  --run output/evirank_confuseval.jsonl \
  --ks 1,3,5
```

## Run the Toy Example

The toy example does not train the model. It verifies the JSON schemas and evaluation scripts with a tiny synthetic run:

```bash
bash examples/run_toy_eval.sh
```

## Reproduce Paper Results

After preparing the licensed benchmark files described in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md), use the following entrypoints.

**Table III: standard benchmark results**

```bash
bash scripts/run_evirank_cv.sh all
```

**Table IV: ConfusEval evaluation**

```bash
BENCHMARK_DIR=data/confuseval \
RUN=output/evirank_confuseval.jsonl \
bash scripts/eval_confuseval.sh
```

**Table V: frozen feature-knockout diagnostics**

```bash
CKPT_DIR=output/artifacts/<dataset_run>/checkpoints \
bash scripts/run_feature_knockout.sh <dataset>
```

**Calibration sensitivity and statistical significance**

The reported calibration values are in
[`supplementary/calibration_sensitivity.csv`](supplementary/calibration_sensitivity.csv).
For paired per-query results, run:

```bash
python tools/paired_permutation_test.py <paired_metrics.csv> \
  --left baseline --right evirank --permutations 100000 --seed 42
```

## Reproducibility Notes

The main locked configuration used in the paper is summarized in `configs/evirank_locked_standard.json`.
Lightweight paper-aligned result summaries are provided in:

- `results/evirank_results_only.csv`
- `results/evirank_feature_knockout_sensitivity.csv`

Complete supporting analyses are indexed in
[`supplementary/README.md`](supplementary/README.md).

## Citation

Please use the metadata in [`CITATION.cff`](CITATION.cff).
