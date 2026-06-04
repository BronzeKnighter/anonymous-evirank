# EviRank

Anonymous implementation of **EviRank**, an evidence-grounded retrieve-and-rerank model for reviewer recommendation.

EviRank is intentionally released here as the model code only. The repository does not include competing model implementations, raw benchmark data, model checkpoints, private annotations, external repositories, or cached embeddings.

## What Is Included

- `r2reviewer/prx_eval.py`: Stage-1 semantic retrieval with SPECTER2-style paper embeddings.
- `r2reviewer/stage1_encoder.py`: text encoding and reviewer-level aggregation utilities used by Stage 1.
- `r2reviewer/reranker_features.py`: query-specific evidence selection and feature construction.
- `r2reviewer/pairwise_reranker.py`: dual-head MLP reranker.
- `r2reviewer/cv_train_reranker.py`: query-level cross-validation training and out-of-fold inference.
- `tools/eval_paper_metrics.py`: standard reviewer-ranking metrics.
- `tools/eval_confuseval.py`: hard-negative diagnostic metrics for ConfusEval-style judged pools.
- `tools/preprocess_paper_data.py`: conversion utility for qrel-style reviewer-ranking data.
- `examples/`: tiny synthetic files for checking data schemas and evaluation scripts.
- `results/`: lightweight EviRank-only result summaries and diagnostic statistics.
- `MODEL_CARD.md`: intended use, limitations, and ethical notes.
- `RELEASE_CHECKLIST.md`: pre-upload safety checklist.

## What Is Not Included

- Competing model implementations.
- Raw reviewer-ranking benchmark data.
- SPECTER2 embedding caches, trained weights, output runs, or checkpoints.
- Author-identifying paths, private notes, or paper drafts.

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

```bash
conda create -n evirank python=3.10 -y
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

## Reproducibility Notes

The main locked configuration used in the paper is summarized in `configs/evirank_locked_standard.json`.

This anonymous release is designed to make the model implementation inspectable and runnable once users provide the corresponding benchmark data.

## Citation

During anonymous review, cite this repository as an anonymous implementation package. After acceptance or public release, update `CITATION.cff` with the final paper metadata.
