#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python tools/eval_paper_metrics.py \
  --runs_dir examples/toy_runs \
  --dataset toy \
  --gold_json examples/toy_data/processed/toy/gold_standard.json \
  --glob "toy_evirank_run.jsonl" \
  --map_k 5,10 \
  --p_k 5,10

python tools/eval_confuseval.py \
  --benchmark_dir examples/toy_data/confuseval \
  --run examples/toy_runs/toy_evirank_run.jsonl \
  --ks 1,3,5
