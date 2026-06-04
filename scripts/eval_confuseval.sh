#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/confuseval}"
RUN="${RUN:-output/evirank_confuseval.jsonl}"
KS="${KS:-1,3,5}"

"$PYTHON" tools/eval_confuseval.py \
  --benchmark_dir "$BENCHMARK_DIR" \
  --run "$RUN" \
  --ks "$KS"
