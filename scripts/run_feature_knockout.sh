#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${1:-kdd}"
CKPT_DIR="${CKPT_DIR:-output/artifacts/${DATASET}_evirank_oof/checkpoints}"
OUT_ROOT="${OUT_ROOT:-output/feature_knockout/${DATASET}}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"

declare -A MODULES=(
  [full]=""
  [wo_interaction]="explicit_interaction"
  [wo_profile]="profile_context"
  [wo_dense]="dense_semantic_stats"
  [wo_lexical]="lexical_matching"
)

for variant in full wo_interaction wo_profile wo_dense wo_lexical; do
  mkdir -p "${OUT_ROOT}/${variant}"
  "$PYTHON" -u r2reviewer/eval_feature_knockout_oof.py \
    --ckpt_dir "$CKPT_DIR" \
    --mask_feature_modules "${MODULES[$variant]}" \
    --out_oof "${OUT_ROOT}/${variant}/${DATASET}_oof_rec_scored.jsonl" \
    --device "$DEVICE"

  "$PYTHON" tools/eval_paper_metrics.py \
    --runs_dir "${OUT_ROOT}/${variant}" \
    --dataset "$DATASET" \
    --gold_json "data/processed/${DATASET}/gold_standard.json" \
    --map_k 5,10 \
    --p_k 5,10 \
    --glob "${DATASET}_oof_rec_scored.jsonl"
done
