#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(nips kdd sigir scirepeval)
fi
if [ "${DATASETS[0]}" = "all" ]; then
  DATASETS=(nips kdd sigir scirepeval)
fi

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

BATCH_SIZE="${BATCH_SIZE:-8}"
SCORE_CHUNK="${SCORE_CHUNK:-8192}"
TOPK="${TOPK:-50}"
EVAL_K="${EVAL_K:-5,10}"
SEED="${SEED:-42}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-200}"
HIDDEN="${HIDDEN:-128}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
POINTWISE_WEIGHT="${POINTWISE_WEIGHT:-0.25}"
LISTWISE_WEIGHT="${LISTWISE_WEIGHT:-0.4}"
MAX_PAIRS="${MAX_PAIRS:-1024}"
NEG_STRATEGY="${NEG_STRATEGY:-mixed}"
HARD_NEG_FEATURE="${HARD_NEG_FEATURE:-lexical}"
HARD_NEG_TOPN="${HARD_NEG_TOPN:-32}"
RAND_NEG_TOPN="${RAND_NEG_TOPN:-16}"
ATTENTION_TAU="${ATTENTION_TAU:-0.2}"
EVIDENCE_TOPK="${EVIDENCE_TOPK:-5}"
STAGE1_FUSE_ALPHA="${STAGE1_FUSE_ALPHA:-0.2}"
STAGE1_FUSE_FEATURE="${STAGE1_FUSE_FEATURE:-top3}"

TITLE_BOOST="${TITLE_BOOST:-4}"
POOLING="${POOLING:-mean}"
VIEW_MODE="${VIEW_MODE:-multi}"
VIEW_FUSE="${VIEW_FUSE:-mean}"
REVIEWER_POOLING="${REVIEWER_POOLING:-hybrid}"
REVIEWER_TOPK="${REVIEWER_TOPK:-5}"
REVIEWER_ALPHA="${REVIEWER_ALPHA:-0.85}"
CACHE_TAG="${CACHE_TAG:-multi_mean_tb4}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

run_dataset() {
  local ds="$1"
  local reviewers="data/processed/${ds}/specter_reviewers.json"
  local queries="data/processed/${ds}/queries_for_inference.json"
  local gold="data/processed/${ds}/gold_standard.json"
  local paper_cache="output/paper_vecs_${ds}_${CACHE_TAG}.npy"
  local query_cache="output/query_vecs_${ds}_${CACHE_TAG}.npy"
  local art_dir="output/artifacts/${ds}_evirank_oof_${TS}"

  if [ ! -f "$reviewers" ] || [ ! -f "$queries" ] || [ ! -f "$gold" ]; then
    echo "Missing processed data for dataset '${ds}'." >&2
    echo "Expected:" >&2
    echo "  $reviewers" >&2
    echo "  $queries" >&2
    echo "  $gold" >&2
    echo "See docs/DATA_FORMAT.md for preprocessing instructions." >&2
    exit 1
  fi

  "$PYTHON" -u r2reviewer/prx_eval.py \
    --reviewers "$reviewers" \
    --queries "$queries" \
    --gold "$gold" \
    --pooling "$POOLING" \
    --view_mode "$VIEW_MODE" \
    --view_fuse "$VIEW_FUSE" \
    --title_boost "$TITLE_BOOST" \
    --reviewer_pooling "$REVIEWER_POOLING" \
    --reviewer_topk "$REVIEWER_TOPK" \
    --reviewer_alpha "$REVIEWER_ALPHA" \
    --legacy_prx 0 \
    --batch_size "$BATCH_SIZE" \
    --score_chunk "$SCORE_CHUNK" \
    --stream_eval 1 \
    --topk "$TOPK" \
    --eval_k "$EVAL_K" \
    --paper_vec_cache "$paper_cache" \
    --query_vec_cache "$query_cache"

  "$PYTHON" -u r2reviewer/cv_train_reranker.py \
    --reviewers "$reviewers" \
    --queries "$queries" \
    --gold "$gold" \
    --paper_vec_cache "$paper_cache" \
    --query_vec_cache "$query_cache" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --folds "$FOLDS" \
    --epochs "$EPOCHS" \
    --hidden "$HIDDEN" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --pointwise_weight "$POINTWISE_WEIGHT" \
    --listwise_weight "$LISTWISE_WEIGHT" \
    --max_pairs_per_query "$MAX_PAIRS" \
    --attention_tau "$ATTENTION_TAU" \
    --evidence_topk "$EVIDENCE_TOPK" \
    --neg_strategy "$NEG_STRATEGY" \
    --hard_neg_feature "$HARD_NEG_FEATURE" \
    --hard_neg_topn "$HARD_NEG_TOPN" \
    --rand_neg_topn "$RAND_NEG_TOPN" \
    --stage1_fuse_alpha "$STAGE1_FUSE_ALPHA" \
    --stage1_fuse_feature "$STAGE1_FUSE_FEATURE" \
    --eval_k "$EVAL_K" \
    --out_oof "${art_dir}/${ds}_oof_rec_scored.jsonl" \
    --out_pairs "${art_dir}/${ds}_oof_pairs.jsonl" \
    --ckpt_dir "${art_dir}/checkpoints"

  "$PYTHON" tools/eval_paper_metrics.py \
    --runs_dir "$art_dir" \
    --dataset "$ds" \
    --gold_json "$gold" \
    --map_k "$EVAL_K" \
    --p_k "$EVAL_K" \
    --glob "${ds}_oof_rec_scored.jsonl"
}

for ds in "${DATASETS[@]}"; do
  run_dataset "$ds"
done

