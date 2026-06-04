import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from utility.logging_utils import suppress_adapter_no_active_warning


def load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_env_settings(argv: List[str], config: Dict) -> Dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--offline", type=int)
    parser.add_argument("--clear_proxy", type=int)
    parser.add_argument("--cuda_device")
    parser.add_argument("--tokenizers_parallelism")
    parser.add_argument("--config")
    cli = vars(parser.parse_known_args(argv)[0])

    settings = {
        "offline": config.get("offline", 1),
        "clear_proxy": config.get("clear_proxy", 1),
        "cuda_device": config.get("cuda_device"),
        "tokenizers_parallelism": config.get("tokenizers_parallelism", "false"),
    }
    for key, value in cli.items():
        if value is not None:
            settings[key] = value
    return settings


def apply_env_settings(settings: Dict) -> None:
    if int(settings.get("clear_proxy", 0)):
        for key in (
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        ):
            os.environ.pop(key, None)
    if int(settings.get("offline", 0)):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cuda_device = settings.get("cuda_device")
    if cuda_device is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    tokenizers_parallelism = settings.get("tokenizers_parallelism")
    if tokenizers_parallelism is not None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", str(tokenizers_parallelism))


def parse_args(config: Dict):
    parser = argparse.ArgumentParser(description="PRX-only evaluation with SPECTER2 embeddings.")
    parser.add_argument("--config", default="r2reviewer/prx_eval_config.json")
    parser.add_argument("--offline", type=int, default=config.get("offline", 1))
    parser.add_argument("--clear_proxy", type=int, default=config.get("clear_proxy", 1))
    parser.add_argument("--cuda_device", default=config.get("cuda_device"))
    parser.add_argument("--tokenizers_parallelism", default=config.get("tokenizers_parallelism", "false"))
    parser.add_argument("--reviewers", default="nips_reviewer_data_raw/processed/specter_reviewers.json")
    parser.add_argument("--queries", default="nips_reviewer_data_raw/processed/queries_for_inference.json")
    parser.add_argument("--gold", default="nips_reviewer_data_raw/processed/gold_standard.json")
    parser.add_argument("--qid_map", default="data/query_id_map.json")
    parser.add_argument("--rid_map", default="data/reviewer_id_map.json")
    parser.add_argument("--prx_cache", default="output/prx_scores_cache.json")
    parser.add_argument("--paper_vec_cache", default="")
    parser.add_argument("--query_vec_cache", default="")
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--max_papers", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--force_recompute", type=int, default=0)
    parser.add_argument("--fast_prx", type=int, default=config.get("fast_prx", 1))
    parser.add_argument("--score_chunk", type=int, default=config.get("score_chunk", 0))
    parser.add_argument("--use_torch", type=int, default=config.get("use_torch", 1))
    parser.add_argument("--group_reduce", type=int, default=config.get("group_reduce", 1))
    parser.add_argument("--gpu_reduce", type=int, default=config.get("gpu_reduce", 1))
    parser.add_argument("--tf32", type=int, default=config.get("tf32", 0))
    parser.add_argument("--legacy_prx", type=int, default=config.get("legacy_prx", 0))
    parser.add_argument("--adapter_sanity_check", type=int, default=config.get("adapter_sanity_check", 1))
    parser.add_argument("--legacy_progress", type=int, default=config.get("legacy_progress", 1))
    parser.add_argument("--suppress_adapter_warning", type=int, default=config.get("suppress_adapter_warning", 1))
    parser.add_argument(
        "--reviewer_pooling",
        choices=[
            "max",
            "topk_mean",
            "logmeanexp",
            "softmax_mean",
            "hybrid",
            "adaptive_hybrid",
            "profile",
            "profile_hybrid",
        ],
        default=config.get("reviewer_pooling", "max"),
        help="How to aggregate paper-level similarities into a reviewer score.",
    )
    parser.add_argument("--reviewer_topk", type=int, default=config.get("reviewer_topk", 3))
    parser.add_argument("--reviewer_tau", type=float, default=config.get("reviewer_tau", 0.1))
    parser.add_argument("--reviewer_alpha", type=float, default=config.get("reviewer_alpha", 0.5))
    parser.add_argument("--reviewer_alpha_slope", type=float, default=config.get("reviewer_alpha_slope", 0.0))
    parser.add_argument("--reviewer_len_penalty", type=float, default=config.get("reviewer_len_penalty", 0.0))
    parser.add_argument("--model_name", default="allenai/specter2_base")
    parser.add_argument("--adapter_name", default="allenai/specter2_proximity")
    parser.add_argument("--clf_adapter", default="allenai/specter2_classification")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--view_mode", choices=["ta", "title", "abstract", "multi"], default="ta")
    parser.add_argument(
        "--view_fuse",
        choices=["mean", "max", "weighted", "score_mean", "score_max", "score_weighted"],
        default="mean",
    )
    parser.add_argument("--view_weights", default="0.34,0.33,0.33")
    parser.add_argument("--title_boost", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--eval_k", default="5")
    parser.add_argument("--restrict_candidates", type=int, default=1)
    parser.add_argument("--out_rec", default="")
    parser.add_argument("--stream_eval", type=int, default=config.get("stream_eval", 0))
    parser.set_defaults(**{k: v for k, v in config.items() if k not in {"offline", "clear_proxy", "cuda_device", "tokenizers_parallelism"}})
    return parser.parse_args()


def parse_k_list(raw: str) -> List[int]:
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="r2reviewer/prx_eval_config.json")
    pre_args, _ = pre.parse_known_args(sys.argv[1:])
    cfg_path = Path(pre_args.config)
    config = load_config(cfg_path)

    env_settings = parse_env_settings(sys.argv[1:], config)
    apply_env_settings(env_settings)
    args = parse_args(config)
    if int(getattr(args, "suppress_adapter_warning", 1)):
        suppress_adapter_no_active_warning()

    from stage1_encoder import build_prx_scores
    repo_root = Path(__file__).resolve().parents[1]

    with open(repo_root / args.gold, "r", encoding="utf-8") as f:
        gold = json.load(f)

    eval_k_list = parse_k_list(args.eval_k)

    soft_sums = {k: 0.0 for k in eval_k_list}
    hard_sums = {k: 0.0 for k in eval_k_list}
    n = 0

    out_f = None
    if args.out_rec:
        out_path = repo_root / args.out_rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    if int(args.stream_eval):
        rid_to_idx = None

        def on_chunk(qids, score_mat, reviewer_ids):
            nonlocal n, rid_to_idx
            if rid_to_idx is None:
                rid_to_idx = {rid: i for i, rid in enumerate(reviewer_ids)}
            for j, qid in enumerate(qids):
                rels = gold.get(qid)
                if not rels:
                    continue
                if args.restrict_candidates:
                    cand_idx = [rid_to_idx[rid] for rid in rels.keys() if rid in rid_to_idx]
                    if not cand_idx:
                        continue
                    cand_scores = score_mat[cand_idx, j]
                    candidates = [reviewer_ids[i] for i in cand_idx]
                else:
                    candidates = reviewer_ids
                    cand_scores = score_mat[:, j]
                if len(candidates) == 0:
                    continue

                order = np.argsort(-cand_scores, kind="mergesort")
                ranked = [candidates[i] for i in order]
                ranked_scores = cand_scores[order]

                soft_rel = {rid for rid, score in rels.items() if int(score) >= 2}
                hard_rel = {rid for rid, score in rels.items() if int(score) == 3}

                for k in eval_k_list:
                    topk = ranked[:k]
                    soft_hits = sum(1 for rid in topk if rid in soft_rel)
                    hard_hits = sum(1 for rid in topk if rid in hard_rel)
                    soft_sums[k] += soft_hits / k
                    hard_sums[k] += hard_hits / k

                if out_f is not None:
                    k = min(int(args.topk), len(ranked))
                    out_f.write(
                        json.dumps(
                            {
                                "query_id": qid,
                                "recommend_reviewer": ranked[:k],
                                # Backward-compatible: older scripts ignore this.
                                "recommend_score": [float(x) for x in ranked_scores[:k]],
                            }
                        )
                        + "\n"
                    )

                n += 1

        build_prx_scores(args, repo_root, on_chunk=on_chunk)
    else:
        prx_scores = build_prx_scores(args, repo_root)
        for qid, rels in gold.items():
            if qid not in prx_scores:
                continue
            scores = prx_scores[qid]
            if args.restrict_candidates:
                candidates = [rid for rid in rels.keys() if rid in scores]
            else:
                candidates = list(scores.keys())
            if not candidates:
                continue
            cand_scores = np.array([scores.get(rid, 0.0) for rid in candidates], dtype=np.float32)
            # Use a stable descending sort to match the stream_eval code path and make results reproducible.
            order = np.argsort(-cand_scores, kind="mergesort")
            ranked = [candidates[i] for i in order]
            ranked_scores = cand_scores[order]

            soft_rel = {rid for rid, score in rels.items() if int(score) >= 2}
            hard_rel = {rid for rid, score in rels.items() if int(score) == 3}

            for k in eval_k_list:
                topk = ranked[:k]
                soft_hits = sum(1 for rid in topk if rid in soft_rel)
                hard_hits = sum(1 for rid in topk if rid in hard_rel)
                soft_sums[k] += soft_hits / k
                hard_sums[k] += hard_hits / k

            if out_f is not None:
                k = min(int(args.topk), len(ranked))
                out_f.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "recommend_reviewer": ranked[:k],
                            "recommend_score": [float(x) for x in ranked_scores[:k]],
                        }
                    )
                    + "\n"
                )

            n += 1

    if out_f is not None:
        out_f.close()

    if n == 0:
        print("No queries evaluated.")
        return

    for k in eval_k_list:
        soft_p = soft_sums[k] / n
        hard_p = hard_sums[k] / n
        print(f"SoftP@{k}: {soft_p:.4f}  HardP@{k}: {hard_p:.4f}  n={n}")


if __name__ == "__main__":
    main()
