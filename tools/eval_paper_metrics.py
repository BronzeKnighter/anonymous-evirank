#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Set


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_jsonl(path: Path) -> Dict[str, list[str]]:
    run: Dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id") or obj.get("id") or obj.get("paper_id") or "")
            if not qid:
                continue
            ranked = obj.get("recommend_reviewer") or obj.get("recommend_api") or []
            run[qid] = [str(x) for x in ranked]
    return run


def parse_k_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def dcg_linear(rels: Sequence[float], k: int) -> float:
    return sum(float(rel) / math.log2(i + 1.0) for i, rel in enumerate(rels[:k], start=1))


def ndcg_at_k(ranked: Sequence[str], rel_map: Mapping[str, float], k: int) -> float:
    rels = [float(rel_map.get(rid, 0.0)) for rid in ranked[:k]]
    dcg = dcg_linear(rels, k)
    ideal = sorted((float(v) for v in rel_map.values()), reverse=True)
    idcg = dcg_linear(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    return sum(1 for rid in ranked[:k] if rid in positives) / max(1, k)


def average_precision_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    if not positives:
        return 0.0
    hits = 0
    score = 0.0
    for i, rid in enumerate(ranked[:k], start=1):
        if rid in positives:
            hits += 1
            score += hits / i
    return score / len(positives)


def graded_from_gold(gold: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    # Evaluation gains: 0 for non-relevant, 1 for soft positive, 2 for hard positive.
    out: Dict[str, Dict[str, float]] = {}
    for qid, rels in gold.items():
        cur = {}
        for rid, score in rels.items():
            s = float(score)
            if s >= 3:
                cur[str(rid)] = 2.0
            elif s >= 2:
                cur[str(rid)] = 1.0
            else:
                cur[str(rid)] = 0.0
        out[str(qid)] = cur
    return out


def evaluate_run(run_path: Path, gold_path: Path, ndcg_ks: list[int], map_ks: list[int], p_ks: list[int]) -> Dict[str, float]:
    run = load_run_jsonl(run_path)
    gold_raw = {str(qid): {str(rid): float(v) for rid, v in rels.items()} for qid, rels in load_json(gold_path).items()}
    graded = graded_from_gold(gold_raw)
    soft_pos = {qid: {rid for rid, v in rels.items() if float(v) >= 2.0} for qid, rels in gold_raw.items()}
    hard_pos = {qid: {rid for rid, v in rels.items() if float(v) >= 3.0} for qid, rels in gold_raw.items()}

    qids = [qid for qid in gold_raw if qid in run]
    n = len(qids)
    if n == 0:
        raise RuntimeError(f"No overlapping queries between run and gold: {run_path}")

    metrics: Dict[str, float] = {
        "queries_evaluated": float(n),
        "queries_total": float(len(gold_raw)),
    }
    for k in ndcg_ks:
        metrics[f"ndcg@{k}"] = sum(ndcg_at_k(run[qid], graded[qid], k) for qid in qids) / n
    for k in map_ks:
        metrics[f"soft_map@{k}"] = sum(average_precision_at_k(run[qid], soft_pos[qid], k) for qid in qids) / n
        metrics[f"hard_map@{k}"] = sum(average_precision_at_k(run[qid], hard_pos[qid], k) for qid in qids) / n
    for k in p_ks:
        metrics[f"soft_p@{k}"] = sum(precision_at_k(run[qid], soft_pos[qid], k) for qid in qids) / n
        metrics[f"hard_p@{k}"] = sum(precision_at_k(run[qid], hard_pos[qid], k) for qid in qids) / n
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EviRank run on a judged reviewer-ranking pool.")
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--gold_json", required=True)
    parser.add_argument("--glob", default="*_oof_rec_scored.jsonl")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--ndcg_k", default="5,10")
    parser.add_argument("--map_k", default="5,10")
    parser.add_argument("--p_k", default="5")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    run_files = sorted(runs_dir.glob(args.glob))
    if not run_files:
        raise FileNotFoundError(f"No run files matched {args.glob} under {runs_dir}")

    ndcg_ks = parse_k_list(args.ndcg_k)
    map_ks = parse_k_list(args.map_k)
    p_ks = parse_k_list(args.p_k)
    for run_path in run_files:
        metrics = evaluate_run(run_path, Path(args.gold_json), ndcg_ks, map_ks, p_ks)
        if args.json:
            print(json.dumps({"run_file": str(run_path), **metrics}, indent=2))
        else:
            label = args.dataset or run_path.stem
            print(f"{label}: {run_path}")
            for key in sorted(metrics):
                print(f"  {key} = {metrics[key]:.6f}")


if __name__ == "__main__":
    main()

