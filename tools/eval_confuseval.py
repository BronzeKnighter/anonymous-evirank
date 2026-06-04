#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple


def load_jsonl_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id") or obj.get("id") or "")
            if not qid:
                continue
            qrels[qid] = {str(rid): int(v) for rid, v in (obj.get("score") or {}).items()}
    return qrels


def load_run_jsonl(path: Path) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
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


def load_run_jsonl_with_scores(path: Path) -> Dict[str, Dict[str, object]]:
    run: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id") or obj.get("id") or obj.get("paper_id") or "")
            if not qid:
                continue
            ranked = [str(x) for x in (obj.get("recommend_reviewer") or obj.get("recommend_api") or [])]
            raw_scores = obj.get("recommend_score") or []
            scores: Dict[str, float] = {}
            for rid, score in zip(ranked, raw_scores):
                try:
                    scores[rid] = float(score)
                except (TypeError, ValueError):
                    continue
            run[qid] = {"ranked": ranked, "scores": scores}
    return run


def dcg_linear(rels: Sequence[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        score += float(rel) / math.log2(i + 1.0)
    return score


def ndcg_at_k(ranked: Sequence[str], rel_map: Mapping[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    rels = [float(rel_map.get(rid, 0.0)) for rid in ranked[:k]]
    dcg = dcg_linear(rels, k)
    ideal = sorted((float(v) for v in rel_map.values()), reverse=True)
    idcg = dcg_linear(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = sum(1 for rid in ranked[:k] if rid in positives)
    return hit / k


def average_precision_at_k(ranked: Sequence[str], positives: Set[str], k: int) -> float:
    if not positives:
        return 0.0
    hit = 0
    score = 0.0
    for i, rid in enumerate(ranked[:k], start=1):
        if rid in positives:
            hit += 1
            score += hit / i
    return score / len(positives)


def build_binary_positives(qrels: Dict[str, Dict[str, int]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for qid, rels in qrels.items():
        out[qid] = {rid for rid, v in rels.items() if int(v) > 0}
    return out


def build_hard_negative_sets(qrels: Dict[str, Dict[str, int]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for qid, rels in qrels.items():
        out[qid] = {rid for rid, v in rels.items() if int(v) == 0}
    return out


def first_rank(ranked: Sequence[str], targets: Set[str], k: int | None = None) -> int | None:
    limit = ranked if k is None else ranked[:k]
    for i, rid in enumerate(limit, start=1):
        if rid in targets:
            return i
    return None


def evaluate_negative_sensitive(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: Sequence[int],
) -> Dict[str, float]:
    positives = build_binary_positives(qrels)
    hard_negatives = build_hard_negative_sets(qrels)
    qids = sorted(qrels.keys())
    covered = [qid for qid in qids if qid in run]

    out: Dict[str, float] = {
        "queries_total": float(len(qids)),
        "queries_covered": float(len(covered)),
        "coverage": float(len(covered) / len(qids)) if qids else 0.0,
    }

    # Global rank-sensitive negative intrusion metrics.
    first_hn_rr_sum = 0.0
    pos_before_hn = 0
    hn_before_pos = 0
    tie_or_neither = 0
    for qid in covered:
        ranked = run[qid]
        pos_rank = first_rank(ranked, positives[qid], None)
        hn_rank = first_rank(ranked, hard_negatives[qid], None)
        if hn_rank is not None:
            first_hn_rr_sum += 1.0 / hn_rank
        pos_cmp = pos_rank if pos_rank is not None else 10**9
        hn_cmp = hn_rank if hn_rank is not None else 10**9
        if pos_cmp < hn_cmp:
            pos_before_hn += 1
        elif hn_cmp < pos_cmp:
            hn_before_pos += 1
        else:
            tie_or_neither += 1

    n_cov = len(covered) or 1
    out["first_hn_mrr"] = first_hn_rr_sum / n_cov
    out["pos_before_first_hn_rate"] = pos_before_hn / n_cov
    out["hn_before_first_pos_rate"] = hn_before_pos / n_cov
    out["tie_or_neither_rate"] = tie_or_neither / n_cov

    # Cutoff-sensitive negative intrusion metrics.
    for k in ks:
        hn_hit = 0
        hn_count_sum = 0
        pos_before_hn_at_k = 0
        for qid in covered:
            ranked = run[qid][:k]
            hn_count = sum(1 for rid in ranked if rid in hard_negatives[qid])
            hn_count_sum += hn_count
            if hn_count > 0:
                hn_hit += 1

            pos_rank = first_rank(ranked, positives[qid], None)
            hn_rank = first_rank(ranked, hard_negatives[qid], None)
            pos_cmp = pos_rank if pos_rank is not None else 10**9
            hn_cmp = hn_rank if hn_rank is not None else 10**9
            if pos_cmp < hn_cmp:
                pos_before_hn_at_k += 1

        out[f"hn_hit@{k}"] = hn_hit / n_cov
        out[f"safe@{k}"] = 1.0 - (hn_hit / n_cov)
        out[f"hn_count@{k}"] = hn_count_sum / n_cov
        out[f"pos_before_hn@{k}"] = pos_before_hn_at_k / n_cov
    return out


def evaluate_discrimination(
    run_with_scores: Dict[str, Dict[str, object]],
    qrels: Dict[str, Dict[str, int]],
    ks: Sequence[int],
) -> Dict[str, float]:
    positives = build_binary_positives(qrels)
    hard_negatives = build_hard_negative_sets(qrels)
    qids = sorted(qrels.keys())
    covered = [qid for qid in qids if qid in run_with_scores]

    out: Dict[str, float] = {
        "queries_total": float(len(qids)),
        "queries_covered": float(len(covered)),
        "coverage": float(len(covered) / len(qids)) if qids else 0.0,
    }
    n_cov = len(covered) or 1

    judged_pool_size_sum = 0.0
    judged_returned_sum = 0.0
    judged_coverage_sum = 0.0
    pos_found = 0
    hn_found = 0
    best_pos_rank_sum = 0.0
    best_hn_rank_sum = 0.0

    rank_pair_total = 0
    rank_pair_decidable = 0
    rank_pair_correct = 0.0
    explicit_rank_pair_total = 0
    explicit_rank_pair_correct = 0.0

    score_pair_total = 0
    score_pair_correct = 0.0
    best_margin_count = 0
    best_margin_sum = 0.0
    avg_margin_count = 0
    avg_margin_sum = 0.0

    query_auc_sum = 0.0
    query_auc_count = 0
    query_winrate_sum = 0.0
    query_winrate_count = 0
    best_pos_rank_sum_q = 0.0
    best_pos_rank_count_q = 0
    positive_mrr_sum = 0.0
    recall1_sum = 0.0
    recall3_sum = 0.0
    recall5_sum = 0.0
    mean_violation_sum = 0.0
    violation_gt0 = 0
    violation_ge2 = 0

    for qid in covered:
        payload = run_with_scores[qid]
        ranked = payload.get("ranked", [])
        scores = payload.get("scores", {})
        rank_map = {rid: i for i, rid in enumerate(ranked, start=1)}
        judged = set(qrels[qid].keys())
        pos_set = positives[qid]
        hn_set = hard_negatives[qid]

        judged_pool_size = len(judged)
        judged_returned = sum(1 for rid in ranked if rid in judged)
        judged_pool_size_sum += judged_pool_size
        judged_returned_sum += judged_returned
        judged_coverage_sum += (judged_returned / judged_pool_size) if judged_pool_size else 0.0

        pos_rank = first_rank(ranked, pos_set, None)
        hn_rank = first_rank(ranked, hn_set, None)
        if pos_rank is not None:
            pos_found += 1
            best_pos_rank_sum += pos_rank
        if hn_rank is not None:
            hn_found += 1
            best_hn_rank_sum += hn_rank

        for pos in pos_set:
            for hn in hn_set:
                rank_pair_total += 1
                p_rank = rank_map.get(pos)
                h_rank = rank_map.get(hn)
                # Decidable if at least one side is explicitly ranked.
                if p_rank is not None or h_rank is not None:
                    rank_pair_decidable += 1
                    if p_rank is None:
                        pass
                    elif h_rank is None:
                        rank_pair_correct += 1.0
                    elif p_rank < h_rank:
                        rank_pair_correct += 1.0
                    elif p_rank == h_rank:
                        rank_pair_correct += 0.5

                if p_rank is not None and h_rank is not None:
                    explicit_rank_pair_total += 1
                    if p_rank < h_rank:
                        explicit_rank_pair_correct += 1.0
                    elif p_rank == h_rank:
                        explicit_rank_pair_correct += 0.5

                p_score = scores.get(pos)
                h_score = scores.get(hn)
                if p_score is not None and h_score is not None:
                    score_pair_total += 1
                    if p_score > h_score:
                        score_pair_correct += 1.0
                    elif p_score == h_score:
                        score_pair_correct += 0.5

        pos_scores = [scores[rid] for rid in pos_set if rid in scores]
        hn_scores = [scores[rid] for rid in hn_set if rid in scores]
        if pos_scores and hn_scores:
            best_margin_sum += max(pos_scores) - max(hn_scores)
            best_margin_count += 1
            avg_margin_sum += (sum(pos_scores) / len(pos_scores)) - (sum(hn_scores) / len(hn_scores))
            avg_margin_count += 1

            pair_total = 0
            pair_correct = 0.0
            for p_score in pos_scores:
                for h_score in hn_scores:
                    pair_total += 1
                    if p_score > h_score:
                        pair_correct += 1.0
                    elif p_score == h_score:
                        pair_correct += 0.5
            if pair_total > 0:
                query_auc_sum += pair_correct / pair_total
                query_auc_count += 1
                query_winrate_sum += pair_correct / pair_total
                query_winrate_count += 1

            best_pos_score = max(pos_scores)
            violations = sum(1 for h_score in hn_scores if h_score > best_pos_score)
            mean_violation_sum += violations
            if violations > 0:
                violation_gt0 += 1
            if violations >= 2:
                violation_ge2 += 1

        if pos_rank is not None:
            best_pos_rank_sum_q += pos_rank
            best_pos_rank_count_q += 1
            positive_mrr_sum += 1.0 / pos_rank
            recall1_sum += 1.0 if pos_rank <= 1 else 0.0
            recall3_sum += 1.0 if pos_rank <= 3 else 0.0
            recall5_sum += 1.0 if pos_rank <= 5 else 0.0

    out["judged_pool_size_avg"] = judged_pool_size_sum / n_cov
    out["judged_candidates_returned_avg"] = judged_returned_sum / n_cov
    out["judged_coverage_avg"] = judged_coverage_sum / n_cov
    out["positive_found_rate"] = pos_found / n_cov
    out["hard_negative_found_rate"] = hn_found / n_cov
    out["best_positive_rank_mean"] = best_pos_rank_sum / pos_found if pos_found else 0.0
    out["best_hard_negative_rank_mean"] = best_hn_rank_sum / hn_found if hn_found else 0.0

    out["rank_pairwise_total"] = float(rank_pair_total)
    out["rank_pairwise_decidable"] = float(rank_pair_decidable)
    out["rank_pairwise_decidable_rate"] = (rank_pair_decidable / rank_pair_total) if rank_pair_total else 0.0
    out["rank_pairwise_acc_pos_vs_hn"] = (rank_pair_correct / rank_pair_decidable) if rank_pair_decidable else 0.0
    out["explicit_rank_pairwise_total"] = float(explicit_rank_pair_total)
    out["explicit_rank_pairwise_coverage"] = (
        explicit_rank_pair_total / rank_pair_total if rank_pair_total else 0.0
    )
    out["explicit_rank_pairwise_acc_pos_vs_hn"] = (
        explicit_rank_pair_correct / explicit_rank_pair_total if explicit_rank_pair_total else 0.0
    )

    out["score_pairwise_total"] = float(score_pair_total)
    out["score_pairwise_coverage_vs_rank_pairs"] = (score_pair_total / rank_pair_total) if rank_pair_total else 0.0
    out["score_pairwise_acc_pos_vs_hn"] = (score_pair_correct / score_pair_total) if score_pair_total else 0.0
    out["best_pos_minus_best_hn_score_margin_mean"] = (
        best_margin_sum / best_margin_count if best_margin_count else 0.0
    )
    out["best_pos_minus_best_hn_score_margin_coverage"] = best_margin_count / n_cov
    out["mean_margin_pos_vs_hn"] = avg_margin_sum / avg_margin_count if avg_margin_count else 0.0
    out["mean_query_auc_pos_vs_hn"] = query_auc_sum / query_auc_count if query_auc_count else 0.0
    out["pairwise_win_rate_pos_vs_hn"] = query_winrate_sum / query_winrate_count if query_winrate_count else 0.0
    out["best_positive_rank_mean_query"] = (
        best_pos_rank_sum_q / best_pos_rank_count_q if best_pos_rank_count_q else 0.0
    )
    out["positive_mrr_within_reviewed_set"] = positive_mrr_sum / n_cov
    out["positive_recall@1_within_reviewed_set"] = recall1_sum / n_cov
    out["positive_recall@3_within_reviewed_set"] = recall3_sum / n_cov
    out["positive_recall@5_within_reviewed_set"] = recall5_sum / n_cov
    out["mean_violations_against_best_positive"] = mean_violation_sum / n_cov
    out["violation_rate_gt0"] = violation_gt0 / n_cov
    out["violation_rate_ge2"] = violation_ge2 / n_cov

    for k in ks:
        coverage_sum = 0.0
        for qid in covered:
            payload = run_with_scores[qid]
            ranked = payload.get("ranked", [])[:k]
            judged = set(qrels[qid].keys())
            judged_pool_size = len(judged)
            judged_returned = sum(1 for rid in ranked if rid in judged)
            coverage_sum += (judged_returned / judged_pool_size) if judged_pool_size else 0.0
        out[f"judged_coverage@{k}"] = coverage_sum / n_cov
    return out


def evaluate(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: Sequence[int],
) -> Dict[str, float]:
    positives = build_binary_positives(qrels)
    qids = sorted(qrels.keys())
    covered = [qid for qid in qids if qid in run]

    metrics: Dict[str, float] = {
        "queries_total": float(len(qids)),
        "queries_covered": float(len(covered)),
        "coverage": float(len(covered) / len(qids)) if qids else 0.0,
    }
    for k in ks:
        ndcg_sum = 0.0
        map_sum = 0.0
        p_sum = 0.0
        n = 0
        for qid in qids:
            ranked = run.get(qid)
            if not ranked:
                continue
            ndcg_sum += ndcg_at_k(ranked, qrels[qid], k)
            map_sum += average_precision_at_k(ranked, positives[qid], k)
            p_sum += precision_at_k(ranked, positives[qid], k)
            n += 1
        metrics[f"ndcg@{k}"] = ndcg_sum / n if n else 0.0
        metrics[f"map@{k}"] = map_sum / n if n else 0.0
        metrics[f"p@{k}"] = p_sum / n if n else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EviRank runs on a ConfusEval-style hard-negative judged pool.")
    parser.add_argument("--benchmark_dir", required=True, help="Directory containing europepmc_queries_test_{hard,soft}.json")
    parser.add_argument("--run", required=True, help="Run JSONL with query_id + recommend_reviewer.")
    parser.add_argument("--ks", default="1,3,5,10", help="Comma-separated cutoffs.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    run_path = Path(args.run)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    hard_qrels = load_jsonl_qrels(benchmark_dir / "europepmc_queries_test_hard.json")
    soft_qrels = load_jsonl_qrels(benchmark_dir / "europepmc_queries_test_soft.json")
    run = load_run_jsonl(run_path)
    run_with_scores = load_run_jsonl_with_scores(run_path)

    out = {
        "benchmark_dir": str(benchmark_dir),
        "run": str(run_path),
        "ks": ks,
        "hard": evaluate(run, hard_qrels, ks),
        "soft": evaluate(run, soft_qrels, ks),
        "negative_sensitive": evaluate_negative_sensitive(run, hard_qrels, ks),
        "discrimination": evaluate_discrimination(run_with_scores, hard_qrels, ks),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
