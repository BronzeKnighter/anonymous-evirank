import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_paper_map(papers_path: Path) -> Dict[str, Tuple[str, str]]:
    paper_map: Dict[str, Tuple[str, str]] = {}
    for item in load_jsonl(papers_path):
        pid = str(item.get("paper") or item.get("id") or "")
        if not pid:
            continue
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        paper_map[pid] = (title, abstract)
    return paper_map


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    for item in load_jsonl(qrels_path):
        qid = str(item.get("query_id") or item.get("id") or "")
        if not qid:
            continue
        scores = {str(rid): int(score) for rid, score in item.get("score", {}).items()}
        qrels[qid] = scores
    return qrels


def sort_ids(ids: List[str], mode: str) -> List[str]:
    if mode == "str":
        return sorted(ids)

    def key(x: str):
        if x.isdigit():
            return (0, int(x))
        return (1, x)

    return sorted(ids, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess paper_data JSONL into SPECTER2-ready JSON.")
    parser.add_argument("--papers", required=True, help="JSONL with papers (paper/title/abstract).")
    parser.add_argument("--reviewers", required=True, help="JSONL with reviewers (reviewer/papers).")
    parser.add_argument("--qrels_soft", required=True, help="JSONL soft qrels.")
    parser.add_argument("--qrels_hard", required=True, help="JSONL hard qrels.")
    parser.add_argument("--out_dir", required=True, help="Output directory for processed JSON.")
    parser.add_argument("--filter_qrels", type=int, default=1, help="Filter reviewers to qrels candidate set.")
    parser.add_argument("--write_maps", type=int, default=1, help="Write query/reviewer id maps.")
    parser.add_argument("--write_split", type=int, default=1, help="Write split soft/hard gold files.")
    parser.add_argument("--write_whitelist", type=int, default=1, help="Write reviewer whitelist text file.")
    parser.add_argument("--id_sort", choices=["int", "str"], default="int", help="Sorting for id maps.")
    parser.add_argument(
        "--gold_mode",
        choices=["global", "per_query"],
        default="global",
        help=(
            "How to build gold_standard.json. "
            "'global' expands each query to the union candidate set across all qrels (treat unlisted as 0). "
            "'per_query' keeps only the reviewers listed in that query's qrels (recommended when qrels are partial)."
        ),
    )
    args = parser.parse_args()

    papers_path = Path(args.papers)
    reviewers_path = Path(args.reviewers)
    qrels_soft_path = Path(args.qrels_soft)
    qrels_hard_path = Path(args.qrels_hard)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_map = build_paper_map(papers_path)
    qrels_soft = load_qrels(qrels_soft_path)
    qrels_hard = load_qrels(qrels_hard_path)

    candidate_ids = set()
    for scores in qrels_soft.values():
        candidate_ids.update(scores.keys())
    for scores in qrels_hard.values():
        candidate_ids.update(scores.keys())

    # Build reviewer profiles with paper texts.
    reviewers_out: List[Dict] = []
    for item in load_jsonl(reviewers_path):
        rid = str(item.get("reviewer") or item.get("id") or "")
        if not rid:
            continue
        if args.filter_qrels and rid not in candidate_ids:
            continue
        papers = []
        for pid in item.get("papers", []):
            pid = str(pid)
            if pid not in paper_map:
                continue
            title, abstract = paper_map[pid]
            papers.append({"id": pid, "title": title, "abstract": abstract})
        reviewers_out.append({"id": rid, "papers": papers})

    # Build gold standard (score=3 for hard, 2 for soft).
    gold_map: Dict[str, Dict[str, int]] = {}
    gold_soft: Dict[str, Dict[str, int]] = {}
    gold_hard: Dict[str, Dict[str, int]] = {}

    all_qids = set(qrels_soft.keys()) | set(qrels_hard.keys())
    reviewer_list = sort_ids(list(candidate_ids), args.id_sort)

    per_query_sizes: List[int] = []
    for qid in sort_ids(list(all_qids), args.id_sort):
        soft_scores = qrels_soft.get(qid, {})
        hard_scores = qrels_hard.get(qid, {})

        if args.gold_mode == "per_query":
            per_q_candidates = set(soft_scores.keys()) | set(hard_scores.keys())
            cur_reviewer_list = sort_ids(list(per_q_candidates), args.id_sort)
        else:
            cur_reviewer_list = reviewer_list

        per_query_sizes.append(len(cur_reviewer_list))
        scores = {}
        scores_soft = {}
        scores_hard = {}
        for rid in cur_reviewer_list:
            soft_val = 2 if int(soft_scores.get(rid, 0)) > 0 else 0
            hard_val = 3 if int(hard_scores.get(rid, 0)) > 0 else 0
            val = hard_val if hard_val > 0 else soft_val
            scores[rid] = val
            scores_soft[rid] = soft_val
            scores_hard[rid] = hard_val
        gold_map[qid] = scores
        gold_soft[qid] = scores_soft
        gold_hard[qid] = scores_hard

    # Build query list for inference.
    queries_out: List[Dict] = []
    for qid in sort_ids(list(gold_map.keys()), args.id_sort):
        if qid not in paper_map:
            continue
        title, abstract = paper_map[qid]
        queries_out.append({"id": qid, "title": title, "abstract": abstract})

    # Write outputs.
    with (out_dir / "specter_reviewers.json").open("w", encoding="utf-8") as f:
        json.dump(reviewers_out, f, indent=2)
    with (out_dir / "queries_for_inference.json").open("w", encoding="utf-8") as f:
        json.dump(queries_out, f, indent=2)
    with (out_dir / "gold_standard.json").open("w", encoding="utf-8") as f:
        json.dump(gold_map, f, indent=2)

    if args.write_split:
        with (out_dir / "gold_standard_soft.json").open("w", encoding="utf-8") as f:
            json.dump(gold_soft, f, indent=2)
        with (out_dir / "gold_standard_hard.json").open("w", encoding="utf-8") as f:
            json.dump(gold_hard, f, indent=2)

    if args.write_whitelist:
        with (out_dir / "reviewer_whitelist.txt").open("w", encoding="utf-8") as f:
            for rid in reviewer_list:
                f.write(f"{rid}\n")

    if args.write_maps:
        qid_map = {qid: idx for idx, qid in enumerate(sort_ids([q["id"] for q in queries_out], args.id_sort))}
        rid_map = {rid: idx for idx, rid in enumerate(reviewer_list)}
        with (out_dir / "query_id_map.json").open("w", encoding="utf-8") as f:
            json.dump(qid_map, f, indent=2)
        with (out_dir / "reviewer_id_map.json").open("w", encoding="utf-8") as f:
            json.dump(rid_map, f, indent=2)

    if args.gold_mode == "per_query" and per_query_sizes:
        avg = sum(per_query_sizes) / len(per_query_sizes)
        min_sz = min(per_query_sizes)
        max_sz = max(per_query_sizes)
        print(f"Reviewers: {len(reviewers_out)} (union candidate set={len(reviewer_list)}; per-query candidates avg={avg:.1f} min={min_sz} max={max_sz})")
    else:
        print(f"Reviewers: {len(reviewers_out)} (candidate set={len(reviewer_list)})")
    print(f"Queries: {len(queries_out)}")
    print(f"Gold queries: {len(gold_map)}")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
