import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from pairwise_reranker import PairwiseMLPReranker, combine_logits
from cv_train_reranker import apply_feature_scale, build_feature_scale_vector
from reranker_features import (
    DEFAULT_SPEC,
    build_lexical_corpus_stats,
    build_query_text_map,
    build_reviewer_to_paper_ids,
    build_reviewer_to_paper_idxs,
    build_reviewer_to_texts,
    compute_features_for_query_candidates,
    extract_stage1_anchor_scores,
    feature_dim,
    feature_spec_from_meta,
    fuse_model_and_stage1_scores,
    load_json,
    load_temporal_query_reviewer_filter,
    load_vec_cache,
    per_query_zscore,
)


def parse_k_list(raw: str) -> List[int]:
    items = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _open_pairs_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fh = path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            fh,
            fieldnames=["query_id", "reviewer_id", "prediction_score", "gt_grade", "gt_soft", "gt_hard", "rank"],
        )
        writer.writeheader()
        return ("csv", fh, writer)
    fh = path.open("w", encoding="utf-8")
    return ("jsonl", fh, None)


def _write_pair_row(kind: str, fh, writer, row: Dict) -> None:
    if kind == "csv":
        assert writer is not None
        writer.writerow(row)
        return
    fh.write(json.dumps(row, ensure_ascii=True) + "\n")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Rerank stage-1 candidates with a trained pairwise MLP reranker.")
    ap.add_argument("--reviewers", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--input_rec", required=True)
    ap.add_argument("--output_rec", required=True)
    ap.add_argument("--paper_vec_cache", required=True)
    ap.add_argument("--query_vec_cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", default="", help="Optional meta JSON (feature spec + model dims).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_k", default="5,10")
    ap.add_argument("--restrict_candidates", type=int, default=1)
    ap.add_argument("--stage1_fuse_alpha", type=float, default=-1.0, help="If < 0, read from meta train.stage1_fuse_alpha; otherwise override.")
    ap.add_argument("--stage1_fuse_feature", default="", help="If set, overrides meta train.stage1_fuse_feature.")
    ap.add_argument("--feature_group_weights", default="", help="Optional feature-group scaling, e.g. interaction=1,profile=1,dense=1,lexical=1.")
    ap.add_argument("--vector_block_weights", default="", help="Optional vector-block scaling, e.g. profile=1,attn=1,hadamard=1,absdiff=1.")
    ap.add_argument("--temporal_paper_filter_jsonl", default="", help="Optional query-aware reviewer-paper filter sidecar JSONL for temporal-safe feature construction.")
    ap.add_argument("--out_pairs", default="", help="Optional pair-level dump for offline case studies.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    reviewers = load_json(repo_root / args.reviewers)
    queries = load_json(repo_root / args.queries)
    gold = load_json(repo_root / args.gold)
    rec_rows = load_jsonl(repo_root / args.input_rec)

    paper_vecs, paper_ids = load_vec_cache(repo_root, args.paper_vec_cache)
    query_vecs, query_ids = load_vec_cache(repo_root, args.query_vec_cache)
    embedding_dim = int(query_vecs.shape[1])
    paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    query_id_to_idx = {qid: i for i, qid in enumerate(query_ids)}

    reviewer_ids, reviewer_to_paper_idxs = build_reviewer_to_paper_idxs(reviewers, paper_id_to_idx)
    _, reviewer_to_paper_ids = build_reviewer_to_paper_ids(reviewers, paper_id_to_idx)
    _, reviewer_to_texts = build_reviewer_to_texts(reviewers, paper_id_to_idx)
    qtext_map = build_query_text_map(queries)
    lexical_stats = build_lexical_corpus_stats(reviewer_to_texts)
    temporal_query_reviewer_filter = (
        load_temporal_query_reviewer_filter(repo_root / args.temporal_paper_filter_jsonl)
        if str(args.temporal_paper_filter_jsonl or "").strip()
        else None
    )
    rid_to_group = {rid: i for i, rid in enumerate(reviewer_ids)}

    spec = DEFAULT_SPEC
    in_dim = feature_dim(spec, embedding_dim)
    hidden = 128
    dropout = 0.1
    meta = {}
    stage1_fuse_alpha = 1.0
    stage1_fuse_feature = "top3"
    head_mode = "dual"
    if args.meta:
        meta = load_json(repo_root / args.meta)
        spec = feature_spec_from_meta(
            meta.get("feature_spec") or {},
            feat_dim_hint=meta.get("feat_dim"),
            embedding_dim=int(meta.get("embedding_dim", embedding_dim)),
        )
        in_dim = int(meta.get("feat_dim", feature_dim(spec, embedding_dim)))
        model_cfg = meta.get("model") or {}
        hidden = int(model_cfg.get("hidden", hidden))
        dropout = float(model_cfg.get("dropout", dropout))
        head_mode = str(model_cfg.get("head_mode", head_mode))
        train_meta = meta.get("train") or {}
        stage1_fuse_alpha = float(train_meta.get("stage1_fuse_alpha", stage1_fuse_alpha))
        stage1_fuse_feature = str(train_meta.get("stage1_fuse_feature", stage1_fuse_feature))
        head_mode = str(train_meta.get("head_mode", head_mode))
    if float(args.stage1_fuse_alpha) >= 0.0:
        stage1_fuse_alpha = float(args.stage1_fuse_alpha)
    if str(args.stage1_fuse_feature or "").strip():
        stage1_fuse_feature = str(args.stage1_fuse_feature)
    feature_scale = build_feature_scale_vector(
        spec,
        embedding_dim,
        feature_group_weights=args.feature_group_weights,
        vector_block_weights=args.vector_block_weights,
    )

    model = PairwiseMLPReranker(in_dim=in_dim, hidden=hidden, dropout=dropout, head_mode=head_mode).to(args.device)
    state = torch.load(repo_root / args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    eval_k = parse_k_list(args.eval_k)
    soft_sums = {k: 0.0 for k in eval_k}
    hard_sums = {k: 0.0 for k in eval_k}
    n = 0

    pairs_kind = ""
    pairs_fh = None
    pairs_writer = None
    out_pairs_path = None
    if args.out_pairs:
        out_pairs_path = repo_root / args.out_pairs
        pairs_kind, pairs_fh, pairs_writer = _open_pairs_writer(out_pairs_path)

    out_path = repo_root / args.output_rec
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for row in rec_rows:
            qid = str(row.get("query_id") or "")
            rels = gold.get(qid)
            if not qid or not rels:
                continue

            cand = [str(x) for x in (row.get("recommend_reviewer") or [])]
            if int(args.restrict_candidates):
                cand = [rid for rid in cand if rid in rels]
            if not cand:
                continue

            qi = query_id_to_idx.get(qid)
            if qi is None:
                continue
            qvec = np.asarray(query_vecs[qi], dtype=np.float32)

            x_raw, kept = compute_features_for_query_candidates(
                paper_vecs=paper_vecs,
                reviewer_to_paper_idxs=reviewer_to_paper_idxs,
                reviewer_ids=reviewer_ids,
                query_vec=qvec,
                candidate_rids=cand,
                rid_to_group=rid_to_group,
                spec=spec,
                device=args.device,
                reviewer_to_texts=reviewer_to_texts,
                reviewer_to_paper_ids=reviewer_to_paper_ids,
                query_text=qtext_map.get(qid),
                lexical_stats=lexical_stats,
                query_id=qid,
                temporal_query_reviewer_filter=temporal_query_reviewer_filter,
            )
            if x_raw.shape[0] == 0:
                continue
            stage1_anchor = extract_stage1_anchor_scores(x_raw, spec, embedding_dim, stage1_fuse_feature)
            x = per_query_zscore(x_raw)
            x = apply_feature_scale(x, feature_scale)
            xt = torch.from_numpy(x).to(args.device)
            soft_logit, hard_logit = model(xt)
            model_score = combine_logits(soft_logit, hard_logit, head_mode=head_mode).detach().cpu().numpy()
            fused_score = fuse_model_and_stage1_scores(model_score, stage1_anchor, stage1_fuse_alpha)
            order = np.argsort(-fused_score, kind="mergesort")
            ranked = [kept[i] for i in order.tolist()]
            ranked_score = [float(fused_score[i]) for i in order.tolist()]

            soft_rel = {rid for rid, s in rels.items() if int(s) >= 2}
            hard_rel = {rid for rid, s in rels.items() if int(s) == 3}
            for k in eval_k:
                topk = ranked[:k]
                soft_hits = sum(1 for rid in topk if rid in soft_rel)
                hard_hits = sum(1 for rid in topk if rid in hard_rel)
                soft_sums[k] += soft_hits / k
                hard_sums[k] += hard_hits / k
            n += 1

            out_f.write(json.dumps({"query_id": qid, "recommend_reviewer": ranked, "recommend_score": ranked_score}) + "\n")

            if out_pairs_path is not None:
                assert pairs_fh is not None
                for r, rid in enumerate(ranked, start=1):
                    _write_pair_row(
                        pairs_kind,
                        pairs_fh,
                        pairs_writer,
                        {
                            "query_id": qid,
                            "reviewer_id": str(rid),
                            "prediction_score": float(ranked_score[r - 1]),
                            "gt_grade": int(rels.get(str(rid), 0)),
                            "gt_soft": 1 if str(rid) in soft_rel else 0,
                            "gt_hard": 1 if str(rid) in hard_rel else 0,
                            "rank": int(r),
                        },
                    )

    if n == 0:
        print("No queries evaluated.")
        return
    for k in eval_k:
        print(f"SoftP@{k}: {soft_sums[k] / n:.4f}  HardP@{k}: {hard_sums[k] / n:.4f}  n={n}")
    if 5 in eval_k:
        obj = 2.0 * (soft_sums[5] / n) + 1.0 * (hard_sums[5] / n)
        print(f"Objective(2*SoftP@5+HardP@5): {obj:.4f}")
    if out_pairs_path is not None:
        assert pairs_fh is not None
        pairs_fh.close()
        print(f"Wrote pair dump: {out_pairs_path}")


if __name__ == "__main__":
    main()
