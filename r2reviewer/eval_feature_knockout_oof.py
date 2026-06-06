#!/usr/bin/env python3
"""Evaluate frozen full-model feature knockout on saved OoF checkpoints.

This script keeps the trained fold checkpoints fixed and zeroes selected
feature modules only at inference time. It is intended for sensitivity
analysis, not retrained ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from cv_train_reranker import (
    apply_feature_scale,
    build_feature_scale_vector,
    feature_module_indices,
    mask_feature_modules,
    normalize_feature_modules,
    parse_feature_group_weights,
    parse_k_list,
    parse_vector_block_weights,
)
from pairwise_reranker import PairwiseMLPReranker, combine_logits, resolve_head_hyperparams
from reranker_features import (
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


def _resolve(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _build_rows(
    *,
    repo_root: Path,
    manifest: Dict,
    device: str,
    mask_feature_modules_raw: str,
    feature_group_weights_raw,
    vector_block_weights_raw,
) -> tuple[List[Dict], int, object, List[int], float, str]:
    data = manifest.get("data") or {}
    feature_raw = manifest.get("feature_spec") or {}
    train_raw = manifest.get("train") or {}
    model_raw = manifest.get("model") or {}

    reviewers = load_json(_resolve(repo_root, str(data["reviewers"])))
    queries = load_json(_resolve(repo_root, str(data["queries"])))
    gold = load_json(_resolve(repo_root, str(data["gold"])))
    paper_vecs, paper_ids = load_vec_cache(repo_root, str(data["paper_vec_cache"]))
    query_vecs, query_ids = load_vec_cache(repo_root, str(data["query_vec_cache"]))

    embedding_dim = int(query_vecs.shape[1])
    spec = feature_spec_from_meta(feature_raw, feat_dim_hint=int(manifest.get("feat_dim", 0) or 0), embedding_dim=embedding_dim)
    feat_dim_value = feature_dim(spec, embedding_dim)
    expected_dim = int(manifest.get("feat_dim", feat_dim_value) or feat_dim_value)
    if feat_dim_value != expected_dim:
        raise SystemExit(f"Feature dimension mismatch: spec={feat_dim_value}, manifest={expected_dim}")

    masked_modules = normalize_feature_modules(mask_feature_modules_raw)
    masked_indices = feature_module_indices(spec, embedding_dim, masked_modules)
    scaling_raw = manifest.get("feature_scaling") or {}
    feature_group_weights = (
        feature_group_weights_raw
        if feature_group_weights_raw not in (None, "")
        else scaling_raw.get("feature_group_weights", "")
    )
    vector_block_weights = (
        vector_block_weights_raw
        if vector_block_weights_raw not in (None, "")
        else scaling_raw.get("vector_block_weights", "")
    )
    feature_scale = build_feature_scale_vector(
        spec,
        embedding_dim,
        feature_group_weights=feature_group_weights,
        vector_block_weights=vector_block_weights,
    )

    paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    query_id_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    reviewer_ids, reviewer_to_paper_idxs = build_reviewer_to_paper_idxs(reviewers, paper_id_to_idx)
    _, reviewer_to_paper_ids = build_reviewer_to_paper_ids(reviewers, paper_id_to_idx)
    _, reviewer_to_texts = build_reviewer_to_texts(reviewers, paper_id_to_idx)
    qtext_map = build_query_text_map(queries)
    lexical_stats = build_lexical_corpus_stats(reviewer_to_texts)
    temporal_filter_raw = str(train_raw.get("temporal_paper_filter_jsonl") or "").strip()
    temporal_query_reviewer_filter = (
        load_temporal_query_reviewer_filter(_resolve(repo_root, temporal_filter_raw))
        if temporal_filter_raw
        else None
    )
    rid_to_group = {rid: i for i, rid in enumerate(reviewer_ids)}

    stage1_feature = str(train_raw.get("stage1_fuse_feature", "top3"))
    rows: List[Dict] = []
    for q in queries:
        qid = str(q.get("id") or "")
        if not qid or qid not in gold:
            continue
        qi = query_id_to_idx.get(qid)
        if qi is None:
            continue
        rels = gold.get(qid, {})
        cands = [str(rid) for rid in rels.keys() if str(rid) in rid_to_group]
        if not cands:
            continue
        x_raw, kept = compute_features_for_query_candidates(
            paper_vecs=paper_vecs,
            reviewer_to_paper_idxs=reviewer_to_paper_idxs,
            reviewer_ids=reviewer_ids,
            query_vec=np.asarray(query_vecs[qi], dtype=np.float32),
            candidate_rids=cands,
            rid_to_group=rid_to_group,
            spec=spec,
            device=device,
            reviewer_to_texts=reviewer_to_texts,
            reviewer_to_paper_ids=reviewer_to_paper_ids,
            query_text=qtext_map.get(qid),
            lexical_stats=lexical_stats,
            query_id=qid,
            temporal_query_reviewer_filter=temporal_query_reviewer_filter,
        )
        if x_raw.shape[0] == 0:
            continue
        stage1_anchor = extract_stage1_anchor_scores(x_raw, spec, embedding_dim, stage1_feature)
        x_full = per_query_zscore(x_raw)
        x_scaled = apply_feature_scale(x_full, feature_scale)
        x = mask_feature_modules(x_scaled, masked_indices)
        rows.append({"qid": qid, "x": x, "cands": kept, "stage1_anchor": stage1_anchor})

    eval_k = parse_k_list(",".join(str(x) for x in (manifest.get("eval_k") or [5, 10])))
    stage1_alpha = float(train_raw.get("stage1_fuse_alpha", 0.2))
    head_mode = str(model_raw.get("head_mode") or train_raw.get("head_mode") or model_raw.get("requested_head_mode") or "dual")
    return rows, feat_dim_value, spec, eval_k, stage1_alpha, head_mode


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen feature-knockout OoF evaluation for saved full-model checkpoints.")
    ap.add_argument("--ckpt_dir", required=True, help="Directory containing fold_XX.pth and run_manifest.json.")
    ap.add_argument("--mask_feature_modules", default="", help="Comma-separated modules to zero at inference time.")
    ap.add_argument("--feature_group_weights", default="", help="Override manifest feature-group weights.")
    ap.add_argument("--vector_block_weights", default="", help="Override manifest vector-block weights.")
    ap.add_argument("--out_oof", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ckpt_dir = _resolve(repo_root, args.ckpt_dir)
    manifest = load_json(ckpt_dir / "run_manifest.json")
    rows, feat_dim_value, _spec, _eval_k, stage1_alpha, head_mode_raw = _build_rows(
        repo_root=repo_root,
        manifest=manifest,
        device=args.device,
        mask_feature_modules_raw=args.mask_feature_modules,
        feature_group_weights_raw=args.feature_group_weights,
        vector_block_weights_raw=args.vector_block_weights,
    )
    if not rows:
        raise SystemExit("No rows built for knockout evaluation.")

    train_raw = manifest.get("train") or {}
    model_raw = manifest.get("model") or {}
    folds = int(train_raw.get("folds", 5))
    hidden = int(model_raw.get("hidden", train_raw.get("hidden", 128)))
    dropout = float(model_raw.get("dropout", train_raw.get("dropout", 0.1)))
    runtime_head_mode, _soft_w, _hard_w, final_soft_coef, final_hard_coef = resolve_head_hyperparams(
        head_mode_raw,
        final_soft_coef=train_raw.get("final_soft_coef"),
        final_hard_coef=train_raw.get("final_hard_coef"),
    )

    splits = [rows[i::folds] for i in range(folds)]
    oof: Dict[str, List[str]] = {}
    oof_score: Dict[str, List[float]] = {}
    for fold_idx, val_rows in enumerate(splits, start=1):
        fold_path = ckpt_dir / f"fold_{fold_idx:02d}.pth"
        if not fold_path.exists():
            raise SystemExit(f"Missing checkpoint: {fold_path}")
        model = PairwiseMLPReranker(in_dim=feat_dim_value, hidden=hidden, dropout=dropout, head_mode=runtime_head_mode).to(args.device)
        state = torch.load(fold_path, map_location=args.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        for row in val_rows:
            x = torch.from_numpy(row["x"]).to(args.device)
            with torch.no_grad():
                s_soft, s_hard = model(x)
                model_score = combine_logits(
                    s_soft,
                    s_hard,
                    head_mode=runtime_head_mode,
                    final_soft_coef=final_soft_coef,
                    final_hard_coef=final_hard_coef,
                ).detach().cpu().numpy()
            fused_score = fuse_model_and_stage1_scores(model_score, row.get("stage1_anchor"), stage1_alpha)
            order = np.argsort(-fused_score, kind="mergesort")
            oof[row["qid"]] = [row["cands"][k] for k in order.tolist()]
            oof_score[row["qid"]] = [float(fused_score[k]) for k in order.tolist()]

    out_path = _resolve(repo_root, args.out_oof)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qid, ranked in oof.items():
            f.write(json.dumps({"query_id": qid, "recommend_reviewer": ranked, "recommend_score": oof_score.get(qid, [])}, ensure_ascii=True) + "\n")

    manifest_out = {
        "source_ckpt_dir": str(ckpt_dir.relative_to(repo_root) if ckpt_dir.is_relative_to(repo_root) else ckpt_dir),
        "mask_feature_modules": normalize_feature_modules(args.mask_feature_modules),
        "feature_group_weights": parse_feature_group_weights(
            args.feature_group_weights or (manifest.get("feature_scaling") or {}).get("feature_group_weights", "")
        ),
        "vector_block_weights": parse_vector_block_weights(
            args.vector_block_weights or (manifest.get("feature_scaling") or {}).get("vector_block_weights", "")
        ),
        "stage1_fuse_alpha": stage1_alpha,
        "head_mode": runtime_head_mode,
        "n_queries": len(oof),
    }
    (out_path.parent / "knockout_manifest.json").write_text(json.dumps(manifest_out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote knockout OoF recs: {out_path}")


if __name__ == "__main__":
    main()
