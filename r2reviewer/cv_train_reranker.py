import argparse
import csv
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pairwise_reranker import (
    PairwiseMLPReranker,
    TrainConfig,
    combine_logits,
    has_hard_head,
    has_soft_head,
    pairwise_logistic_loss,
    resolve_head_hyperparams,
)
from reranker_features import (
    DEFAULT_SPEC,
    FeatureSpec,
    build_lexical_corpus_stats,
    build_query_text_map,
    build_reviewer_to_paper_ids,
    build_reviewer_to_paper_idxs,
    build_reviewer_to_texts,
    compute_features_for_query_candidates,
    extract_stage1_anchor_scores,
    feature_dim,
    feature_layout,
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


def build_feature_spec(args: argparse.Namespace) -> FeatureSpec:
    return FeatureSpec(
        topk_list=list(DEFAULT_SPEC.topk_list),
        logmeanexp_tau_list=list(DEFAULT_SPEC.logmeanexp_tau_list),
        softmax_tau_list=list(DEFAULT_SPEC.softmax_tau_list),
        attention_tau=float(args.attention_tau),
        include_attention_stats=not bool(int(args.disable_attention_stats)),
        include_interaction_features=not bool(int(args.legacy_features)),
        include_lexical_features=not bool(int(args.disable_lexical_features)),
        evidence_topk=int(args.evidence_topk),
    )


def make_pairs(pos_idx: List[int], neg_idx: List[int], max_pairs: int, rng: random.Random) -> List[Tuple[int, int]]:
    if not pos_idx or not neg_idx:
        return []
    pairs = [(p, n) for p in pos_idx for n in neg_idx]
    if len(pairs) <= max_pairs:
        return pairs
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def select_negative_indices(
    neg_idx: List[int],
    *,
    difficulty: np.ndarray,
    strategy: str,
    hard_topn: int,
    rand_topn: int,
    rng: random.Random,
) -> List[int]:
    if not neg_idx:
        return []
    strat = str(strategy or "mixed").lower()
    if strat == "random":
        return list(neg_idx)

    hard_topn = max(0, int(hard_topn))
    rand_topn = max(0, int(rand_topn))
    hard_sorted = sorted(neg_idx, key=lambda i: float(difficulty[i]), reverse=True)
    hard = hard_sorted[: min(hard_topn, len(hard_sorted))] if hard_topn > 0 else []
    if strat == "hard":
        return hard
    if strat != "mixed":
        raise ValueError(f"Unknown neg strategy: {strategy}")

    hard_set = set(hard)
    remaining = [i for i in neg_idx if i not in hard_set]
    rng.shuffle(remaining)
    rand = remaining[: min(rand_topn, len(remaining))] if rand_topn > 0 else []
    return hard + rand


def _hardneg_difficulty(x: np.ndarray, spec: FeatureSpec, embedding_dim: int, name: str) -> np.ndarray:
    layout = feature_layout(spec, embedding_dim)
    topk_idx = list(layout["topk"])
    top1_idx = topk_idx[spec.topk_list.index(1)] if 1 in spec.topk_list else topk_idx[0]
    profile_idx = int(layout["profile_sim"])
    logn_idx = int(layout["log_npapers"])
    attn_idx = int(layout.get("attn_sim", profile_idx))
    lexical = layout.get("lexical") or {}
    lex_top1_bm25_idx = lexical.get("lex_top1_bm25")
    lex_topk_bm25_idx = lexical.get("lex_topk_bm25_mean")

    key = str(name or "hybrid").lower()
    if key in {"top1", "max", "max_sim"}:
        return x[:, top1_idx]
    if key in {"profile", "profile_sim"}:
        return x[:, profile_idx]
    if key in {"attn", "attention", "attn_sim"}:
        return x[:, attn_idx]
    if key in {"logn", "log_npapers"}:
        return x[:, logn_idx]
    if key in {"lex", "bm25", "lexical"} and lex_top1_bm25_idx is not None:
        return x[:, int(lex_top1_bm25_idx)]
    if key == "hybrid":
        parts = [x[:, top1_idx], x[:, profile_idx]]
        if "attn_sim" in layout:
            parts.append(x[:, attn_idx])
        if lex_top1_bm25_idx is not None:
            parts.append(x[:, int(lex_top1_bm25_idx)])
        if lex_topk_bm25_idx is not None:
            parts.append(x[:, int(lex_topk_bm25_idx)])
        return np.mean(np.stack(parts, axis=0), axis=0)
    raise ValueError(f"Unknown hardneg feature: {name}")


def _safe_git_info(repo_root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        inside = (
            subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        if inside != "true":
            return out
        out["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL).decode("utf-8").strip()
        out["status_porcelain"] = (
            subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return {}
    return out


def _write_run_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    cfg: TrainConfig,
    spec: FeatureSpec,
    feat_dim_value: int,
    embedding_dim: int,
    folds: int,
    eval_k: List[int],
    n_queries: int,
    git_info: Dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": list(getattr(args, "_argv", [])),
        "data": {
            "reviewers": str(args.reviewers),
            "queries": str(args.queries),
            "gold": str(args.gold),
            "paper_vec_cache": str(args.paper_vec_cache),
            "query_vec_cache": str(args.query_vec_cache),
        },
        "feature_spec": {
            "topk_list": list(spec.topk_list),
            "logmeanexp_tau_list": list(spec.logmeanexp_tau_list),
            "softmax_tau_list": list(spec.softmax_tau_list),
            "attention_tau": float(spec.attention_tau),
            "include_attention_stats": bool(spec.include_attention_stats),
            "include_interaction_features": bool(spec.include_interaction_features),
            "include_lexical_features": bool(spec.include_lexical_features),
            "evidence_topk": int(spec.evidence_topk),
        },
        "embedding_dim": int(embedding_dim),
        "feat_dim": int(feat_dim_value),
        "model": {
            "hidden": int(cfg.hidden),
            "dropout": float(cfg.dropout),
            "requested_head_mode": str(getattr(args, "head_mode", cfg.head_mode)),
            "head_mode": str(cfg.head_mode),
        },
        "train": {
            "requested_head_mode": str(getattr(args, "head_mode", cfg.head_mode)),
            "head_mode": str(cfg.head_mode),
            "folds": int(folds),
            "epochs": int(cfg.epochs),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "soft_pair_weight": float(cfg.soft_pair_weight),
            "hard_pair_weight": float(cfg.hard_pair_weight),
            "final_soft_coef": float(cfg.final_soft_coef),
            "final_hard_coef": float(cfg.final_hard_coef),
            "pointwise_weight": float(cfg.pointwise_weight),
            "listwise_weight": float(cfg.listwise_weight),
            "max_pairs_per_query": int(cfg.max_pairs_per_query),
            "neg_strategy": str(getattr(args, "neg_strategy", "mixed")),
            "hard_neg_feature": str(getattr(args, "hard_neg_feature", "hybrid")),
            "hard_neg_topn": int(getattr(args, "hard_neg_topn", 32)),
            "rand_neg_topn": int(getattr(args, "rand_neg_topn", 16)),
            "stage1_fuse_alpha": float(getattr(args, "stage1_fuse_alpha", 1.0)),
            "stage1_fuse_feature": str(getattr(args, "stage1_fuse_feature", "top3")),
            "seed": int(args.seed),
            "device": str(args.device),
        },
        "eval_k": eval_k,
        "n_queries": int(n_queries),
        "git": git_info,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)


def _open_pairs_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fh = path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            fh,
            fieldnames=["query_id", "reviewer_id", "prediction_score", "gt_grade", "gt_soft", "gt_hard", "rank", "fold"],
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
def eval_rows(
    model: PairwiseMLPReranker,
    rows: List[Dict],
    *,
    device: str,
    eval_k: List[int],
    stage1_fuse_alpha: float,
    head_mode: str,
    final_soft_coef: float,
    final_hard_coef: float,
):
    model.eval()
    soft_sums = {k: 0.0 for k in eval_k}
    hard_sums = {k: 0.0 for k in eval_k}
    n = 0
    for row in rows:
        x = torch.from_numpy(row["x"]).to(device)
        soft_logit, hard_logit = model(x)
        model_score = combine_logits(
            soft_logit,
            hard_logit,
            head_mode=head_mode,
            final_soft_coef=final_soft_coef,
            final_hard_coef=final_hard_coef,
        ).detach().cpu().numpy()
        score = fuse_model_and_stage1_scores(model_score, row.get("stage1_anchor"), stage1_fuse_alpha)
        order = np.argsort(-score, kind="mergesort")
        ranked = [row["cands"][i] for i in order.tolist()]
        soft_rel = set(row["soft_rel"])
        hard_rel = set(row["hard_rel"])
        for k in eval_k:
            topk = ranked[:k]
            soft_hits = sum(1 for rid in topk if rid in soft_rel)
            hard_hits = sum(1 for rid in topk if rid in hard_rel)
            soft_sums[k] += soft_hits / k
            hard_sums[k] += hard_hits / k
        n += 1
    if n == 0:
        return {k: 0.0 for k in eval_k}, {k: 0.0 for k in eval_k}, 0.0
    soft = {k: soft_sums[k] / n for k in eval_k}
    hard = {k: hard_sums[k] / n for k in eval_k}
    obj = 2.0 * soft[5] + hard[5] if 5 in eval_k else 0.0
    return soft, hard, obj


def _pointwise_bce_loss(logits: torch.Tensor, labels: np.ndarray, device: str) -> torch.Tensor:
    target = torch.from_numpy(labels.astype(np.float32)).to(device)
    pos = float(target.sum().item())
    neg = float(target.numel() - pos)
    pos_weight = torch.tensor((neg + 1.0) / (pos + 1.0), device=device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def _listwise_loss(logits: torch.Tensor, grades: np.ndarray, device: str) -> torch.Tensor:
    target = torch.from_numpy(grades.astype(np.float32)).to(device)
    if float(target.sum().item()) <= 0.0:
        return torch.tensor(0.0, device=device)
    target = target / target.sum().clamp(min=1e-8)
    return -(target * torch.log_softmax(logits, dim=0)).sum()


def train_one_fold(
    train_rows: List[Dict],
    val_rows: List[Dict],
    *,
    cfg: TrainConfig,
    device: str,
    seed: int,
    eval_k: List[int],
    neg_strategy: str,
    hard_neg_feature: str,
    hard_neg_topn: int,
    rand_neg_topn: int,
    spec: FeatureSpec,
    embedding_dim: int,
    stage1_fuse_alpha: float,
):
    rng = random.Random(int(seed))
    in_dim = int(train_rows[0]["x"].shape[1])
    model = PairwiseMLPReranker(in_dim=in_dim, hidden=cfg.hidden, dropout=cfg.dropout, head_mode=cfg.head_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_obj = -1e9
    best_state = None
    for _epoch in range(1, cfg.epochs + 1):
        model.train()
        rng.shuffle(train_rows)
        for row in train_rows:
            x = torch.from_numpy(row["x"]).to(device)
            y_soft = row["y_soft"]
            y_hard = row["y_hard"]
            y_grade = row["y_grade"]

            pos_soft = [i for i, v in enumerate(y_soft) if v == 1]
            neg_soft = [i for i, v in enumerate(y_soft) if v == 0]
            pos_hard = [i for i, v in enumerate(y_hard) if v == 1]
            neg_hard = [i for i, v in enumerate(y_hard) if v == 0]

            difficulty = _hardneg_difficulty(row["x"], spec, embedding_dim, hard_neg_feature)
            neg_soft_sel = select_negative_indices(neg_soft, difficulty=difficulty, strategy=neg_strategy, hard_topn=hard_neg_topn, rand_topn=rand_neg_topn, rng=rng)
            neg_hard_sel = select_negative_indices(neg_hard, difficulty=difficulty, strategy=neg_strategy, hard_topn=hard_neg_topn, rand_topn=rand_neg_topn, rng=rng)

            pairs_soft = make_pairs(pos_soft, neg_soft_sel, cfg.max_pairs_per_query, rng)
            pairs_hard = make_pairs(pos_hard, neg_hard_sel, cfg.max_pairs_per_query, rng)
            if not pairs_soft and not pairs_hard and cfg.pointwise_weight <= 0.0 and cfg.listwise_weight <= 0.0:
                continue

            soft_logit, hard_logit = model(x)
            final_logit = combine_logits(
                soft_logit,
                hard_logit,
                head_mode=cfg.head_mode,
                final_soft_coef=cfg.final_soft_coef,
                final_hard_coef=cfg.final_hard_coef,
            )
            loss = torch.tensor(0.0, device=device)
            if has_soft_head(cfg.head_mode) and pairs_soft:
                pi = torch.tensor([p for p, _ in pairs_soft], device=device, dtype=torch.long)
                ni = torch.tensor([n for _, n in pairs_soft], device=device, dtype=torch.long)
                loss = loss + cfg.soft_pair_weight * pairwise_logistic_loss(soft_logit[pi], soft_logit[ni])
            if has_hard_head(cfg.head_mode) and pairs_hard:
                pi = torch.tensor([p for p, _ in pairs_hard], device=device, dtype=torch.long)
                ni = torch.tensor([n for _, n in pairs_hard], device=device, dtype=torch.long)
                loss = loss + cfg.hard_pair_weight * pairwise_logistic_loss(hard_logit[pi], hard_logit[ni])
            if cfg.pointwise_weight > 0.0:
                pointwise_terms = []
                if has_soft_head(cfg.head_mode):
                    pointwise_terms.append(_pointwise_bce_loss(soft_logit, y_soft, device))
                if has_hard_head(cfg.head_mode):
                    pointwise_terms.append(_pointwise_bce_loss(hard_logit, y_hard, device))
                if pointwise_terms:
                    loss = loss + cfg.pointwise_weight * sum(pointwise_terms)
            if cfg.listwise_weight > 0.0:
                loss = loss + cfg.listwise_weight * _listwise_loss(final_logit, y_grade, device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        _soft, _hard, obj = eval_rows(
            model,
            val_rows,
            device=device,
            eval_k=eval_k,
            stage1_fuse_alpha=stage1_fuse_alpha,
            head_mode=cfg.head_mode,
            final_soft_coef=cfg.final_soft_coef,
            final_hard_coef=cfg.final_hard_coef,
        )
        if obj > best_obj:
            best_obj = obj
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    soft, hard, obj = eval_rows(
        model,
        val_rows,
        device=device,
        eval_k=eval_k,
        stage1_fuse_alpha=stage1_fuse_alpha,
        head_mode=cfg.head_mode,
        final_soft_coef=cfg.final_soft_coef,
        final_hard_coef=cfg.final_hard_coef,
    )
    return model, soft, hard, obj


def main() -> None:
    ap = argparse.ArgumentParser(description="K-fold CV training for pairwise reranker (query-level OoF evaluation).")
    ap.add_argument("--reviewers", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--paper_vec_cache", required=True)
    ap.add_argument("--query_vec_cache", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head_mode", choices=["dual", "dual_hard_main", "single_soft", "single_hard"], default="dual")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--soft_pair_weight", type=float, default=None)
    ap.add_argument("--hard_pair_weight", type=float, default=None)
    ap.add_argument("--final_soft_coef", type=float, default=None)
    ap.add_argument("--final_hard_coef", type=float, default=None)
    ap.add_argument("--pointwise_weight", type=float, default=0.25)
    ap.add_argument("--listwise_weight", type=float, default=0.2)
    ap.add_argument("--max_pairs_per_query", type=int, default=1024)
    ap.add_argument("--eval_k", default="5,10")
    ap.add_argument("--attention_tau", type=float, default=DEFAULT_SPEC.attention_tau)
    ap.add_argument("--legacy_features", type=int, default=0, help="Disable interaction-vector features.")
    ap.add_argument("--disable_attention_stats", type=int, default=0, help="Disable attention-derived scalar features.")
    ap.add_argument("--disable_lexical_features", type=int, default=0, help="Disable lexical/evidence features.")
    ap.add_argument("--evidence_topk", type=int, default=DEFAULT_SPEC.evidence_topk)
    ap.add_argument(
        "--neg_strategy",
        choices=["random", "hard", "mixed"],
        default="mixed",
        help="Negative sampling strategy within the judged candidate set for each query.",
    )
    ap.add_argument(
        "--hard_neg_feature",
        choices=["top1", "profile", "attn", "hybrid", "log_npapers", "lexical"],
        default="hybrid",
        help="Difficulty signal used for hard-negative mining (higher = harder).",
    )
    ap.add_argument("--hard_neg_topn", type=int, default=32, help="Only for neg_strategy in {hard,mixed}.")
    ap.add_argument("--rand_neg_topn", type=int, default=16, help="Only for neg_strategy=mixed.")
    ap.add_argument("--stage1_fuse_alpha", type=float, default=0.2, help="Final score = alpha * reranker + (1-alpha) * stage1 anchor after per-query min-max normalization.")
    ap.add_argument("--stage1_fuse_feature", default="top3", help="Anchor feature used for stage1/reranker score fusion.")
    ap.add_argument("--temporal_paper_filter_jsonl", default="", help="Optional query-aware reviewer-paper filter sidecar JSONL for temporal-safe feature construction.")
    ap.add_argument("--out_oof", default="output/kdd_oof_supervised.jsonl")
    ap.add_argument("--out_pairs", default="", help="Optional pair-level dump for case studies / error analysis.")
    ap.add_argument("--ckpt_dir", default="", help="Optional directory to save best fold checkpoints and a run manifest.")
    args = ap.parse_args()
    args._argv = list(getattr(__import__("sys"), "argv", []))

    repo_root = Path(__file__).resolve().parents[1]
    git_info = _safe_git_info(repo_root)
    reviewers = load_json(repo_root / args.reviewers)
    queries = load_json(repo_root / args.queries)
    gold = load_json(repo_root / args.gold)

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

    spec = build_feature_spec(args)
    feat_dim_value = feature_dim(spec, embedding_dim)

    q_rows = []
    for q in queries:
        qid = str(q.get("id") or "")
        if not qid or qid not in gold:
            continue
        qi = query_id_to_idx.get(qid)
        if qi is None:
            continue
        qvec = np.asarray(query_vecs[qi], dtype=np.float32)
        rels = gold.get(qid, {})
        cands = [str(rid) for rid in rels.keys() if str(rid) in rid_to_group]
        if not cands:
            continue

        x_raw, kept = compute_features_for_query_candidates(
            paper_vecs=paper_vecs,
            reviewer_to_paper_idxs=reviewer_to_paper_idxs,
            reviewer_ids=reviewer_ids,
            query_vec=qvec,
            candidate_rids=cands,
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
        stage1_anchor = extract_stage1_anchor_scores(x_raw, spec, embedding_dim, str(args.stage1_fuse_feature))
        x = per_query_zscore(x_raw)

        kept_set = set(kept)
        soft_rel = [rid for rid, s in rels.items() if int(s) >= 2 and str(rid) in kept_set]
        hard_rel = [rid for rid, s in rels.items() if int(s) == 3 and str(rid) in kept_set]
        labels_soft = np.array([1 if rid in set(soft_rel) else 0 for rid in kept], dtype=np.int64)
        labels_hard = np.array([1 if rid in set(hard_rel) else 0 for rid in kept], dtype=np.int64)
        labels_grade = np.array([int(rels.get(rid, 0)) for rid in kept], dtype=np.int64)

        q_rows.append(
            {
                "qid": qid,
                "x": x,
                "cands": kept,
                "stage1_anchor": stage1_anchor,
                "y_soft": labels_soft,
                "y_hard": labels_hard,
                "y_grade": labels_grade,
                "soft_rel": soft_rel,
                "hard_rel": hard_rel,
            }
        )

    if not q_rows:
        raise SystemExit("No training queries found. Check paths/caches.")

    folds = max(2, int(args.folds))
    eval_k = parse_k_list(args.eval_k)
    runtime_head_mode, soft_pair_weight, hard_pair_weight, final_soft_coef, final_hard_coef = resolve_head_hyperparams(
        str(args.head_mode),
        soft_pair_weight=args.soft_pair_weight,
        hard_pair_weight=args.hard_pair_weight,
        final_soft_coef=args.final_soft_coef,
        final_hard_coef=args.final_hard_coef,
    )
    cfg = TrainConfig(
        hidden=int(args.hidden),
        dropout=float(args.dropout),
        head_mode=runtime_head_mode,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        soft_pair_weight=soft_pair_weight,
        hard_pair_weight=hard_pair_weight,
        final_soft_coef=final_soft_coef,
        final_hard_coef=final_hard_coef,
        pointwise_weight=float(args.pointwise_weight),
        listwise_weight=float(args.listwise_weight),
        max_pairs_per_query=int(args.max_pairs_per_query),
    )

    ckpt_dir: Optional[Path] = None
    if args.ckpt_dir:
        ckpt_dir = (repo_root / args.ckpt_dir).resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _write_run_manifest(
            ckpt_dir / "run_manifest.json",
            args=args,
            cfg=cfg,
            spec=spec,
            feat_dim_value=feat_dim_value,
            embedding_dim=embedding_dim,
            folds=folds,
            eval_k=eval_k,
            n_queries=len(q_rows),
            git_info=git_info,
        )

    pairs_kind = ""
    pairs_fh = None
    pairs_writer = None
    out_pairs_path: Optional[Path] = None
    if args.out_pairs:
        out_pairs_path = repo_root / args.out_pairs
        pairs_kind, pairs_fh, pairs_writer = _open_pairs_writer(out_pairs_path)

    splits = [q_rows[i::folds] for i in range(folds)]
    oof: Dict[str, List[str]] = {}
    oof_score: Dict[str, List[float]] = {}
    for i in range(folds):
        val = splits[i]
        train = [r for j, sp in enumerate(splits) if j != i for r in sp]
        model, soft, hard, obj = train_one_fold(
            train,
            val,
            cfg=cfg,
            device=args.device,
            seed=int(args.seed) + i,
            eval_k=eval_k,
            neg_strategy=str(args.neg_strategy),
            hard_neg_feature=str(args.hard_neg_feature),
            hard_neg_topn=int(args.hard_neg_topn),
            rand_neg_topn=int(args.rand_neg_topn),
            spec=spec,
            embedding_dim=embedding_dim,
            stage1_fuse_alpha=float(args.stage1_fuse_alpha),
        )
        print(f"fold={i+1}/{folds} obj={obj:.4f} soft5={soft.get(5,0.0):.4f} hard5={hard.get(5,0.0):.4f}")

        if ckpt_dir is not None:
            torch.save(model.state_dict(), ckpt_dir / f"fold_{i+1:02d}.pth")

        model.eval()
        for row in val:
            x = torch.from_numpy(row["x"]).to(args.device)
            with torch.no_grad():
                s_soft, s_hard = model(x)
                model_score = combine_logits(
                    s_soft,
                    s_hard,
                    head_mode=cfg.head_mode,
                    final_soft_coef=cfg.final_soft_coef,
                    final_hard_coef=cfg.final_hard_coef,
                ).detach().cpu().numpy()
                fused_score = fuse_model_and_stage1_scores(model_score, row.get("stage1_anchor"), float(args.stage1_fuse_alpha))
                order = np.argsort(-fused_score, kind="mergesort")
                ranked = [row["cands"][k] for k in order.tolist()]
                oof[row["qid"]] = ranked
                oof_score[row["qid"]] = [float(fused_score[k]) for k in order.tolist()]

                if out_pairs_path is not None:
                    assert pairs_fh is not None
                    y_soft = row["y_soft"]
                    y_hard = row["y_hard"]
                    rels = gold.get(row["qid"], {})
                    for rank_idx, cand_pos in enumerate(order.tolist(), start=1):
                        rid = row["cands"][cand_pos]
                        _write_pair_row(
                            pairs_kind,
                            pairs_fh,
                            pairs_writer,
                            {
                                "query_id": row["qid"],
                                "reviewer_id": str(rid),
                                "prediction_score": float(fused_score[cand_pos]),
                                "gt_grade": int(rels.get(str(rid), 0)),
                                "gt_soft": int(y_soft[cand_pos]),
                                "gt_hard": int(y_hard[cand_pos]),
                                "rank": int(rank_idx),
                                "fold": int(i + 1),
                            },
                        )

    soft_sums = {k: 0.0 for k in eval_k}
    hard_sums = {k: 0.0 for k in eval_k}
    n = 0
    for row in q_rows:
        qid = row["qid"]
        ranked = oof.get(qid)
        if not ranked:
            continue
        soft_rel = set(row["soft_rel"])
        hard_rel = set(row["hard_rel"])
        for k in eval_k:
            topk = ranked[:k]
            soft_hits = sum(1 for rid in topk if rid in soft_rel)
            hard_hits = sum(1 for rid in topk if rid in hard_rel)
            soft_sums[k] += soft_hits / k
            hard_sums[k] += hard_hits / k
        n += 1

    for k in eval_k:
        print(f"OOF SoftP@{k}: {soft_sums[k]/n:.4f}  OOF HardP@{k}: {hard_sums[k]/n:.4f}  n={n}")
    if 5 in eval_k:
        obj = 2.0 * (soft_sums[5] / n) + (hard_sums[5] / n)
        print(f"OOF Objective(2*SoftP@5+HardP@5): {obj:.4f}")

    out_path = repo_root / args.out_oof
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qid, ranked in oof.items():
            f.write(json.dumps({"query_id": qid, "recommend_reviewer": ranked, "recommend_score": oof_score.get(qid, [])}, ensure_ascii=True) + "\n")
    print(f"Wrote OOF recs: {out_path}")

    if out_pairs_path is not None:
        assert pairs_fh is not None
        pairs_fh.close()
        print(f"Wrote pair dump: {out_pairs_path}")


if __name__ == "__main__":
    main()
