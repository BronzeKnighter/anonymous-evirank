from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from adapters import AutoAdapterModel
from tqdm import tqdm
from transformers import AutoTokenizer


def normalize_vecs(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom
    return last_hidden_state[:, 0, :]


@contextmanager
def inference_context():
    if hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            yield
    else:
        with torch.no_grad():
            yield


def build_encoder(model_name: str, adapter_name: str, device: str, pooling: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    adapter = None
    if adapter_name:
        adapter = model.load_adapter(adapter_name, source="hf", set_active=True)
    model.to(device)
    if adapter:
        model.set_active_adapters(adapter)
        model._forced_adapter = adapter
    model.eval()
    return tokenizer, model


def make_text(title: str, abstract: str, *, view_mode: str, title_boost: int) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if view_mode == "title":
        return title
    if view_mode == "abstract":
        return abstract
    boosted_title = " ".join([title] * max(1, int(title_boost))).strip()
    return f"{boosted_title} {abstract}".strip()


def encode_texts(tokenizer, model, texts: List[str], batch_size: int, device: str, pooling: str, desc: str) -> np.ndarray:
    outputs = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with inference_context():
            forced = getattr(model, "_forced_adapter", None)
            if forced:
                model.set_active_adapters(forced)
            out = model(**inputs)
            emb = pool_embeddings(out.last_hidden_state, inputs["attention_mask"], pooling)
        outputs.append(emb.detach().cpu().numpy())
    return np.vstack(outputs).astype("float32")


def cache_ids_path(vec_path: Path) -> Path:
    return vec_path.with_suffix(vec_path.suffix + ".ids.json")


def load_or_encode(
    *,
    ids: List[str],
    texts: List[str],
    vec_path: Path,
    tokenizer,
    model,
    batch_size: int,
    device: str,
    pooling: str,
    desc: str,
    force_recompute: bool,
) -> np.ndarray:
    ids_path = cache_ids_path(vec_path)
    if vec_path.exists() and ids_path.exists() and not force_recompute:
        with ids_path.open("r", encoding="utf-8") as f:
            cached_ids = [str(x) for x in json.load(f)]
        if cached_ids == ids:
            return normalize_vecs(np.load(vec_path, mmap_mode="r"))
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    vecs = encode_texts(tokenizer, model, texts, batch_size, device, pooling, desc)
    vecs = normalize_vecs(vecs)
    np.save(vec_path, vecs)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    return vecs


def reduce_reviewer_scores(
    paper_scores: np.ndarray,
    reviewer_to_paper_idxs: List[List[int]],
    *,
    pooling: str,
    topk: int,
    alpha: float,
) -> np.ndarray:
    n_reviewers = len(reviewer_to_paper_idxs)
    n_queries = paper_scores.shape[1]
    out = np.full((n_reviewers, n_queries), -1e9, dtype="float32")
    topk = max(1, int(topk))
    alpha = float(alpha)
    for ridx, pidxs in enumerate(reviewer_to_paper_idxs):
        if not pidxs:
            continue
        vals = paper_scores[np.asarray(pidxs, dtype=np.int64), :]
        max_scores = vals.max(axis=0)
        if pooling == "max":
            out[ridx, :] = max_scores
        else:
            k = min(topk, vals.shape[0])
            top_vals = np.partition(vals, vals.shape[0] - k, axis=0)[-k:, :]
            topk_mean = top_vals.mean(axis=0)
            if pooling == "topk_mean":
                out[ridx, :] = topk_mean
            elif pooling == "hybrid":
                out[ridx, :] = alpha * max_scores + (1.0 - alpha) * topk_mean
            else:
                raise ValueError(f"Unsupported reviewer_pooling for EviRank release: {pooling}")
    return out


def build_prx_scores(args, repo_root: Path, on_chunk=None) -> Dict[str, Dict[str, float]]:
    reviewers_path = repo_root / args.reviewers
    queries_path = repo_root / args.queries
    gold_path = repo_root / args.gold

    with reviewers_path.open("r", encoding="utf-8") as f:
        reviewers = json.load(f)
    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)
    with gold_path.open("r", encoding="utf-8") as f:
        gold = json.load(f)

    allowed_reviewers = {str(rid) for rels in gold.values() for rid in rels.keys()}
    reviewers = [r for r in reviewers if str(r.get("id")) in allowed_reviewers]

    paper_ids: List[str] = []
    paper_texts: List[str] = []
    reviewer_ids: List[str] = []
    reviewer_to_paper_idxs: List[List[int]] = []
    seen_papers: Dict[str, int] = {}
    for reviewer in reviewers:
        reviewer_ids.append(str(reviewer.get("id")))
        cur_idxs: List[int] = []
        for paper in reviewer.get("papers", []):
            pid = str(paper.get("id") or "")
            if not pid:
                continue
            if pid not in seen_papers:
                seen_papers[pid] = len(paper_ids)
                paper_ids.append(pid)
                paper_texts.append(
                    make_text(
                        paper.get("title") or "",
                        paper.get("abstract") or "",
                        view_mode=args.view_mode,
                        title_boost=args.title_boost,
                    )
                )
            cur_idxs.append(seen_papers[pid])
        reviewer_to_paper_idxs.append(cur_idxs)

    query_ids = [str(q.get("id")) for q in queries]
    query_texts = [
        make_text(q.get("title") or "", q.get("abstract") or "", view_mode=args.view_mode, title_boost=args.title_boost)
        for q in queries
    ]

    if not torch.cuda.is_available() and str(getattr(args, "device", "cuda")) == "cuda":
        raise RuntimeError("CUDA requested but not available. Set DEVICE=cpu only for small smoke tests.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = build_encoder(args.model_name, args.adapter_name, device, args.pooling)

    paper_vec_path = repo_root / args.paper_vec_cache
    query_vec_path = repo_root / args.query_vec_cache
    force = bool(int(getattr(args, "force_recompute", 0)))
    paper_vecs = load_or_encode(
        ids=paper_ids,
        texts=paper_texts,
        vec_path=paper_vec_path,
        tokenizer=tokenizer,
        model=model,
        batch_size=int(args.batch_size),
        device=device,
        pooling=args.pooling,
        desc="Encode reviewer papers",
        force_recompute=force,
    )
    query_vecs = load_or_encode(
        ids=query_ids,
        texts=query_texts,
        vec_path=query_vec_path,
        tokenizer=tokenizer,
        model=model,
        batch_size=int(args.batch_size),
        device=device,
        pooling=args.pooling,
        desc="Encode query papers",
        force_recompute=force,
    )

    score_chunk = int(getattr(args, "score_chunk", 0) or 0)
    if score_chunk <= 0:
        score_chunk = len(query_ids)

    all_scores: Dict[str, Dict[str, float]] = {}
    for start in tqdm(range(0, len(query_ids), score_chunk), desc="Score Stage1", unit="chunk"):
        end = min(start + score_chunk, len(query_ids))
        qids = query_ids[start:end]
        qvec = np.asarray(query_vecs[start:end], dtype="float32")
        paper_scores = np.asarray(paper_vecs, dtype="float32") @ qvec.T
        reviewer_scores = reduce_reviewer_scores(
            paper_scores,
            reviewer_to_paper_idxs,
            pooling=args.reviewer_pooling,
            topk=args.reviewer_topk,
            alpha=args.reviewer_alpha,
        )
        if on_chunk is not None:
            on_chunk(qids, reviewer_scores, reviewer_ids)
        else:
            for j, qid in enumerate(qids):
                all_scores[qid] = {rid: float(reviewer_scores[i, j]) for i, rid in enumerate(reviewer_ids)}
    return all_scores

