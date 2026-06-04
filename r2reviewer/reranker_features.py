import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

WORD_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "from", "at", "as",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "we", "our", "you", "your", "they", "their",
}


@dataclass
class FeatureSpec:
    topk_list: List[int]
    logmeanexp_tau_list: List[float]
    softmax_tau_list: List[float]
    attention_tau: float = 0.2
    include_attention_stats: bool = True
    include_interaction_features: bool = True
    include_lexical_features: bool = True
    evidence_topk: int = 3


LEGACY_SPEC = FeatureSpec(
    topk_list=[1, 3, 5, 10],
    logmeanexp_tau_list=[0.1, 0.3],
    softmax_tau_list=[0.1, 0.3],
    attention_tau=0.2,
    include_attention_stats=False,
    include_interaction_features=False,
    include_lexical_features=False,
    evidence_topk=3,
)

DEFAULT_SPEC = FeatureSpec(
    topk_list=[1, 3, 5, 10],
    logmeanexp_tau_list=[0.1, 0.3],
    softmax_tau_list=[0.1, 0.3],
    attention_tau=0.2,
    include_attention_stats=True,
    include_interaction_features=True,
    include_lexical_features=True,
    evidence_topk=3,
)


LEXICAL_FEATURE_NAMES = [
    "lex_top1_overlap",
    "lex_top1_jaccard",
    "lex_top1_bm25",
    "lex_topk_overlap_mean",
    "lex_topk_jaccard_mean",
    "lex_topk_bm25_mean",
    "lex_reviewer_union_overlap",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (repo_root / p)


def load_vec_cache(repo_root: Path, vec_path: str) -> Tuple[np.ndarray, List[str]]:
    vp = resolve_path(repo_root, vec_path)
    ids_path = Path(str(vp) + ".ids.json")
    vecs = np.load(vp, mmap_mode="r")
    with ids_path.open("r", encoding="utf-8") as f:
        ids = json.load(f)
    return vecs, [str(x) for x in ids]


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in WORD_RE.findall(text or "")]
    return [t for t in toks if t and t not in STOPWORDS]


def build_query_text_map(queries: List[Dict]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for q in queries:
        qid = str(q.get("id") or "")
        if not qid:
            continue
        out[qid] = ((q.get("title") or "").strip(), (q.get("abstract") or "").strip())
    return out


def build_reviewer_to_texts(reviewers: List[Dict], paper_id_to_idx: Dict[str, int]) -> Tuple[List[str], List[List[Tuple[str, str]]]]:
    reviewer_ids = [str(r.get("id")) for r in reviewers]
    rid_to_group = {rid: i for i, rid in enumerate(reviewer_ids)}
    out: List[List[Tuple[str, str]]] = [[] for _ in reviewer_ids]

    seen = set()
    for rev in reviewers:
        rid = str(rev.get("id"))
        group = rid_to_group.get(rid)
        if group is None:
            continue
        for p in rev.get("papers", []):
            pid = str(p.get("id") or "")
            if not pid:
                continue
            idx = paper_id_to_idx.get(pid)
            if idx is None:
                continue
            key = (group, idx)
            if key in seen:
                continue
            seen.add(key)
            out[group].append(((p.get("title") or "").strip(), (p.get("abstract") or "").strip()))
    return reviewer_ids, out


def build_reviewer_to_paper_ids(reviewers: List[Dict], paper_id_to_idx: Dict[str, int]) -> Tuple[List[str], List[List[str]]]:
    reviewer_ids = [str(r.get("id")) for r in reviewers]
    rid_to_group = {rid: i for i, rid in enumerate(reviewer_ids)}
    out: List[List[str]] = [[] for _ in reviewer_ids]

    seen = set()
    for rev in reviewers:
        rid = str(rev.get("id"))
        group = rid_to_group.get(rid)
        if group is None:
            continue
        for p in rev.get("papers", []):
            pid = str(p.get("id") or "")
            if not pid:
                continue
            idx = paper_id_to_idx.get(pid)
            if idx is None:
                continue
            key = (group, idx)
            if key in seen:
                continue
            seen.add(key)
            out[group].append(pid)
    return reviewer_ids, out


def build_lexical_corpus_stats(reviewer_to_texts: List[List[Tuple[str, str]]]) -> Dict[str, object]:
    df = Counter()
    total_len = 0
    n_docs = 0
    for paper_list in reviewer_to_texts:
        for title, abstract in paper_list:
            toks = tokenize(f"{title} {abstract}")
            if not toks:
                continue
            n_docs += 1
            total_len += len(toks)
            df.update(set(toks))
    return {
        "df": dict(df),
        "n_docs": int(n_docs),
        "avgdl": float(total_len) / max(1, n_docs),
    }


def load_temporal_query_reviewer_filter(path: Path) -> Dict[str, Dict[str, Set[str]]]:
    out: Dict[str, Dict[str, Set[str]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("query_id") or "")
            rid = str(row.get("reviewer_id") or "")
            if not qid or not rid:
                continue
            paper_ids = {str(pid) for pid in (row.get("paper_ids") or []) if str(pid)}
            if paper_ids:
                out[qid][rid] = paper_ids
    return {qid: dict(v) for qid, v in out.items()}


def scalar_feature_dim(spec: FeatureSpec) -> int:
    dim = len(spec.topk_list) + len(spec.logmeanexp_tau_list) + len(spec.softmax_tau_list)
    dim += 2
    if spec.include_attention_stats:
        dim += 2
    if spec.include_lexical_features:
        dim += len(LEXICAL_FEATURE_NAMES)
    return dim


def feature_dim(spec: FeatureSpec, embedding_dim: int) -> int:
    dim = scalar_feature_dim(spec)
    if spec.include_interaction_features:
        dim += int(embedding_dim) * 4
    return dim


def feature_layout(spec: FeatureSpec, embedding_dim: int) -> Dict[str, object]:
    cursor = 0
    layout: Dict[str, object] = {}

    layout["topk"] = list(range(cursor, cursor + len(spec.topk_list)))
    cursor += len(spec.topk_list)

    layout["logmeanexp"] = list(range(cursor, cursor + len(spec.logmeanexp_tau_list)))
    cursor += len(spec.logmeanexp_tau_list)

    layout["softmax"] = list(range(cursor, cursor + len(spec.softmax_tau_list)))
    cursor += len(spec.softmax_tau_list)

    layout["profile_sim"] = cursor
    cursor += 1

    if spec.include_attention_stats:
        layout["attn_sim"] = cursor
        cursor += 1
        layout["attn_max_weight"] = cursor
        cursor += 1

    layout["log_npapers"] = cursor
    cursor += 1

    if spec.include_lexical_features:
        lexical_map: Dict[str, int] = {}
        for name in LEXICAL_FEATURE_NAMES:
            lexical_map[name] = cursor
            cursor += 1
        layout["lexical"] = lexical_map

    if spec.include_interaction_features:
        emb_dim = int(embedding_dim)
        for name in ("profile_vec", "attn_vec", "attn_hadamard", "attn_absdiff"):
            layout[name] = slice(cursor, cursor + emb_dim)
            cursor += emb_dim

    layout["total_dim"] = cursor
    return layout


def feature_spec_from_meta(raw: Dict, *, feat_dim_hint: int | None = None, embedding_dim: int | None = None) -> FeatureSpec:
    raw = raw or {}
    spec = FeatureSpec(
        topk_list=list(raw.get("topk_list", DEFAULT_SPEC.topk_list)),
        logmeanexp_tau_list=list(raw.get("logmeanexp_tau_list", DEFAULT_SPEC.logmeanexp_tau_list)),
        softmax_tau_list=list(raw.get("softmax_tau_list", DEFAULT_SPEC.softmax_tau_list)),
        attention_tau=float(raw.get("attention_tau", DEFAULT_SPEC.attention_tau)),
        include_attention_stats=bool(raw.get("include_attention_stats", DEFAULT_SPEC.include_attention_stats)),
        include_interaction_features=bool(raw.get("include_interaction_features", DEFAULT_SPEC.include_interaction_features)),
        include_lexical_features=bool(raw.get("include_lexical_features", DEFAULT_SPEC.include_lexical_features)),
        evidence_topk=int(raw.get("evidence_topk", DEFAULT_SPEC.evidence_topk)),
    )

    if feat_dim_hint is not None and embedding_dim is not None and int(feat_dim_hint) == feature_dim(spec, int(embedding_dim)):
        return spec

    if "include_attention_stats" not in raw or "include_interaction_features" not in raw or "include_lexical_features" not in raw:
        legacy_dim = len(spec.topk_list) + len(spec.logmeanexp_tau_list) + len(spec.softmax_tau_list) + 2
        attn_only_dim = legacy_dim + 2
        if feat_dim_hint is not None:
            feat_dim_hint = int(feat_dim_hint)
            if feat_dim_hint == legacy_dim:
                spec.include_attention_stats = False
                spec.include_interaction_features = False
                spec.include_lexical_features = False
            elif feat_dim_hint == attn_only_dim:
                spec.include_attention_stats = True
                spec.include_interaction_features = False
                spec.include_lexical_features = False
    return spec


def build_reviewer_paper_index(reviewers: List[Dict], paper_id_to_idx: Dict[str, int]) -> Tuple[List[str], List[int]]:
    reviewer_ids = [str(r.get("id")) for r in reviewers]
    rid_to_group = {rid: i for i, rid in enumerate(reviewer_ids)}

    paper_owner = [0] * len(paper_id_to_idx)
    seen = np.zeros((len(paper_id_to_idx),), dtype=np.uint8)
    group_sizes = [0] * len(reviewer_ids)

    for rev in reviewers:
        rid = str(rev.get("id"))
        g = rid_to_group.get(rid)
        if g is None:
            continue
        for p in rev.get("papers", []):
            pid = str(p.get("id") or "")
            if not pid:
                continue
            idx = paper_id_to_idx.get(pid)
            if idx is None:
                continue
            if seen[idx]:
                continue
            seen[idx] = 1
            paper_owner[idx] = g
            group_sizes[g] += 1

    return reviewer_ids, paper_owner, group_sizes


def build_reviewer_to_paper_idxs(reviewers: List[Dict], paper_id_to_idx: Dict[str, int]) -> Tuple[List[str], List[List[int]]]:
    reviewer_ids = [str(r.get("id")) for r in reviewers]
    rid_to_group = {rid: i for i, rid in enumerate(reviewer_ids)}
    out: List[List[int]] = [[] for _ in reviewer_ids]

    seen = set()
    for rev in reviewers:
        rid = str(rev.get("id"))
        g = rid_to_group.get(rid)
        if g is None:
            continue
        for p in rev.get("papers", []):
            pid = str(p.get("id") or "")
            if not pid:
                continue
            idx = paper_id_to_idx.get(pid)
            if idx is None:
                continue
            key = (g, idx)
            if key in seen:
                continue
            seen.add(key)
            out[g].append(int(idx))
    return reviewer_ids, out


def _logmeanexp_torch(x: torch.Tensor, tau: float) -> torch.Tensor:
    t = float(tau)
    if t <= 0:
        raise ValueError("tau must be > 0")
    n = max(1, int(x.numel()))
    return t * (torch.logsumexp(x / t, dim=0) - math.log(n))


def _softmax_mean_torch(x: torch.Tensor, tau: float) -> torch.Tensor:
    t = float(tau)
    if t <= 0:
        raise ValueError("tau must be > 0")
    w = torch.softmax(x / t, dim=0)
    return torch.sum(w * x, dim=0)


def _l2_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _l2_normalize_vec(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm().clamp(min=1e-8)


def _token_overlap(query_terms: List[str], doc_terms: List[str]) -> float:
    if not query_terms:
        return 0.0
    qset = set(query_terms)
    dset = set(doc_terms)
    return len(qset & dset) / max(1, len(qset))


def _jaccard(query_terms: List[str], doc_terms: List[str]) -> float:
    qset = set(query_terms)
    dset = set(doc_terms)
    union = qset | dset
    if not union:
        return 0.0
    return len(qset & dset) / len(union)


def _bm25_score(query_terms: List[str], doc_terms: List[str], lexical_stats: Optional[Dict[str, object]]) -> float:
    if not query_terms or not doc_terms or not lexical_stats:
        return 0.0
    df = lexical_stats.get("df") or {}
    n_docs = int(lexical_stats.get("n_docs") or 0)
    avgdl = float(lexical_stats.get("avgdl") or 0.0)
    if n_docs <= 0 or avgdl <= 0.0:
        return 0.0

    tf = Counter(doc_terms)
    doc_len = len(doc_terms)
    k1 = 1.2
    b = 0.75
    score = 0.0
    for term in set(query_terms):
        freq = tf.get(term, 0)
        if freq <= 0:
            continue
        df_t = int(df.get(term, 0))
        idf = math.log(1.0 + (n_docs - df_t + 0.5) / (df_t + 0.5))
        denom = freq + k1 * (1.0 - b + b * doc_len / max(avgdl, 1e-8))
        score += idf * (freq * (k1 + 1.0)) / max(denom, 1e-8)
    return float(score)


def _compute_lexical_feature_block(
    *,
    query_text: Tuple[str, str],
    paper_texts: List[Tuple[str, str]],
    evidence_local_idxs: List[int],
    lexical_stats: Optional[Dict[str, object]],
) -> List[float]:
    title, abstract = query_text
    q_terms = tokenize(f"{title} {abstract}")
    if not q_terms or not paper_texts or not evidence_local_idxs:
        return [0.0 for _ in LEXICAL_FEATURE_NAMES]

    top_docs = []
    union_terms: List[str] = []
    for local_idx in evidence_local_idxs:
        if local_idx < 0 or local_idx >= len(paper_texts):
            continue
        p_title, p_abs = paper_texts[local_idx]
        doc_terms = tokenize(f"{p_title} {p_abs}")
        top_docs.append(doc_terms)
        union_terms.extend(doc_terms)
    if not top_docs:
        return [0.0 for _ in LEXICAL_FEATURE_NAMES]

    top1_terms = top_docs[0]
    overlaps = [_token_overlap(q_terms, d) for d in top_docs]
    jaccards = [_jaccard(q_terms, d) for d in top_docs]
    bm25s = [_bm25_score(q_terms, d, lexical_stats) for d in top_docs]
    return [
        _token_overlap(q_terms, top1_terms),
        _jaccard(q_terms, top1_terms),
        _bm25_score(q_terms, top1_terms, lexical_stats),
        float(np.mean(overlaps)),
        float(np.mean(jaccards)),
        float(np.mean(bm25s)),
        _token_overlap(q_terms, union_terms),
    ]


def compute_features_for_query_candidates(
    *,
    paper_vecs: np.ndarray,
    reviewer_to_paper_idxs: List[List[int]],
    reviewer_ids: List[str],
    query_vec: np.ndarray,
    candidate_rids: List[str],
    rid_to_group: Dict[str, int],
    spec: FeatureSpec = DEFAULT_SPEC,
    device: str = "cuda",
    reviewer_to_texts: Optional[List[List[Tuple[str, str]]]] = None,
    reviewer_to_paper_ids: Optional[List[List[str]]] = None,
    query_text: Optional[Tuple[str, str]] = None,
    lexical_stats: Optional[Dict[str, object]] = None,
    query_id: str = "",
    temporal_query_reviewer_filter: Optional[Dict[str, Dict[str, Set[str]]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    cand_groups = []
    kept = []
    for rid in candidate_rids:
        g = rid_to_group.get(str(rid))
        if g is None:
            continue
        cand_groups.append(int(g))
        kept.append(str(rid))

    emb_dim = int(np.asarray(query_vec, dtype=np.float32).shape[0])
    full_dim = feature_dim(spec, emb_dim)
    if not kept:
        return np.zeros((0, full_dim), dtype=np.float32), []

    q = torch.from_numpy(np.asarray(query_vec, dtype=np.float32).copy()).to(device)
    q = _l2_normalize_vec(q)

    rows: List[np.ndarray] = []
    tau = max(float(spec.attention_tau), 1e-4)
    for row_idx, g in enumerate(cand_groups):
        rid = kept[row_idx]
        idxs_all = reviewer_to_paper_idxs[g] if 0 <= g < len(reviewer_to_paper_idxs) else []
        idxs = idxs_all
        paper_texts = reviewer_to_texts[g] if reviewer_to_texts is not None and 0 <= g < len(reviewer_to_texts) else None
        if temporal_query_reviewer_filter and query_id:
            removed_ids = ((temporal_query_reviewer_filter.get(str(query_id)) or {}).get(str(rid)) or set())
            if removed_ids and reviewer_to_paper_ids is not None and 0 <= g < len(reviewer_to_paper_ids):
                kept_local = [i for i, pid in enumerate(reviewer_to_paper_ids[g]) if pid not in removed_ids]
                idxs = [idxs_all[i] for i in kept_local if i < len(idxs_all)]
                if paper_texts is not None:
                    paper_texts = [paper_texts[i] for i in kept_local if i < len(paper_texts)]
        if not idxs:
            rows.append(np.zeros((full_dim,), dtype=np.float32))
            continue

        pv = np.asarray(paper_vecs[idxs], dtype=np.float32).copy()
        pv_t = torch.from_numpy(pv).to(device)
        pv_t = _l2_normalize_rows(pv_t)
        sims = pv_t @ q

        feats: List[float] = []
        for k in spec.topk_list:
            kk = min(max(1, int(k)), int(sims.shape[0]))
            if kk == 1:
                feats.append(float(torch.max(sims).item()))
            else:
                vals = torch.topk(sims, k=kk, largest=True).values
                feats.append(float(vals.mean().item()))

        for tau_lme in spec.logmeanexp_tau_list:
            feats.append(float(_logmeanexp_torch(sims, tau_lme).item()))
        for tau_smx in spec.softmax_tau_list:
            feats.append(float(_softmax_mean_torch(sims, tau_smx).item()))

        profile_vec = _l2_normalize_vec(pv_t.mean(dim=0))
        feats.append(float((profile_vec @ q).item()))

        attn_w = None
        attn_vec = profile_vec
        if spec.include_attention_stats or spec.include_interaction_features:
            attn_w = torch.softmax(sims / tau, dim=0)
            attn_vec = _l2_normalize_vec(torch.sum(attn_w.unsqueeze(1) * pv_t, dim=0))

        if spec.include_attention_stats:
            feats.append(float((attn_vec @ q).item()))
            feats.append(float(torch.max(attn_w).item() if attn_w is not None else 0.0))

        n_p = max(1, int(sims.shape[0]))
        feats.append(float(math.log(n_p)))

        if spec.include_lexical_features:
            lexical_block = [0.0 for _ in LEXICAL_FEATURE_NAMES]
            if query_text is not None and paper_texts is not None:
                kk = min(max(1, int(spec.evidence_topk)), int(sims.shape[0]))
                evidence_local_idxs = torch.topk(sims, k=kk, largest=True).indices.detach().cpu().tolist()
                lexical_block = _compute_lexical_feature_block(
                    query_text=query_text,
                    paper_texts=paper_texts,
                    evidence_local_idxs=evidence_local_idxs,
                    lexical_stats=lexical_stats,
                )
            feats.extend(lexical_block)

        if spec.include_interaction_features:
            inter = torch.cat([profile_vec, attn_vec, q * attn_vec, torch.abs(q - attn_vec)], dim=0)
            feats.extend(inter.detach().cpu().numpy().astype(np.float32).tolist())

        rows.append(np.asarray(feats, dtype=np.float32))

    return np.vstack(rows).astype(np.float32, copy=False), kept


def per_query_zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd



def extract_stage1_anchor_scores(x_raw: np.ndarray, spec: FeatureSpec, embedding_dim: int, name: str = "top3") -> np.ndarray:
    if x_raw.size == 0:
        return np.zeros((0,), dtype=np.float32)

    layout = feature_layout(spec, embedding_dim)
    topk_idx = list(layout["topk"])
    topk_map = {f"top{k}": int(topk_idx[i]) for i, k in enumerate(spec.topk_list)}
    topk_map.update({f"topk{k}": int(topk_idx[i]) for i, k in enumerate(spec.topk_list)})

    key = str(name or "top3").lower().replace("_", "")
    if key in {"max", "maxsim"}:
        key = "top1"
    elif key.isdigit():
        key = f"top{key}"

    profile_idx = int(layout["profile_sim"])
    attn_idx = int(layout.get("attn_sim", profile_idx))

    if key in topk_map:
        return np.asarray(x_raw[:, topk_map[key]], dtype=np.float32)
    if key in {"profile", "profilesim"}:
        return np.asarray(x_raw[:, profile_idx], dtype=np.float32)
    if key in {"attn", "attention", "attnsim"}:
        return np.asarray(x_raw[:, attn_idx], dtype=np.float32)
    if key == "hybrid":
        parts = []
        if "top3" in topk_map:
            parts.append(x_raw[:, topk_map["top3"]])
        elif "top1" in topk_map:
            parts.append(x_raw[:, topk_map["top1"]])
        parts.append(x_raw[:, profile_idx])
        if "attn_sim" in layout:
            parts.append(x_raw[:, attn_idx])
        return np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)
    raise ValueError(f"Unknown stage1 fuse feature: {name}")



def minmax_normalize_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)



def fuse_model_and_stage1_scores(model_scores: np.ndarray, stage1_scores: Optional[np.ndarray], alpha: float) -> np.ndarray:
    model_arr = np.asarray(model_scores, dtype=np.float32)
    if model_arr.size == 0:
        return model_arr
    if stage1_scores is None or float(alpha) >= 1.0:
        return model_arr
    if float(alpha) <= 0.0:
        return minmax_normalize_1d(np.asarray(stage1_scores, dtype=np.float32))
    stage1_arr = np.asarray(stage1_scores, dtype=np.float32)
    if stage1_arr.shape != model_arr.shape:
        raise ValueError(f"Shape mismatch in score fusion: model={model_arr.shape} stage1={stage1_arr.shape}")
    model_norm = minmax_normalize_1d(model_arr)
    stage1_norm = minmax_normalize_1d(stage1_arr)
    return (float(alpha) * model_norm + (1.0 - float(alpha)) * stage1_norm).astype(np.float32)
