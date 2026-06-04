from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


SUPPORTED_HEAD_MODES = {"dual", "dual_hard_main", "single_soft", "single_hard"}


def normalize_head_mode(head_mode: str) -> str:
    mode = str(head_mode or "dual")
    if mode == "dual_hard_main":
        return "dual"
    return mode


def has_soft_head(head_mode: str) -> bool:
    return normalize_head_mode(head_mode) in {"dual", "single_soft"}


def has_hard_head(head_mode: str) -> bool:
    return normalize_head_mode(head_mode) in {"dual", "single_hard"}


def resolve_head_hyperparams(
    head_mode: str,
    *,
    soft_pair_weight: float | None = None,
    hard_pair_weight: float | None = None,
    final_soft_coef: float | None = None,
    final_hard_coef: float | None = None,
) -> tuple[str, float, float, float, float]:
    mode_raw = str(head_mode or "dual")
    mode = normalize_head_mode(mode_raw)

    if mode_raw == "dual_hard_main":
        soft_pair_default = 1.0
        hard_pair_default = 2.0
        final_soft_default = 0.3
        final_hard_default = 1.0
    else:
        soft_pair_default = 2.0
        hard_pair_default = 1.0
        final_soft_default = 2.0
        final_hard_default = 1.0

    soft_pair = soft_pair_default if soft_pair_weight is None else float(soft_pair_weight)
    hard_pair = hard_pair_default if hard_pair_weight is None else float(hard_pair_weight)
    soft_coef = final_soft_default if final_soft_coef is None else float(final_soft_coef)
    hard_coef = final_hard_default if final_hard_coef is None else float(final_hard_coef)
    return mode, soft_pair, hard_pair, soft_coef, hard_coef


class PairwiseMLPReranker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1, head_mode: str = "dual"):
        super().__init__()
        self.head_mode_raw = str(head_mode)
        self.head_mode = normalize_head_mode(head_mode)
        self.input_norm = nn.LayerNorm(in_dim)
        self.skip = nn.Linear(in_dim, hidden)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.soft_head = None
        self.hard_head = None
        if has_soft_head(head_mode):
            self.soft_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        if has_hard_head(head_mode):
            self.hard_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        if self.head_mode_raw not in SUPPORTED_HEAD_MODES:
            raise ValueError(f"Unsupported head_mode: {self.head_mode}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        h = self.skip(x) + self.trunk(x)
        if self.soft_head is not None:
            soft = self.soft_head(h).squeeze(-1)
        else:
            soft = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        if self.hard_head is not None:
            hard = self.hard_head(h).squeeze(-1)
        else:
            hard = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        return soft, hard


@dataclass
class TrainConfig:
    hidden: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    soft_pair_weight: float = 2.0
    hard_pair_weight: float = 1.0
    final_soft_coef: float = 2.0
    final_hard_coef: float = 1.0
    pointwise_weight: float = 0.25
    listwise_weight: float = 0.2
    max_pairs_per_query: int = 512
    head_mode: str = "dual"


def pairwise_logistic_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-(pos - neg)).mean()


def combine_logits(
    soft: torch.Tensor,
    hard: torch.Tensor,
    head_mode: str = "dual",
    *,
    final_soft_coef: float | None = None,
    final_hard_coef: float | None = None,
) -> torch.Tensor:
    mode = normalize_head_mode(head_mode)
    if mode == "dual":
        soft_coef = 2.0 if final_soft_coef is None else float(final_soft_coef)
        hard_coef = 1.0 if final_hard_coef is None else float(final_hard_coef)
        return soft_coef * soft + hard_coef * hard
    if mode == "single_soft":
        return soft
    if mode == "single_hard":
        return hard
    raise ValueError(f"Unsupported head_mode: {head_mode}")
