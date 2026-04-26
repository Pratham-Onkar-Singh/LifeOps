# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for LifeOps GRPO: 0–1 reward aggregation and uniform-action baselines."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scripts.rl_action_utils import resolve_allowed_actions


def weighted_reward_unit(
    env_u: float,
    fmt_u: float,
    w_env: float = 1.0 / 1.45,
    w_fmt: float = 0.45 / 1.45,
) -> float:
    """Same linear combination TRL uses when `reward_weights=(w_env, w_fmt)` on two reward heads."""
    return float(w_env * float(env_u) + w_fmt * float(fmt_u))


def uniform_action_baseline_stats(
    dataset: Sequence[Any],
    env_reward_fn: Callable[..., List[float]],
    format_reward_fn: Callable[..., List[float]],
    *,
    n_rows: int = 40,
    seed: int = 0,
    reward_weights: Tuple[float, float] = (1.0 / 1.45, 0.45 / 1.45),
    justification_text: str = "Uniform baseline policy evaluation.",
) -> Dict[str, Any]:
    """
    For each sampled row, average the **weighted** (env + format) reward over every
    allowed action with a fixed template completion. Mirrors a simple "pick uniformly
    among legal moves" policy — useful as a horizontal baseline on reward plots.
    """
    w_env, w_fmt = reward_weights
    n = len(dataset)
    if n <= 0:
        return {"baseline_mean": 0.0, "rows_used": 0, "per_row_means": []}

    rng = random.Random(seed)
    idxs = rng.sample(range(n), k=min(int(n_rows), n))
    per_row: List[float] = []

    for i in idxs:
        ex = dataset[i]
        messages = ex["prompt"]
        blob = ex.get("allowed_actions_json")
        allowed = resolve_allowed_actions(prompt=messages, allowed_actions_json=blob)
        if not allowed:
            continue
        row_scores: List[float] = []
        for action in sorted(allowed):
            completion = f"Action: {action}\nJustification: {justification_text}"
            eu = env_reward_fn(
                [completion],
                prompts=[messages],
                allowed_actions_json=[blob],
            )[0]
            fu = format_reward_fn([completion])[0]
            row_scores.append(weighted_reward_unit(eu, fu, w_env, w_fmt))
        per_row.append(sum(row_scores) / len(row_scores))

    overall = sum(per_row) / len(per_row) if per_row else 0.0
    return {
        "baseline_mean": float(overall),
        "rows_used": len(per_row),
        "per_row_means": per_row,
        "reward_weights": [w_env, w_fmt],
    }
