# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple


def coerce_prompt_text(prompt: object) -> Optional[str]:
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        chunks: list[str] = []
        for msg in prompt:
            role = str(msg.get("role", "")).strip()
            content = msg.get("content", "")
            if isinstance(content, str):
                chunks.append(f"{role}:\n{content}".strip())
            else:
                chunks.append(f"{role}:\n{str(content)}".strip())
        return "\n\n".join([c for c in chunks if c])
    return str(prompt)


def parse_allowed_actions(prompt: object) -> Optional[set[str]]:
    prompt_txt = coerce_prompt_text(prompt)
    if not prompt_txt:
        return None
    m = re.search(r"Allowed Actions:\s*([^\n]+)", prompt_txt, re.IGNORECASE)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return set(parts) if parts else None


def _phrase_to_snake(phrase: str) -> str:
    s = phrase.strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2019']", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def extract_raw_action_phrase(text: str) -> Optional[str]:
    """
    Extract the human-readable action phrase after 'Action:' from the first line.
    This intentionally allows free-form titles (models often ignore strict enums early).
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return None
    m = re.match(r"^\s*Action\s*:\s*(.+?)\s*$", lines[0], re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def map_phrase_to_allowed_action(phrase: str, allowed: Optional[Iterable[str]]) -> Tuple[Optional[str], float]:
    """
    Map a free-form phrase to a concrete allowed enum string.

    Returns (mapped_action_or_none, mapping_penalty).
    """
    if not phrase:
        return None, 0.0

    snake = _phrase_to_snake(phrase)
    if not snake:
        return None, 0.0

    allowed_list = list(allowed) if allowed is not None else []

    if allowed is not None:
        if snake in allowed:
            return snake, 0.0

        # Direct substring hits against allowed tokens (high precision heuristics).
        for a in allowed_list:
            if a in snake or snake in a:
                return a, -0.15

    # Keyword routing (only returns actions that exist in allowed when provided).
    # Order matters: put high-signal life/home cues before generic "work" cues.
    rules: list[tuple[str, str]] = [
        ("wedding|friend's wedding|best friend", "go_to_family_event"),
        ("birthday|partner|dinner|family", "go_to_family_event"),
        ("child|school performance|kid", "go_to_family_event"),
        ("vet|plumber|burst pipe|home", "go_to_family_event"),
        ("delegate|hand off|reassign", "delegate_work"),
        ("ask for|understanding|communicate|call|message|text", "ask_for_understanding"),
        ("cancel|decline|skip plans", "cancel_plans"),
        ("rest|sleep|break", "rest"),
        ("exercise|gym|run", "exercise"),
        ("spend|buy|money|budget", "spend_money"),
        ("skip work|skip", "skip_work"),
        ("work hard|grind|focus on work", "work_hard"),
        ("stay late|work late|deadline|prep|review|project|client|career|work", "stay_late_work"),
        ("nothing|no action", "do_nothing"),
    ]

    for pat, action in rules:
        if re.search(pat, snake, re.IGNORECASE):
            if allowed is None or action in allowed:
                return action, -0.25

    # Last resort: overlap scoring on token pieces.
    if allowed is not None and allowed_list:
        pieces = [p for p in snake.split("_") if len(p) >= 4]
        best = None
        best_score = 0
        for a in allowed_list:
            score = 0
            for p in pieces:
                if p and p in a:
                    score += len(p)
            if score > best_score:
                best_score = score
                best = a
        if best is not None and best_score >= 4:
            return best, -0.45

    return None, 0.0


def action_line_is_snake_enum(action_phrase: str) -> bool:
    s = action_phrase.strip()
    if not s:
        return False
    if " " in s:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_]+", s):
        return False
    return "_" in s and s == s.lower()
