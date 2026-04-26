# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
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


def parse_allowed_actions_json(blob: object) -> Optional[set[str]]:
    """
    Parse allowed actions from a JSON array string, e.g. '["stay_late_work", ...]'.
    """
    if blob is None:
        return None
    if isinstance(blob, list):
        items = [str(x).strip() for x in blob if str(x).strip()]
        return set(items) if items else None
    if not isinstance(blob, str) or not blob.strip():
        return None
    try:
        data = json.loads(blob)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    items = [str(x).strip() for x in data if str(x).strip()]
    return set(items) if items else None


def resolve_allowed_actions(
    *,
    prompt: object,
    allowed_actions_json: object = None,
) -> Optional[set[str]]:
    """
    Prefer structured JSON (dataset column). Fall back to parsing the prompt text.
    """
    allowed = parse_allowed_actions_json(allowed_actions_json)
    if allowed:
        return allowed
    return parse_allowed_actions(prompt)


# NOTE: Do not include "\nJustification:" or the leading line's "\nAction:" as spill —
# the task requires exactly those lines. (A second "\nAction:" is still a useful spill cue.)
_SPILL_MARKERS = (
    "\nHuman:",
    "\nAssistant:",
    "\nUser:",
    "\nSystem:",
    "\nObservation:",
    "\nThought:",
    "\nAction:",
)


def coerce_completion_text(completion: object, tokenizer: object = None) -> str:
    """
    Normalize TRL / Unsloth completion payloads to a single string.

    In conversational mode, TRL often passes each completion as a list of chat
    messages (e.g. one assistant message), not a raw string — that must be handled
    before regex-based reward code runs.
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return coerce_completion_text([completion], tokenizer=tokenizer)

    try:
        import torch

        if isinstance(completion, torch.Tensor):
            completion = completion.detach().cpu().tolist()
    except Exception:
        pass

    if hasattr(completion, "tolist") and not isinstance(completion, (str, bytes, dict, list, tuple)):
        try:
            completion = completion.tolist()
        except Exception:
            pass

    if isinstance(completion, (list, tuple)):
        if not completion:
            return ""
        # Conversational: list[dict] — prefer last assistant/model message content
        if isinstance(completion[0], dict):
            for msg in reversed(completion):
                role = str(msg.get("role", "")).strip().lower()
                if role in ("assistant", "model"):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        pieces: list[str] = []
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                pieces.append(part["text"])
                            elif isinstance(part, str):
                                pieces.append(part)
                        if pieces:
                            return "\n".join(pieces)
                    return coerce_completion_text(content, tokenizer=tokenizer)
            parts: list[str] = []
            for msg in completion:
                c = msg.get("content", "")
                if isinstance(c, str) and c.strip():
                    parts.append(c)
            return "\n".join(parts) if parts else ""

        if len(completion) == 1 and not isinstance(completion[0], int):
            return coerce_completion_text(completion[0], tokenizer=tokenizer)

        if all(isinstance(x, int) for x in completion):
            if tokenizer is not None and hasattr(tokenizer, "decode"):
                return tokenizer.decode(list(completion), skip_special_tokens=True)
            return ""

        if all(isinstance(x, str) for x in completion):
            avg = sum(len(x) for x in completion) / len(completion)
            return "\n".join(completion) if avg > 1.5 else "".join(completion)

    return str(completion)


def strip_generative_spill(text: object, tokenizer: object = None) -> str:
    """
    Models often keep "helpfully" continuing as chat after the 2-line answer.
    For env scoring, keep only the prefix before obvious multi-turn spill markers.
    """
    text = coerce_completion_text(text, tokenizer=tokenizer)
    if not text:
        return text
    lower = text.lower()
    cut = len(text)
    for marker in _SPILL_MARKERS:
        idx = lower.find(marker.lower())
        if idx != -1 and idx < cut:
            cut = idx
    trimmed = text[:cut].strip()

    # If we cut inside the justification line, try to keep full first two lines only.
    lines = [ln.rstrip() for ln in trimmed.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    if len(lines) >= 2:
        return f"{lines[0].strip()}\n{lines[1].strip()}".strip()
    return trimmed


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


def extract_justification_phrase(
    text: object,
    tokenizer: object = None,
) -> str:
    """
    Text after the first 'Justification:' line, from a stripped completion.
    The environment's parser uses this (tool routing, etc.), so it must be the
    model's actual line — not a constant placeholder.
    """
    s = strip_generative_spill(text, tokenizer=tokenizer)
    if not s:
        return ""
    lines = [ln.rstrip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    if len(lines) >= 2:
        m2 = re.match(
            r"^\s*Justification\s*:\s*(.*)$", lines[1], re.IGNORECASE
        )
        if m2:
            return m2.group(1).strip()
    m = re.search(r"(?im)^\s*Justification\s*:\s*(.+)$", s, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""


def compute_format_reward(text: object, tokenizer: object = None) -> float:
    """
    Format shaping in roughly [-3, +2] for GRPO. Kept in sync with training scripts
    (role spill, duplicate lines, 2-line structure) to avoid a degenerate mode where
    every sample shares the same score and group-relative advantage is zero.
    """
    score = 0.0
    text = strip_generative_spill(text, tokenizer=tokenizer)
    if re.search(r"(?im)\b(human|assistant|user|system)\s*:", str(text or "")):
        score -= 1.0
    if len(re.findall(r"(?im)^\s*action\s*:", str(text or ""))) > 1:
        score -= 0.8
    if len(re.findall(r"(?im)^\s*justification\s*:", str(text or ""))) > 1:
        score -= 0.8

    lines = [ln.rstrip() for ln in str(text or "").splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]

    if len(lines) == 2:
        score += 0.6
    elif len(lines) == 1:
        score -= 0.3
    else:
        score -= 1.0

    if len(lines) >= 1 and re.match(r"^\s*Action\s*:\s*\S", lines[0], re.IGNORECASE):
        score += 0.4
    else:
        score -= 0.8

    raw_action = extract_raw_action_phrase(str(text or ""))
    if raw_action and not action_line_is_snake_enum(raw_action):
        score -= 0.9

    if len(lines) >= 2:
        if re.match(r"^\s*Justification\s*:\s*\S", lines[1], re.IGNORECASE):
            score += 0.4
        else:
            score -= 0.45

    if len(lines) >= 2:
        just_line = lines[1]
        m = re.match(
            r"^\s*Justification\s*:\s*(.+)\s*$", just_line, re.IGNORECASE
        )
        if m:
            body = m.group(1).strip()
            if "\n" in body:
                score -= 1.0
            wc = len(body.split())
            if wc > 28:
                score -= min(2.0, 0.08 * (wc - 28))
            if len(body) > 220:
                score -= min(2.0, 0.01 * (len(body) - 220))
            sentences = re.split(r"(?<=[.!?])\s+", body)
            sentences = [s for s in sentences if s.strip()]
            if len(sentences) > 1:
                score -= 0.7

    if re.search(
        r"\b(however|therefore|moreover|additionally)\b", str(text or ""), re.IGNORECASE
    ):
        score -= 0.35

    return float(max(-3.0, min(2.0, score)))
