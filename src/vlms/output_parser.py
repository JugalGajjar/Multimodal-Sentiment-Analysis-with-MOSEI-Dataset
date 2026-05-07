"""Parse VLM outputs into structured records.

VLMs are asked to return strict JSON, but in practice they leak prose,
markdown fences, trailing commentary, etc. This module:

  1. tries strict JSON parsing on the whole response
  2. falls back to a regex-extracted ``{...}`` block (greedy, then balanced)
  3. falls back to keyword search for a valid label in the raw text

so a high fraction of responses still produce a usable label even when the
model doesn't follow the JSON contract perfectly.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1)
        text = re.sub(r"```\s*$", "", text, count=1)
    return text.strip()


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction from arbitrary VLM output."""
    if not isinstance(text, str):
        return None
    candidate = _strip_code_fences(text)

    # 1. Direct parse
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. First single-level brace span
    m = re.search(r"\{[^{}]*\}", candidate, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Greedy outermost brace span (handles nested but tolerates trailing text)
    m = re.search(r"\{.*\}", candidate, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_label(text: str, valid_labels: tuple[str, ...] | list[str]) -> str | None:
    """Find a matching label in VLM output, JSON or freeform.

    Matches are case-insensitive and prefer JSON-extracted values; falls
    back to a keyword search in the raw text.
    """
    if not isinstance(text, str):
        return None

    valid_lower = {label.lower(): label for label in valid_labels}

    obj = extract_json(text)
    if obj and "label" in obj:
        candidate = str(obj["label"]).strip().lower()
        if candidate in valid_lower:
            return valid_lower[candidate]

    # Free-text fallback: prefer the *earliest* mention so "neutral" doesn't
    # win when the model later says "but the speaker sounds positive".
    text_lower = text.lower()
    earliest: tuple[int, str] | None = None
    for label in valid_labels:
        idx = text_lower.find(label.lower())
        if idx >= 0 and (earliest is None or idx < earliest[0]):
            earliest = (idx, label)
    return earliest[1] if earliest else None


def parse_confidence(text: str) -> float:
    """Pull a confidence value out of VLM output, defaulting to 0.0."""
    obj = extract_json(text)
    if obj and "confidence" in obj:
        try:
            v = float(obj["confidence"])
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def parse_response(text: str, valid_labels: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Convenience: parse label + confidence + explanation into one dict."""
    obj = extract_json(text) or {}
    label = parse_label(text, valid_labels)
    return {
        "label": label,
        "confidence": parse_confidence(text),
        "explanation": str(obj.get("explanation", "")).strip() if isinstance(obj.get("explanation"), (str, int, float)) else "",
        "parsed_ok": label is not None,
        "raw": text,
    }
