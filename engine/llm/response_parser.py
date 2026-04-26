"""Response Parser — extract structured data from LLM text responses.

Handles extraction of:
  - Article proposals (JSON arrays) from LLM output
  - Various JSON formats: clean, markdown-fenced, embedded in text

All parsers are defensive — they return empty/default on invalid input
rather than raising exceptions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_proposals_from_text(text: str) -> list[dict]:
    """
    Extract ArticleProposal[] from LLM's text response.

    Strategies tried in order:
      1. Direct JSON parse of entire text
      2. Extract JSON from markdown code fences (```json ... ```)
      3. Find first JSON array [...] in text
      4. Return [] if nothing found

    Args:
        text: Raw text from LLM (may contain markdown, explanations, etc.)

    Returns:
        List of validated proposal dicts. Never raises.
    """
    if not text or not isinstance(text, str):
        return []

    text = text.strip()

    # Strategy 1: direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            return [_validate_proposal(p) for p in data]
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from markdown fences
    fence_match = re.search(
        r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL
    )
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            if isinstance(data, list) and len(data) > 0:
                return [_validate_proposal(p) for p in data]
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first JSON array in text (non-greedy)
    array_match = re.search(r'\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]', text)
    if array_match:
        try:
            data = json.loads(array_match.group())
            if isinstance(data, list) and len(data) > 0:
                return [_validate_proposal(p) for p in data]
        except json.JSONDecodeError:
            pass

    # Fallback: nothing parseable found
    logger.debug("Could not extract proposals from text (length=%d)", len(text))
    return []


def _validate_proposal(p: dict) -> dict:
    """
    Validate and normalise a single proposal dict.

    Ensures required fields exist with sensible defaults.
    """
    if not isinstance(p, dict):
        return {"title": "[invalid proposal]", "thesis": "[invalid]"}

    required_defaults = {
        "title": "[не указан заголовок]",
        "thesis": "[не указан тезис]",
    }

    for field, default in required_defaults.items():
        value = p.get(field)
        if not value or not str(value).strip():
            p[field] = default

    # Set defaults for optional fields
    p.setdefault("confidence", 0.5)
    p.setdefault("sources_available", 0)
    p.setdefault("sources_needed", 5)
    p.setdefault("key_references", [])
    p.setdefault("target_audience", "general_public")
    p.setdefault("gap_filled", "")

    # Normalise confidence to [0, 1]
    try:
        conf = float(p["confidence"])
        p["confidence"] = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        p["confidence"] = 0.5

    # Ensure key_references is a list
    if not isinstance(p.get("key_references"), list):
        p["key_references"] = []

    return p


def parse_single_json_object(text: str) -> dict | None:
    """
    Try to extract a single JSON object from text.

    Useful for extracting analysis results, decisions, etc.
    Returns None on failure.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try markdown fence
    fence_match = re.search(
        r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL
    )
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Try finding {...}
    obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def extract_confidence_score(text: str) -> float | None:
    """
    Try to extract a confidence/quality score from text.

    Looks for patterns like "confidence: 0.85" or "оценка: 8/10".
    Returns None if no score found.
    """
    if not text:
        return None

    # Pattern: confidence: X.XX or confidence=X.XX
    match = re.search(
        r'(?:confidence|уверенность|score|оценка)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        text,
        re.IGNORECASE,
    )
    if match:
        try:
            val = float(match.group(1))
            # If score looks like it's out of 10, normalise to 0-1
            if val > 1.0:
                val = val / 10.0
            return max(0.0, min(1.0, val))
        except ValueError:
            pass

    return None
