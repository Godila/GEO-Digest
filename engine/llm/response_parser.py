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

    # Strategy 4: find multiple JSON objects (not wrapped in array)
    obj_matches = re.findall(r'\{[^{}]*(?:"title"\s*:\s*"[^"]+")[^{}]*\}', text, re.DOTALL)
    if obj_matches and len(obj_matches) >= 1:
        proposals = []
        for om in obj_matches[:MAX_PROPOSALS_FROM_TEXT]:
            try:
                p = json.loads(om)
                if isinstance(p, dict) and p.get("title"):
                    proposals.append(_validate_proposal(p))
            except (json.JSONDecodeError, ValueError):
                continue
        if proposals:
            logger.info("Extracted %d proposals from loose JSON objects", len(proposals))
            return proposals

    # Strategy 5: extract from structured prose (fallback when LLM ignores JSON format)
    proposals = _extract_proposals_from_prose(text)
    if proposals:
        logger.info("Extracted %d proposals from prose text (JSON parse failed)", len(proposals))
        return proposals

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


MAX_PROPOSALS_FROM_TEXT = 3  # Max proposals to extract from loose prose


def _extract_proposals_from_prose(text: str) -> list[dict]:
    """
    Fallback parser: extract proposals from free-form Russian text.

    When LLM ignores JSON format and outputs structured prose,
    try to extract proposal-like structures using heuristics.

    Looks for patterns like:
      - Numbered/bulleted sections with titles
      - "Вариант N" / "Предложение N" / "Концепт N"
      - DOI lists near thesis-like text
    """
    if not text or len(text) < 50:
        return []

    proposals = []

    # Pattern 1: numbered sections like "1. Заголовок" or "Вариант 1:"
    # Split by common delimiters
    section_patterns = [
        r'(?:^|\n)\s*(?:\d+\.|Вариант\s+\d+|Предложение\s+\d+|Концепт\s+\d+|Option\s+\d+)\s*[:.\s]*\n?\s*\*{0,2}(.+?)(?:\*{0,2})',
        r'(?:^|\n)\s*(?:###?\s*|\*\*)(.{10,100}?)(?:\*\*|\n)',
    ]

    # Try to find title-like lines (bold, headers, or numbered)
    lines = text.split('\n')
    current_proposal = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect new proposal start
        is_header = bool(re.match(r'^(?:\d+[\.\)]\s*|Вариант|Предложение|Концепт|Option|#)', stripped, re.IGNORECASE))
        is_bold_header = stripped.startswith('**') and stripped.endswith('**') and len(stripped) > 5
        looks_like_title = (
            len(stripped) > 15 and len(stripped) < 200
            and not stripped.startswith('|') and not stripped.startswith('-')
            and not stripped.startswith('```') and not stripped.startswith('"')
            and ('статья' in stripped.lower() or 'анализ' in stripped.lower()
                 or 'исследование' in stripped.lower() or 'review' in stripped.lower()
                 or 'emissions' in stripped.lower() or 'permafrost' in stripped.lower()
                 or 'модель' in stripped.lower() or 'метод' in stripped.lower())
            and current_proposal is None  # only start new if previous was saved
        )

        if is_header or is_bold_header:
            # Save previous proposal if exists
            if current_proposal and current_proposal.get("title"):
                proposals.append(current_proposal)

            title = stripped.lstrip('#* \t').rstrip('*:')
            current_proposal = {"title": title, "thesis": "", "key_references": [], "confidence": 0.6}

        elif current_proposal and stripped and not is_header:
            # Accumulate content into thesis
            # Check for DOIs in this line
            doi_matches = re.findall(r'10\.\d{4,}/[^\s,\]\)}]+', stripped)
            for doi in doi_matches:
                doi_clean = doi.rstrip('`').rstrip(',').rstrip('.')
                ref_key = f"DOI:{doi_clean}"
                if ref_key not in current_proposal["key_references"]:
                    current_proposal["key_references"].append(ref_key)

            # Add non-empty, non-DOI-only lines to thesis
            line_for_thesis = re.sub(r'10\.\d{4,}/[^\s,\]\)}]+', '[DOI]', stripped).strip()
            if line_for_thesis and len(line_for_thesis) > 10 and not line_for_thesis.startswith(('-', '|', '•', '—')):
                if current_proposal["thesis"]:
                    current_proposal["thesis"] += " " + line_for_thesis
                else:
                    current_proposal["thesis"] = line_for_thesis

    # Don't forget last proposal
    if current_proposal and current_proposal.get("title"):
        proposals.append(current_proposal)

    # Validate and clean up
    validated = []
    for p in proposals[:3]:  # max 3 from prose
        p = _validate_proposal(p)
        # Only keep if we got a reasonable thesis (not just garbage)
        if len(p.get("thesis", "")) > 20:
            validated.append(p)

    return validated


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
