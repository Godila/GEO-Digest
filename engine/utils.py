"""
Shared utilities for the engine.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime


def title_hash(title: str, year: str = "") -> str:
    """
    Canonical SHA-256 hash of article title + year.
    
    Used for deduplication when DOI is not available.
    Single source of truth — same as scripts/sources/base.py
    """
    text = f"{title}|{year}".strip().lower()
    text = re.sub(r"\s+", " ", text)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def generate_id(prefix: str = "") -> str:
    """Generate unique ID with optional prefix."""
    uid = uuid.uuid4().hex[:12]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uid}" if prefix else f"{ts}_{uid}"


def truncate(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def safe_load_json(text: str, default=None):
    """Safely parse JSON string."""
    if not text or not text.strip():
        return default or {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default or {}


def extract_doi(text: str) -> str | None:
    """
    Extract DOI from various formats.
    
    Handles:
      - 10.3390/rs15071857
      - https://doi.org/10.3390/rs15071857
      - doi: 10.3390/rs15071857
    """
    if not text:
        return None
    
    # Already clean DOI
    if re.match(r"^10\.\d{4,9}/", text.strip()):
        return text.strip()
    
    # URL format
    m = re.search(r"doi\.org/(10\.\d{4,9}/[^\s]+)", text)
    if m:
        return m.group(1)
    
    # "doi: 10.xxxx" format
    m = re.search(r"doi:\s*(10\.\d{4,9}/[^\s]+)", text, re.I)
    if m:
        return m.group(1)
    
    return None


def word_count(text: str) -> int:
    """Count words in text (handles both Latin and Cyrillic)."""
    if not text:
        return 0
    # Split on whitespace and punctuation
    words = re.findall(r"\w+", text, re.UNICODE)
    return len(words)


def format_citation(article: dict) -> str:
    """
    Format a citation string from article data.
    
    Returns something like:
      "Smith et al. (2024). Title. Journal, Vol(Issue), pages."
    """
    authors = article.get("authors", "")
    year = article.get("year") or "????"
    title = article.get("title_ru") or article.get("title", "")
    journal = article.get("journal", "")
    
    # Shorten authors: "First Author et al." or just first name
    if authors:
        first = authors.split(";")[0].split(",")[0].strip()
        if ";" in authors or "," in authors:
            authors_str = f"{first} et al."
        else:
            authors_str = first
    else:
        authors_str = "Unknown"
    
    parts = [f"{authors_str} ({year}). {title}"]
    if journal:
        parts.append(journal)
    
    return ". ".join(parts)


def timestamp() -> str:
    """Current UTC ISO timestamp."""
    return datetime.utcnow().isoformat() + "Z"
