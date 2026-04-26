
"""Utility functions for GEO-Digest Engine."""
from __future__ import annotations
import hashlib


def title_hash(title: str, year: str = "") -> str:
    text = f"{title}|{year}".strip("|")
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m{s:.0f}s"
