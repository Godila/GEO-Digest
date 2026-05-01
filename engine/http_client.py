"""
HTTP client adapter wrapping Scrapling Fetcher + StealthyFetcher.

Replaces urllib.request for Category A HTTP calls (PDF downloads, Unpaywall, Crossref).
Provides automatic retry with backoff, anti-bot bypass, and cookie persistence.

Usage:
    from engine.http_client import fetch_json, fetch_bytes

    # API call (Unpaywall / Crossref)
    data = fetch_json("https://api.unpaywall.org/v2/...", timeout=10)

    # PDF download
    pdf_bytes = fetch_bytes("https://publisher.com/paper.pdf", timeout=30)

Lifecycle:
    - Fetcher (curl_cffi): lightweight, created per call, TLS impersonation
    - StealthyFetcher (Patchright+Chromium): singleton, lazy-started on first 403,
      reused across all subsequent calls, cookies persist in browser context
"""

from __future__ import annotations

import json
import sys
import time
from typing import Optional

# ── Error classification ────────────────────────────────────────────────────
# Status codes that determine retry strategy

SKIP_CODES = frozenset({404, 410})               # Permanent — don't retry
STEALTH_CODES = frozenset({403, 401})             # Anti-bot → retry with StealthyFetcher
BACKOFF_CODES = frozenset({429, 502, 503, 504})   # Transient → exponential backoff


# ── StealthyFetcher singleton ───────────────────────────────────────────────
# Patchright Chromium is expensive (~2s startup, ~200MB RAM).
# Created once on first 403, then reused for the lifetime of the worker process.
# Cookies persist in the browser context between calls (e.g., publisher "accept" banners).

_stealth_instance: object | None = None


def _get_stealth():
    """Lazy-initialize StealthyFetcher singleton. Thread-safe via GIL."""
    global _stealth_instance
    if _stealth_instance is None:
        from scrapling import StealthyFetcher  # type: ignore[import-untyped]
        print("  [HTTP] Starting StealthyFetcher (Patchright + Chromium)...", file=sys.stderr)
        _stealth_instance = StealthyFetcher(auto_install=True)
    return _stealth_instance


def shutdown_stealth() -> None:
    """Release StealthyFetcher browser. Call from worker shutdown hook or atexit."""
    global _stealth_instance
    if _stealth_instance is not None:
        print("  [HTTP] Shutting down StealthyFetcher", file=sys.stderr)
        _stealth_instance = None


# ── Core fetch with retry ───────────────────────────────────────────────────

def _fetch_with_retry(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 15,
    max_retries: int = 3,
    accept: str = "*/*",
    stealth_on_block: bool = True,
) -> tuple[int, bytes]:
    """
    Fetch URL with automatic retry, stealth escalation, and exponential backoff.

    Strategy per status code:
        404/410   → return immediately (permanent failure, skip)
        403/401   → escalate to StealthyFetcher, then backoff retry
        429/502-4 → exponential backoff (2s, 4s, 8s ... max 30s)
        200-399   → return success
        other     → retry with backoff

    Returns:
        (status_code, body_bytes)

    Raises:
        Last exception on network failure after all retries exhausted.
    """
    from scrapling import Fetcher  # type: ignore[import-untyped]

    merged_headers = {
        "User-Agent": "GEO-Digest/1.0 (mailto:geo-digest@research.bot)",
        "Accept": accept,
        **(headers or {}),
    }

    last_exc: Exception | None = None
    used_stealth = False

    for attempt in range(max_retries):
        try:
            # Phase 1: Fast Fetcher (curl_cffi with TLS impersonation)
            fetcher = Fetcher(auto_install=False)
            resp = fetcher.get(url, headers=merged_headers, timeout=timeout)
            status = resp.status_code

            # Permanent skip — don't waste retries
            if status in SKIP_CODES:
                return status, resp.content

            # Anti-bot block → escalate to StealthyFetcher (once)
            if status in STEALTH_CODES and stealth_on_block and not used_stealth:
                print(
                    f"  [HTTP] {status} → stealth retry: {url[:70]}",
                    file=sys.stderr,
                )
                used_stealth = True
                stealth = _get_stealth()
                if stealth is None:
                    raise RuntimeError("StealthyFetcher not available")
                resp = stealth.fetch(url, headless=True, timeout=timeout * 2)  # type: ignore[union-attr]
                status = resp.status_code
                # Return whatever stealth gave us (could be success or another error)
                return status, resp.content

            # Rate limit / server error → backoff and retry
            if status in BACKOFF_CODES:
                wait = min(2 ** attempt * 2, 30)
                print(
                    f"  [HTTP] {status} → backoff {wait}s "
                    f"(attempt {attempt + 1}/{max_retries}): {url[:70]}",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue

            # Success (2xx/3xx) or unclassified status → return as-is
            return status, resp.content

        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    # All retries exhausted
    if last_exc:
        raise last_exc
    raise RuntimeError(f"All {max_retries} retries exhausted for {url[:80]}")


# ── Public API ──────────────────────────────────────────────────────────────

def fetch_json(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 15,
) -> dict:
    """
    GET url → parse JSON body.

    Returns {} on HTTP error (4xx/5xx) or JSON parse failure.
    Drop-in replacement for the urllib.request + json.loads() pattern
    used in Unpaywall and Crossref API calls.

    Example (replaces 8-line urllib block):
        data = fetch_json(url, timeout=10)
        if not data:
            return
    """
    try:
        status, body = _fetch_with_retry(url, headers=headers, timeout=timeout)
        if status >= 400:
            return {}
        return json.loads(body)
    except Exception as e:
        print(f"  [HTTP] fetch_json failed {url[:70]}: {e}", file=sys.stderr)
        return {}


def fetch_bytes(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 30,
    validate_pdf: bool = True,
) -> Optional[bytes]:
    """
    GET url → raw bytes.

    Returns None on failure or if response is not a valid PDF.
    Drop-in replacement for the urllib.request binary download in _try_download_pdf().

    PDF validation: checks body starts with b"%PDF" and is > 1000 bytes.
    Set validate_pdf=False for non-PDF binary downloads.

    Example (replaces 10-line urllib block):
        data = fetch_bytes(url, timeout=timeout)
        if data is None:
            return None
        cache_path.write_bytes(data)
    """
    pdf_headers = {
        "User-Agent": "Mozilla/5.0 (GEO-Digest Research Agent; contact@geo-digest.org)",
        "Accept": "application/pdf,*/*",
        **(headers or {}),
    }

    try:
        status, body = _fetch_with_retry(
            url,
            headers=pdf_headers,
            timeout=timeout,
            accept="application/pdf,*/*",
        )

        if status >= 400 or not body:
            return None

        if validate_pdf and (len(body) < 1000 or not body.startswith(b"%PDF")):
            return None

        return body

    except Exception as e:
        print(f"  [HTTP] fetch_bytes failed {url[:70]}: {e}", file=sys.stderr)
        return None
