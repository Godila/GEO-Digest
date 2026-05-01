# Scrapling Integration Architecture — Minimal Change Design

## Overview

Replace all `urllib.request` HTTP calls in Category A (PDF downloads, Unpaywall, Crossref)
with Scrapling-backed fetchers that provide TLS impersonation, anti-bot bypass, retry with
backoff, and cookie persistence.

**Files changed: 4** (1 new + 3 modified)
**New lines: ~160** | **Deleted lines: ~50** | **Net delta: ~110 lines**

---

## File 1: NEW — `engine/http_client.py` (~130 lines)

Thin adapter wrapping Scrapling. Single module, no class hierarchy.

### Public API (2 functions)

```python
# engine/http_client.py

"""
HTTP client adapter wrapping Scrapling Fetcher + StealthyFetcher.
Replaces urllib.request for all Category A HTTP calls (PDF, Unpaywall, Crossref).

Design decisions:
  - Fetcher (curl_cffi): lightweight, created per call
  - StealthyFetcher (Patchright+Chromium): singleton, lazy-started, reused
  - Module-level functions, NOT a class — keeps call sites minimal
  - Cookies persist on the StealthyFetcher browser context between calls
"""

import json
import sys
import time
from typing import Optional

# ── Error classification ────────────────────────────────────

SKIP_CODES       = frozenset({404, 410})         # Permanent — don't retry
STEALTH_CODES    = frozenset({403, 401})          # Retry with StealthyFetcher
BACKOFF_CODES    = frozenset({429, 502, 503, 504}) # Retry with exponential backoff

# ── StealthyFetcher singleton ───────────────────────────────
# Expensive (~2s startup, ~200MB RAM for Chromium). Created once, reused forever.

_stealth_instance = None

def _get_stealth():
    """Lazy-initialize StealthyFetcher singleton."""
    global _stealth_instance
    if _stealth_instance is None:
        from scrapling import StealthyFetcher
        print("  [HTTP] Starting StealthyFetcher (Patchright+Chromium)...", file=sys.stderr)
        _stealth_instance = StealthyFetcher(auto_install=True)
    return _stealth_instance

def shutdown_stealth():
    """Graceful shutdown. Call from worker shutdown hook."""
    global _stealth_instance
    if _stealth_instance is not None:
        # StealthyFetcher closes browser on GC; explicitly release
        _stealth_instance = None

# ── Core fetch with retry ───────────────────────────────────

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
    Fetch URL with Fetcher → stealth fallback → exponential backoff retry.

    Returns (status_code, body_bytes).
    Raises only on network-level failure after all retries.
    """
    from scrapling import Fetcher

    merged = {
        "User-Agent": "GEO-Digest/1.0 (mailto:geo-digest@research.bot)",
        "Accept": accept,
        **(headers or {}),
    }

    last_exc = None
    used_stealth = False

    for attempt in range(max_retries):
        try:
            # ── Phase 1: Fast Fetcher (curl_cffi with TLS impersonation) ──
            resp = Fetcher(auto_install=False).get(url, headers=merged, timeout=timeout)
            status = resp.status_code

            # Permanent skip
            if status in SKIP_CODES:
                return status, resp.content

            # Anti-bot block → escalate to StealthyFetcher
            if status in STEALTH_CODES and stealth_on_block and not used_stealth:
                print(f"  [HTTP] {status} → stealth retry: {url[:60]}", file=sys.stderr)
                used_stealth = True
                stealth = _get_stealth()
                resp = stealth.fetch(url, headless=True, timeout=timeout * 2)
                status = resp.status_code
                if status not in SKIP_CODES:
                    return status, resp.content
                return status, resp.content

            # Rate limit / transient → backoff and retry
            if status in BACKOFF_CODES:
                wait = min(2 ** attempt * 2, 30)
                print(f"  [HTTP] {status} → backoff {wait}s (attempt {attempt+1}/{max_retries}): {url[:60]}", file=sys.stderr)
                time.sleep(wait)
                continue

            # Success or other code
            return status, resp.content

        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    if last_exc:
        raise last_exc
    raise RuntimeError(f"All {max_retries} retries exhausted for {url[:80]}")

# ── Public API ──────────────────────────────────────────────

def fetch_json(url: str, *, headers: dict | None = None, timeout: int = 15) -> dict:
    """
    GET url → parse JSON. Returns {} on HTTP error or parse failure.
    Drop-in replacement for the Unpaywall/Crossref urllib blocks.
    """
    try:
        status, body = _fetch_with_retry(url, headers=headers, timeout=timeout)
        if status >= 400:
            return {}
        return json.loads(body)
    except Exception as e:
        print(f"  [HTTP] fetch_json failed {url[:60]}: {e}", file=sys.stderr)
        return {}


def fetch_bytes(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 30,
    validate_pdf: bool = True,
) -> Optional[bytes]:
    """
    GET url → raw bytes. Returns None on failure or if not a valid PDF.
    Drop-in replacement for _try_download_pdf()'s urllib block.
    """
    pdf_headers = {
        "User-Agent": "Mozilla/5.0 (GEO-Digest Research Agent; contact@geo-digest.org)",
        "Accept": "application/pdf,*/*",
        **(headers or {}),
    }

    try:
        status, body = _fetch_with_retry(
            url, headers=pdf_headers, timeout=timeout,
            accept="application/pdf,*/*",
        )
        if status >= 400 or not body:
            return None

        if validate_pdf and (len(body) < 1000 or not body.startswith(b"%PDF")):
            return None

        return body

    except Exception as e:
        print(f"  [HTTP] fetch_bytes failed {url[:60]}: {e}", file=sys.stderr)
        return None
```

---

## File 2: MODIFY — `engine/agents/tools.py`

### Change 1: Remove urllib imports, add http_client import

```python
# BEFORE (line 19-20):
import urllib.error
import urllib.request

# AFTER:
# (remove both lines — no longer needed)
```

No new import needed at top level. Import `engine.http_client` inline where used
(to match the existing pattern in the codebase where heavy imports are local).

### Change 2: `enrich_oa_url()` — lines 326–361

Replace the 12-line urllib block (lines 340–361) with 4-line http_client call:

```python
# BEFORE (lines 340-349):
        try:
            import urllib.parse, json
            email = os.environ.get("UNPAYWALL_EMAIL", "geo-digest@research.bot")
            url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"

            req = urllib.request.Request(url, headers={
                "User-Agent": "GEO-Digest/1.0 (mailto:geo-digest@research.bot)"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

# AFTER:
        try:
            import urllib.parse
            from engine.http_client import fetch_json

            email = os.environ.get("UNPAYWALL_EMAIL", "geo-digest@research.bot")
            url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
            data = fetch_json(url, timeout=10)
            if not data:
                return
```

Everything after line 350 (parsing data for pdf_url) stays **identical**.

### Change 3: `_try_download_pdf()` — lines 421–449

Replace the 10-line urllib block (lines 432–448) with 5-line http_client call:

```python
# BEFORE (lines 432-448):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (GEO-Digest Research Agent; contact@geo-digest.org)",
                "Accept": "application/pdf,*/*",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                
            # Validate it's actually a PDF
            if len(data) < 1000 or not data.startswith(b"%PDF"):
                return None
            
            cache_path.write_bytes(data)
            return cache_path
            
        except Exception as e:
            print(f"  [PDF] Failed {url[:60]}...: {e}", file=sys.stderr)
            return None

# AFTER:
        try:
            from engine.http_client import fetch_bytes

            data = fetch_bytes(url, timeout=timeout)
            if data is None:
                return None

            cache_path.write_bytes(data)
            return cache_path

        except Exception as e:
            print(f"  [PDF] Failed {url[:60]}...: {e}", file=sys.stderr)
            return None
```

PDF validation is now handled inside `fetch_bytes(validate_pdf=True)` — no duplicate check needed.

---

## File 3: MODIFY — `engine/agents/reader.py`

### Change 1: `_enrich_pdf_url()` — lines 245–269

Replace the 10-line urllib block (lines 248–255) with http_client call:

```python
# BEFORE (lines 248-255):
        try:
            email = os.environ.get("UNPAYWALL_EMAIL", "geo-digest@research.bot")
            url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "GEO-Digest/1.0 (mailto:geo-digest@research.bot)"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

# AFTER:
        try:
            from engine.http_client import fetch_json

            email = os.environ.get("UNPAYWALL_EMAIL", "geo-digest@research.bot")
            url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
            data = fetch_json(url, timeout=10)
            if not data:
                return
```

The remaining logic (lines 257–269) stays **identical**.

### Change 2: `_resolve_doi_from_api()` — lines 271–330

Replace TWO urllib blocks:

**Block A — Unpaywall call (lines 282–298):**

```python
# BEFORE (lines 282-298):
        try:
            email = os.environ.get("UNPAYWALL_EMAIL", "geo-digest@research.bot")
            url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
            req = urllib.request.Request(url, headers={...})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            title = data.get("title", "")
            ...

# AFTER:
        try:
            from engine.http_client import fetch_json

            email = os.environ.get("UNPAYWALL_EMAIL", "geo-digest@research.bot")
            url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
            data = fetch_json(url, timeout=10)
            if data:
                title = data.get("title", "")
                year = data.get("year")
                pdf_url = (data.get("best_oa_location") or {}).get("url_for_pdf", "")
                if not pdf_url:
                    pdf_url = (data.get("first_oa_location") or {}).get("url_for_pdf", "")
                for a in (data.get("z_authors") or [])[:10]:
                    authors.append(a.get("given", "") + " " + a.get("family", ""))
```

**Block B — Crossref call (lines 301–316):**

```python
# BEFORE (lines 301-316):
        if not title or not abstract:
            try:
                url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
                req = urllib.request.Request(url, headers={...})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    cr = json.loads(resp.read()).get("message", {})
                ...

# AFTER:
        if not title or not abstract:
            try:
                url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
                cr = fetch_json(url, timeout=10)
                if cr:
                    cr = cr.get("message", {})
                if not cr:
                    cr = {}
                ...
```

The rest of the function (lines 309–330) stays **identical**.

---

## File 4: MODIFY — `worker/Dockerfile`

```dockerfile
# BEFORE:
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# ...

# Python deps
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart pyyaml \
    requests httpx \
    'markitdown[pdf]'

# AFTER:
FROM python:3.12-slim

WORKDIR /app

# System deps + Chromium for Patchright (StealthyFetcher)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget gnupg \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# ...

# Python deps
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart pyyaml \
    requests httpx \
    'markitdown[pdf]' \
    'scrapling[all]'

# Install Patchright Chromium browser
RUN python -c "from patchright.sync_api import sync_playwright; print('Patchright OK')" \
    || patchright install chromium
```

**Size impact:** ~+500MB (Chromium binary + shared libs)
**Memory:** StealthyFetcher Chromium process ~150-200MB when active

---

## File 5: MODIFY — `docker-compose.yml`

```yaml
# BEFORE (line 52):
          memory: 2G

# AFTER:
          memory: 4G
```

Single line change. Also bump reservation:

```yaml
# BEFORE (line 54):
          memory: 128M

# AFTER:
          memory: 512M
```

---

## Summary of Function Signatures

### New: `engine/http_client.py`

| Function | Signature | Purpose |
|----------|-----------|---------|
| `fetch_json` | `(url: str, *, headers: dict\|None=None, timeout: int=15) -> dict` | API calls (Unpaywall, Crossref) |
| `fetch_bytes` | `(url: str, *, headers: dict\|None=None, timeout: int=30, validate_pdf: bool=True) -> Optional[bytes]` | PDF downloads |
| `shutdown_stealth` | `() -> None` | Worker shutdown hook |

### Changed: `engine/agents/tools.py`

| Function | Change | Lines affected |
|----------|--------|---------------|
| `enrich_oa_url` | Replace urllib block with `fetch_json()` | 340–361 → 4 lines |
| `_try_download_pdf` | Replace urllib block with `fetch_bytes()` | 432–448 → 5 lines |

### Changed: `engine/agents/reader.py`

| Function | Change | Lines affected |
|----------|--------|---------------|
| `_enrich_pdf_url` | Replace urllib block with `fetch_json()` | 248–255 → 3 lines |
| `_resolve_doi_from_api` | Replace 2 urllib blocks with `fetch_json()` | 282–298, 301–316 → 6 lines |

### Changed: Infrastructure

| File | Change |
|------|--------|
| `worker/Dockerfile` | Add Chromium libs, `scrapling[all]`, `patchright install chromium` |
| `docker-compose.yml` | `memory: 2G` → `4G`, reservation `128M` → `512M` |

---

## StealthyFetcher Lifecycle

```
Worker starts
  └─ StealthyFetcher = None (lazy)

First 403/Cloudflare encountered
  └─ _get_stealth() called
      └─ StealthyFetcher(auto_install=True)
          └─ Patchright launches Chromium (~2s, ~200MB)
          └─ Browser stays alive in background
          └─ Cookies persist in browser context

Subsequent calls
  └─ Reuse same Chromium instance
  └─ Session cookies carry over (e.g., publisher "I accept" banners)

Worker shutdown / container stop
  └─ shutdown_stealth() — release browser
  └─ Or just let container kill clean it up
```

**Key design choice:** StealthyFetcher is ONLY activated on 403/block.
For 95% of requests (Unpaywall API, Crossref API, direct OA PDFs),
the lightweight Fetcher (curl_cffi) handles it without Chromium overhead.

---

## Error Classification Flow

```
HTTP Response
  ├─ 404/410     → return immediately (skip, no retry)
  ├─ 403/401     → first: retry with StealthyFetcher
  │                 still failing → backoff retry
  ├─ 429/502/503 → exponential backoff (2s, 4s, 8s... max 30s)
  ├─ 200-399     → return success
  └─ other 4xx   → return as-is (don't retry)
```

---

## What Does NOT Change

- `engine/llm/` — LLM API calls stay on their own HTTP client (httpx/openai SDK)
- `dashboard/app.py` — Proxy stays on FastAPI's built-in client
- `markitdown` PDF→text extraction — unchanged
- `engine/schemas.py` — no changes
- `engine/storage/` — no changes
- `engine/agents/base.py` — no changes
- All other agent files (scout, writer, etc.) — no changes
