#!/usr/bin/env python3
"""
One-time migration: add _id, _md_path, _enriched_at to existing articles.jsonl.

Run inside worker container:
  docker exec geo-digest-worker python3 /app/scripts/migrate_add_ids.py
Or locally:
  python3 scripts/migrate_add_ids.py
"""

import json
import re
import sys
from pathlib import Path

# Ensure scripts are importable
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sources.base import title_hash

DATA_DIR = Path("/app/data")
ARTICLES_DB = DATA_DIR / "articles.jsonl"
ARTICLES_MD_DIR = DATA_DIR / "articles"


def canonical_id(article: dict) -> str:
    """Generate canonical article ID: doi:... or hash:..."""
    doi = (article.get("doi") or "").strip()
    if doi:
        return f"doi:{doi.lower()}"
    h = title_hash(article.get("title", ""), str(article.get("year", "")))
    return f"hash:{h}"


def make_md_path(title: str) -> str:
    """Generate safe path for enrichment .md file."""
    safe = re.sub(r'[/\\:*?"<>|]', "_", (title or "untitled")[:80])
    return f"articles/{safe}.md"


def find_md_for_article(title: str) -> tuple[str, bool]:
    """Find existing .md file for an article. Returns (path, exists)."""
    # Try the standard naming convention
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (title or "untitled"))[:100]
    path = ARTICLES_MD_DIR / f"{safe}.md"
    if path.exists():
        return f"articles/{safe}.md", True

    # Try alternative naming (used by _make_md_path)
    safe2 = re.sub(r'[/\\:*?"<>|]', "_", (title or "untitled")[:80])
    path2 = ARTICLES_MD_DIR / f"{safe2}.md"
    if path2.exists():
        return f"articles/{safe2}.md", True

    return "", False


def migrate():
    if not ARTICLES_DB.exists():
        print("ERROR: articles.jsonl not found at", ARTICLES_DB)
        sys.exit(1)

    lines = ARTICLES_DB.read_text(encoding="utf-8").splitlines()

    updated = 0
    with_id = 0
    with_md = 0
    with_enriched = 0
    output_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        try:
            art = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"WARNING: Skipping invalid JSON on line {i+1}: {e}")
            output_lines.append(line)
            continue

        changed = False

        # Add _id
        if "_id" not in art:
            art["_id"] = canonical_id(art)
            updated += 1
        else:
            with_id += 1

        # Add _md_path
        if "_md_path" not in art:
            md_path, md_exists = find_md_for_article(art.get("title", ""))
            art["_md_path"] = md_path
            if md_exists:
                with_md += 1
            changed = True
        else:
            if art["_md_path"]:
                full = DATA_DIR / art["_md_path"]
                if full.exists():
                    with_md += 1

        # Add _enriched_at from .md file mtime
        if "_enriched_at" not in art and art.get("_md_path"):
            full = DATA_DIR / art["_md_path"]
            if full.exists():
                from datetime import datetime, timezone
                mtime = datetime.fromtimestamp(full.stat().st_mtime, tz=timezone.utc)
                art["_enriched_at"] = mtime.isoformat()
                with_enriched += 1
                changed = True

        output_lines.append(json.dumps(art, ensure_ascii=False))

    # Write back
    ARTICLES_DB.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    total = len(output_lines)
    print(f"Migration complete:")
    print(f"  Total articles:     {total}")
    print(f"  Updated (new IDs): {updated}")
    print(f"  Already had _id:   {with_id}")
    print(f"  With .md file:     {with_md}")
    print(f"  With _enriched_at: {with_enriched}")

    # Verify no duplicates
    ids = set()
    dup_count = 0
    for line in output_lines:
        line = line.strip()
        if not line:
            continue
        art = json.loads(line)
        aid = art.get("_id", "")
        if aid in ids:
            dup_count += 1
            print(f"  DUPLICATE ID: {aid} — {art.get('title', '?')[:50]}")
        ids.add(aid)

    if dup_count == 0:
        print(f"  Duplicate check:    OK ({len(ids)} unique IDs)")
    else:
        print(f"  Duplicate check:    {dup_count} duplicates FOUND!")


if __name__ == "__main__":
    migrate()
