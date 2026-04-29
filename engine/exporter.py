"""GEO-Digest Article Exporter.

Converts final article (Markdown + LaTeX) to DOCX and PDF via Pandoc.

Features:
  - LaTeX formulas → OMML (editable in Word/LibreOffice)
  - Native tables
  - Russian/Cyrillic text
  - Styled document (title page, headers, margins)
  - Optional PDF via LibreOffice headless (if available)

Usage:
  from engine.exporter import export_article
  path = export_article(job, fmt="docx", output_dir="/tmp")
"""

import logging
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("engine.exporter")

# ---------------------------------------------------------------------------
# Pandoc binary — pypandoc_binary bundles it
# ---------------------------------------------------------------------------
try:
    import pypandoc
    _PANDOC_PATH = pypandoc.get_pandoc_path()
    _PANDOC_VER = pypandoc.get_pandoc_version()
    _HAS_PANDOC = True
except Exception:
    _HAS_PANDOC = False
    _PANDOC_PATH = ""
    _PANDOC_VER = ""

logger.info(f"Exporter init: pandoc={'yes' if _HAS_PANDOC else 'NO'} ({_PANDOC_VER})")


# ---------------------------------------------------------------------------
# Markdown preprocessing
# ---------------------------------------------------------------------------

def _ensure_latex_dollars(md: str) -> str:
    r"""Convert \[...\] and \(...\) LaTeX delimiters to $$...$$ and $...$.

    Pandoc only recognises $...$$ delimiters from markdown+tex_math_dollars.
    """
    # Display math: \[...\] → $$...$$
    md = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', md, flags=re.DOTALL)
    # Inline math: \(...\) → $...$
    md = re.sub(r'\((.+?)\\\)', r'$\1$', md, flags=re.DOTALL)
    return md


def _inject_metadata(md: str, title: str, language: str = "ru") -> str:
    """Add YAML front matter for Pandoc (title, lang)."""
    lang_tag = "ru-RU" if language == "ru" else "en-US"
    front_matter = f"""---
title: "{title}"
lang: {lang_tag}
---

"""
    # Avoid double front-matter
    if md.startswith("---"):
        return md
    return front_matter + md


def _prepare_markdown(article_text: str, title: str, language: str) -> str:
    """Full preprocessing pipeline for Markdown source."""
    md = _ensure_latex_dollars(article_text)
    md = _inject_metadata(md, title, language)
    return md


# ---------------------------------------------------------------------------
# DOCX reference doc (styles template)
# ---------------------------------------------------------------------------

def _get_reference_docx() -> Optional[str]:
    """Return path to custom DOCX reference doc, or None."""
    ref = Path(__file__).parent / "assets" / "reference.docx"
    if ref.exists():
        return str(ref)
    return None


# ---------------------------------------------------------------------------
# Core export functions
# ---------------------------------------------------------------------------

def _to_docx(md: str, output_path: str) -> str:
    """Convert Markdown + LaTeX to DOCX via Pandoc.

    LaTeX formulas become OMML (Office Math Markup Language) —
    fully editable in Microsoft Word and LibreOffice Writer.
    """
    if not _HAS_PANDOC:
        raise RuntimeError(
            "Pandoc not available. Install: pip install pypandoc_binary"
        )

    assert pypandoc is not None  # guaranteed by _HAS_PANDOC

    extra_args = [
        "--from", "markdown+tex_math_dollars+pipe_tables+raw_tex",
        "--to", "docx",
        "--standalone",
        "--wrap=none",
    ]

    ref_doc = _get_reference_docx()
    if ref_doc:
        extra_args.extend(["--reference-doc", ref_doc])

    pypandoc.convert_text(
        md,
        "docx",
        format="markdown+tex_math_dollars+pipe_tables+raw_tex",
        outputfile=output_path,
        extra_args=extra_args,
    )

    size = os.path.getsize(output_path)
    logger.info(f"DOCX exported: {output_path} ({size:,} bytes)")
    return output_path


def _to_pdf_via_libreoffice(docx_path: str, output_dir: str) -> str:
    """Convert DOCX → PDF using LibreOffice headless (soffice).

    Falls back gracefully if LibreOffice is not installed.
    """
    import subprocess
    lo_bin = os.environ.get("SOFFICE_BIN", "soffice")
    try:
        result = subprocess.run(
            [
                lo_bin,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", output_dir,
                docx_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning(f"LibreOffice PDF conversion failed: {result.stderr}")
            raise RuntimeError(f"soffice failed: {result.stderr}")

        pdf_path = docx_path.replace(".docx", ".pdf")
        if os.path.exists(pdf_path):
            size = os.path.getsize(pdf_path)
            logger.info(f"PDF exported: {pdf_path} ({size:,} bytes)")
            return pdf_path
        raise RuntimeError("PDF file not found after soffice conversion")

    except FileNotFoundError:
        raise RuntimeError(
            f"LibreOffice ({lo_bin}) not found. "
            "Install: apt-get install libreoffice-writer"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_article(
    job_or_article: Union[object, dict],
    fmt: str = "docx",
    output_dir: Optional[str] = None,
) -> str:
    """Export a pipeline job's final article to DOCX or PDF.

    Args:
        job_or_article: PipelineJob object or dict with 'final_article' field.
        fmt: Export format — "docx" or "pdf".
        output_dir: Directory for output file. Default: temp directory.

    Returns:
        Absolute path to the exported file.

    Raises:
        RuntimeError: If article is missing or conversion fails.
    """
    # ── Extract article text ──
    if isinstance(job_or_article, dict):
        fa = job_or_article.get("final_article") or job_or_article
        job_id = job_or_article.get("job_id", "unknown")
    else:
        fa = getattr(job_or_article, "final_article", None)
        job_id = getattr(job_or_article, "job_id", "unknown")

    if fa is None:
        raise RuntimeError("No final article found in job")

    # final_article may be dict or WrittenArticle dataclass
    if isinstance(fa, dict):
        text = fa.get("text") or fa.get("content") or fa.get("body", "")
        title = fa.get("title") or f"Article {job_id}"
        language = fa.get("language", "ru")
    else:
        text = getattr(fa, "text", "") or ""
        title = getattr(fa, "title", "") or f"Article {job_id}"
        language = getattr(fa, "language", "ru")

    if not text.strip():
        raise RuntimeError("Article text is empty — nothing to export")

    # ── Prepare output dir ──
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="geo_export_")
    os.makedirs(output_dir, exist_ok=True)

    # ── Prepare Markdown ──
    md = _prepare_markdown(text, title, language)

    # ── Export ──
    # Sanitize job_id for filename
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(job_id))[:40]
    base_name = f"article_{safe_id}"

    if fmt == "docx":
        out_path = os.path.join(output_dir, f"{base_name}.docx")
        _to_docx(md, out_path)
        return os.path.abspath(out_path)

    elif fmt == "pdf":
        # Strategy: MD → DOCX (with OMML formulas) → PDF via LibreOffice
        docx_path = os.path.join(output_dir, f"{base_name}_tmp.docx")
        _to_docx(md, docx_path)
        pdf_path = _to_pdf_via_libreoffice(docx_path, output_dir)
        # Clean up temp DOCX
        try:
            os.unlink(docx_path)
        except OSError:
            pass
        return os.path.abspath(pdf_path)

    elif fmt == "md":
        out_path = os.path.join(output_dir, f"{base_name}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return os.path.abspath(out_path)

    else:
        raise ValueError(f"Unsupported export format: {fmt}. Use: docx, pdf, md")


def get_available_formats() -> list[str]:
    """Return list of export formats available in this environment."""
    formats = ["md"]
    if _HAS_PANDOC:
        formats.append("docx")
        # Check LibreOffice for PDF
        import shutil
        if shutil.which("soffice") or os.environ.get("SOFFICE_BIN"):
            formats.append("pdf")
    return formats
