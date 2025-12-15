from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import fitz


DEFAULT_DPI = 300
DEFAULT_MAX_PAGES = 5
LOWRES_MAX_WIDTH = 1200
DEFAULT_TIMEOUT_S = 30
DEFAULT_USER_AGENT = "CodeArXivBot/0.1 (+https://arxiv.org)"
DEFAULT_PLACEHOLDER_TEXT = "Thumbnail unavailable"


def _safe_stem(value: str) -> str:
    stem = str(value or "").strip()
    stem = stem.replace("/", "_").replace("\\", "_")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = stem.strip("._-")
    return stem or "paper"


def _download_pdf(pdf_url: str, *, timeout_s: int = DEFAULT_TIMEOUT_S, user_agent: str = DEFAULT_USER_AGENT) -> bytes:
    req = urllib.request.Request(
        str(pdf_url),
        headers={
            "User-Agent": user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _ensure_rgb_no_alpha(pix: fitz.Pixmap) -> fitz.Pixmap:
    if pix.colorspace is not None and pix.colorspace != fitz.csRGB:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    return pix


def _render_pages_doc(doc: fitz.Document, *, max_pages: int = DEFAULT_MAX_PAGES, dpi: int = DEFAULT_DPI) -> List[fitz.Pixmap]:
    n = min(max_pages, doc.page_count)
    pixmaps: List[fitz.Pixmap] = []
    for i in range(n):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        pixmaps.append(_ensure_rgb_no_alpha(pix))
    return pixmaps


def _render_pages_from_path(
    pdf_path: Path, *, max_pages: int = DEFAULT_MAX_PAGES, dpi: int = DEFAULT_DPI
) -> List[fitz.Pixmap]:
    doc = fitz.open(str(pdf_path))
    try:
        return _render_pages_doc(doc, max_pages=max_pages, dpi=dpi)
    finally:
        doc.close()


def _render_pages_from_bytes(
    pdf_bytes: bytes, *, max_pages: int = DEFAULT_MAX_PAGES, dpi: int = DEFAULT_DPI
) -> List[fitz.Pixmap]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return _render_pages_doc(doc, max_pages=max_pages, dpi=dpi)
    finally:
        doc.close()


def _stitch_horiz(pixmaps: List[fitz.Pixmap], *, dpi: int = DEFAULT_DPI) -> fitz.Pixmap:
    if not pixmaps:
        raise ValueError("No pages rendered.")

    total_w = sum(p.width for p in pixmaps)
    max_h = max(p.height for p in pixmaps)

    out = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, total_w, max_h), 0)
    out.set_rect(out.irect, (255, 255, 255))

    x = 0
    for p in pixmaps:
        # NOTE: fitz.Pixmap(p) may add an alpha channel (default alpha=1),
        # which then breaks Pixmap.copy() when target alpha differs.
        p.set_origin(x, 0)
        out.copy(p, p.irect)
        x += p.width

    out.set_dpi(dpi, dpi)
    return out


def _save_png(pix: fitz.Pixmap, out_png: Path, *, dpi: Optional[int] = None) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if dpi is not None:
        pix.set_dpi(dpi, dpi)
    pix.save(str(out_png))


def _save_lowres_variant_pixmap(
    pix: fitz.Pixmap, lowres_png: Path, *, max_width: int = LOWRES_MAX_WIDTH
) -> Tuple[int, int]:
    lowres_png.parent.mkdir(parents=True, exist_ok=True)

    pix = _ensure_rgb_no_alpha(pix)

    if pix.width <= max_width:
        pix.save(str(lowres_png))
        return pix.width, pix.height

    scale = max_width / pix.width
    w = int(max_width)
    h = max(1, int(round(pix.height * scale)))
    low = fitz.Pixmap(pix, w, h)
    low.save(str(lowres_png))
    return w, h


def _placeholder_pixmap(
    *,
    width: int = 600,
    height: int = 200,
    text: str = DEFAULT_PLACEHOLDER_TEXT,
    background_rgb: Tuple[int, int, int] = (240, 240, 240),
    text_rgb: Tuple[int, int, int] = (100, 100, 100),
) -> fitz.Pixmap:
    doc = fitz.open()
    try:
        page = doc.new_page(width=width, height=height)
        rect = fitz.Rect(0, 0, width, height)
        fill = tuple(c / 255.0 for c in background_rgb)
        page.draw_rect(rect, fill=fill, color=None)

        fontsize = 20
        text_width = fitz.get_text_length(text, fontname="helv", fontsize=fontsize)
        x = max(0.0, (width - text_width) / 2.0)
        y = max(0.0, (height + fontsize) / 2.0)
        color = tuple(c / 255.0 for c in text_rgb)
        page.insert_text((x, y), text, fontsize=fontsize, fontname="helv", color=color)

        pix = page.get_pixmap(dpi=72, alpha=False)
        return _ensure_rgb_no_alpha(pix)
    finally:
        doc.close()


def generate_thumbnail(
    pdf_url: str, output_dir: Path, paper_id: str, max_pages: int = DEFAULT_MAX_PAGES
) -> Optional[str]:
    """
    Download a PDF and create a horizontal thumbnail of the first few pages.

    - Success: writes `{paper_id}.png` and `{paper_id}_small.png`
    - Failure: writes `{paper_id}_placeholder.png` and `{paper_id}_placeholder_small.png`
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdf_bytes = _download_pdf(pdf_url)
    except Exception:  # noqa: BLE001 - best-effort thumbnail generation
        pdf_bytes = b""
    return generate_thumbnail_from_pdf_bytes(
        pdf_bytes,
        output_dir,
        paper_id,
        max_pages=max_pages,
        dpi=DEFAULT_DPI,
        lowres_max_width=LOWRES_MAX_WIDTH,
    )


def generate_thumbnail_from_pdf_bytes(
    pdf_bytes: bytes,
    output_dir: Path,
    paper_id: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    *,
    dpi: int = DEFAULT_DPI,
    lowres_max_width: int = LOWRES_MAX_WIDTH,
) -> str:
    """
    Render a PDF (already in memory) and create a horizontal thumbnail of the first few pages.

    - Success: writes `{paper_id}.png` and `{paper_id}_small.png`
    - Failure: writes `{paper_id}_placeholder.png` and `{paper_id}_placeholder_small.png`
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_stem(paper_id)

    try:
        pixmaps = _render_pages_from_bytes(pdf_bytes, max_pages=max_pages, dpi=dpi)
        stitched = _stitch_horiz(pixmaps, dpi=dpi)

        out_png = output_dir / f"{safe_id}.png"
        out_small_png = output_dir / f"{safe_id}_small.png"
        _save_png(stitched, out_png, dpi=dpi)
        try:
            _save_lowres_variant_pixmap(stitched, out_small_png, max_width=lowres_max_width)
        except Exception:
            pass
        return str(out_png)
    except Exception:  # noqa: BLE001 - best-effort thumbnail generation
        placeholder = _placeholder_pixmap()
        out_png = output_dir / f"{safe_id}_placeholder.png"
        out_small_png = output_dir / f"{safe_id}_placeholder_small.png"
        _save_png(placeholder, out_png, dpi=72)
        try:
            _save_lowres_variant_pixmap(placeholder, out_small_png, max_width=lowres_max_width)
        except Exception:
            pass
        return str(out_png)


def generate_thumbnails(
    pdf_path: Path,
    out_png: Path,
    out_small_png: Path,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    dpi: int = DEFAULT_DPI,
    lowres_max_width: int = LOWRES_MAX_WIDTH,
) -> Tuple[int, int, int, int]:
    """
    Returns: (width, height, small_width, small_height)
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_small_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        pixmaps = _render_pages_from_path(pdf_path, max_pages=max_pages, dpi=dpi)
        stitched = _stitch_horiz(pixmaps, dpi=dpi)
    except Exception:  # noqa: BLE001 - best-effort thumbnail generation
        stitched = _placeholder_pixmap(width=600, height=200, text=DEFAULT_PLACEHOLDER_TEXT)
        dpi = 72

    _save_png(stitched, out_png, dpi=dpi)
    try:
        small_w, small_h = _save_lowres_variant_pixmap(stitched, out_small_png, max_width=lowres_max_width)
    except Exception:
        small_w, small_h = stitched.width, stitched.height
    return stitched.width, stitched.height, small_w, small_h
