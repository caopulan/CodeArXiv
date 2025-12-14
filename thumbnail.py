from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import fitz


DEFAULT_DPI = 300
DEFAULT_MAX_PAGES = 5
LOWRES_MAX_WIDTH = 1200


def _render_pages(pdf_path: Path, *, max_pages: int = DEFAULT_MAX_PAGES, dpi: int = DEFAULT_DPI) -> List[fitz.Pixmap]:
    doc = fitz.open(str(pdf_path))
    try:
        n = min(max_pages, doc.page_count)
        pixmaps: List[fitz.Pixmap] = []
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            # Ensure consistent RGB + no alpha.
            if pix.colorspace is not None and pix.colorspace != fitz.csRGB:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)
            pixmaps.append(pix)
        return pixmaps
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


def _save_lowres_variant(
    highres_png: Path, lowres_png: Path, *, max_width: int = LOWRES_MAX_WIDTH
) -> Tuple[int, int]:
    lowres_png.parent.mkdir(parents=True, exist_ok=True)

    pix = fitz.Pixmap(str(highres_png))
    if pix.colorspace is not None and pix.colorspace != fitz.csRGB:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)

    if pix.width <= max_width:
        pix.save(str(lowres_png))
        return pix.width, pix.height

    scale = max_width / pix.width
    w = int(max_width)
    h = max(1, int(round(pix.height * scale)))
    low = fitz.Pixmap(pix, w, h)
    low.save(str(lowres_png))
    return w, h


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

    pixmaps = _render_pages(pdf_path, max_pages=max_pages, dpi=dpi)
    stitched = _stitch_horiz(pixmaps, dpi=dpi)
    stitched.save(str(out_png))

    small_w, small_h = _save_lowres_variant(out_png, out_small_png, max_width=lowres_max_width)
    return stitched.width, stitched.height, small_w, small_h
