#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

import run_daily as daily


PAPERS_COOL_URL_TEMPLATE = "https://papers.cool/arxiv/{category}"

PAPER_START_RE = re.compile(
    r'<div[^>]*\bid="([^"]+)"[^>]*\bclass="[^"]*\bpaper\b[^"]*"',
    flags=re.IGNORECASE,
)
TITLE_RE = re.compile(r'class="title-link\b[^"]*"[^>]*>(.*?)</a>', flags=re.IGNORECASE | re.DOTALL)
PDF_RE = re.compile(r'class="title-pdf\b[^"]*"[^>]*\bdata="([^"]+)"', flags=re.IGNORECASE)
SUMMARY_RE = re.compile(r'<p[^>]*\bclass="summary\b[^"]*"[^>]*>(.*?)</p>', flags=re.IGNORECASE | re.DOTALL)
DATE_RE = re.compile(
    r'<p[^>]*\bclass="metainfo date\b[^"]*"[^>]*>.*?<span[^>]*\bclass="date-data\b[^"]*"[^>]*>(.*?)</span>',
    flags=re.IGNORECASE | re.DOTALL,
)
AUTHOR_RE = re.compile(r'class="author\b[^"]*"[^>]*>(.*?)</a>', flags=re.IGNORECASE | re.DOTALL)


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_str(name: str) -> Optional[str]:
    raw = (os.getenv(name) or "").strip()
    return raw or None


def _papers_cool_url(category: str, *, list_date: str, skip: int, show: int) -> str:
    params = urllib.parse.urlencode(
        {
            "date": list_date,
            "skip": str(int(skip)),
            "show": str(int(show)),
        }
    )
    return f"{PAPERS_COOL_URL_TEMPLATE.format(category=category)}?{params}"


def _clean_html_text(fragment: str) -> str:
    if not fragment:
        return ""
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_publish_date(text: str, fallback: date) -> date:
    if not text:
        return fallback
    for part in text.strip().split():
        try:
            return date.fromisoformat(part)
        except ValueError:
            continue
    return fallback


def _parse_papers_cool_html(html_text: str, *, fallback_date: date) -> Dict[str, Dict[str, Any]]:
    papers: Dict[str, Dict[str, Any]] = {}
    matches = list(PAPER_START_RE.finditer(html_text))
    for idx, m in enumerate(matches):
        paper_id = m.group(1).strip()
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(html_text)
        block = html_text[start:end]

        title_m = TITLE_RE.search(block)
        title_en = _clean_html_text(title_m.group(1)) if title_m else ""

        summary_m = SUMMARY_RE.search(block)
        abstract_en = _clean_html_text(summary_m.group(1)) if summary_m else ""

        pdf_url: Optional[str] = None
        pdf_m = PDF_RE.search(block)
        if pdf_m:
            pdf_url = (pdf_m.group(1) or "").strip()
            if pdf_url and not pdf_url.endswith(".pdf"):
                pdf_url = f"{pdf_url}.pdf"

        date_m = DATE_RE.search(block)
        publish_date = _parse_publish_date(_clean_html_text(date_m.group(1)) if date_m else "", fallback=fallback_date)

        authors = [_clean_html_text(a) for a in AUTHOR_RE.findall(block)]

        papers[paper_id] = {
            "arxiv_id": paper_id,
            "title_en": title_en,
            "abstract_en": abstract_en,
            "authors": authors,
            "pdf_url": pdf_url,
            "published": publish_date.isoformat(),
        }
    return papers


def fetch_papers_cool(
    *,
    category: str,
    list_date: date,
    cfg: daily.FetchConfig,
    show_size: int,
    max_pages: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch paper metadata from papers.cool for a given arXiv category and list date.

    Returns a mapping: {arxiv_id: {"title_en": ..., "abstract_en": ..., ...}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    date_str = list_date.isoformat()

    for page in range(max(1, int(max_pages))):
        skip = page * int(show_size)
        url = _papers_cool_url(category, list_date=date_str, skip=skip, show=int(show_size))
        html_text = daily._fetch_text(url, cfg)
        page_items = _parse_papers_cool_html(html_text, fallback_date=list_date)
        if not page_items:
            break
        out.update(page_items)
        if len(page_items) < int(show_size):
            break

    return out


def _merge_list_fields(entry: Dict[str, Any], *, list_date: str, categories: List[str], show_size: int) -> None:
    entry["list_date"] = list_date
    entry.setdefault("errors", {})

    cats_existing = set(entry.get("list_categories") or [])
    if entry.get("list_category"):
        cats_existing.add(str(entry["list_category"]))
    cats_existing.update(categories)
    entry["list_categories"] = sorted(str(c).strip() for c in cats_existing if str(c).strip())

    if not entry.get("list_category") and entry["list_categories"]:
        entry["list_category"] = entry["list_categories"][0]

    urls_existing = entry.get("list_urls") if isinstance(entry.get("list_urls"), dict) else {}
    for c in entry["list_categories"]:
        urls_existing[c] = _papers_cool_url(c, list_date=list_date, skip=0, show=show_size)
    entry["list_urls"] = urls_existing

    if not entry.get("list_url") and entry.get("list_category"):
        entry["list_url"] = _papers_cool_url(str(entry["list_category"]), list_date=list_date, skip=0, show=show_size)


def _fill_missing_from_papers_cool(entry: Dict[str, Any], cool: Dict[str, Any]) -> None:
    if cool.get("title_en") and not (entry.get("title_en") or "").strip():
        entry["title_en"] = cool["title_en"]
    if cool.get("abstract_en") and not (entry.get("abstract_en") or "").strip():
        entry["abstract_en"] = cool["abstract_en"]
    if cool.get("authors") and not entry.get("authors"):
        entry["authors"] = cool["authors"]
    if cool.get("pdf_url") and not (entry.get("pdf_url") or "").strip():
        entry["pdf_url"] = cool["pdf_url"]
    if cool.get("published") and not (entry.get("published") or "").strip():
        entry["published"] = cool["published"]


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    daily.configure_data_paths()
    default_codex_model = _env_str("CODEX_MODEL")
    default_codex_batch_size = max(1, _env_int("CODEX_BATCH_SIZE", 5))
    default_codex_timeout = max(1, _env_int("CODEX_TIMEOUT", 300))
    default_codex_sleep = max(0.0, _env_float("CODEX_SLEEP", 0.2))
    default_codex_overwrite = _env_bool("CODEX_OVERWRITE", False)

    parser = argparse.ArgumentParser(
        description=(
            "Fetch a specified list date from papers.cool, then reuse run_daily.py's arXiv metadata fetch, "
            "thumbnail generation, and (optional) Codex JSON fill."
        )
    )
    parser.add_argument("--date", required=True, help="List date (YYYY-MM-DD).")
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(daily.LIST_CATEGORIES),
        help=f"Comma-separated arXiv categories (default: {','.join(daily.LIST_CATEGORIES)}).",
    )
    parser.add_argument(
        "--show-size",
        type=int,
        default=2000,
        help="papers.cool page size (show=...).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Max pagination pages per category.",
    )
    parser.add_argument(
        "--thumb-workers",
        type=int,
        default=daily.DEFAULT_THUMB_WORKERS,
        help=f"Thumbnail generation concurrency (default: {daily.DEFAULT_THUMB_WORKERS}).",
    )
    parser.add_argument(
        "--thumb-overwrite",
        action="store_true",
        help="Regenerate thumbnails even if image files already exist.",
    )
    parser.add_argument(
        "--thumb-timing",
        action="store_true",
        help="Print per-paper thumbnail step timings as JSON log lines.",
    )
    parser.add_argument(
        "--skip-codex",
        action="store_true",
        help="Skip running codex_fill_zh.py after metadata is fetched.",
    )
    parser.add_argument(
        "--codex-fill",
        type=Path,
        default=Path(__file__).resolve().parent / "codex_fill_zh.py",
        help="Path to codex_fill_zh.py (used after metadata is fetched).",
    )
    parser.add_argument(
        "--codex-model",
        type=str,
        default=default_codex_model,
        help="Forwarded to codex_fill_zh.py --model (default: CODEX_MODEL).",
    )
    parser.add_argument(
        "--codex-batch-size",
        type=int,
        default=default_codex_batch_size,
        help="Forwarded to codex_fill_zh.py --batch-size (default: CODEX_BATCH_SIZE).",
    )
    parser.add_argument(
        "--codex-timeout",
        type=int,
        default=default_codex_timeout,
        help="Forwarded to codex_fill_zh.py --timeout (default: CODEX_TIMEOUT).",
    )
    parser.add_argument(
        "--codex-sleep",
        type=float,
        default=default_codex_sleep,
        help="Forwarded to codex_fill_zh.py --sleep (default: CODEX_SLEEP).",
    )
    parser.add_argument(
        "--codex-overwrite",
        action=argparse.BooleanOptionalAction,
        default=default_codex_overwrite,
        help="Forwarded to codex_fill_zh.py --overwrite.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip generating DashScope embeddings into the results JSON.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=daily.DEFAULT_EMBEDDING_BATCH_SIZE,
        help=f"DashScope embedding batch size (default: {daily.DEFAULT_EMBEDDING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--embedding-timeout",
        type=int,
        default=daily.DEFAULT_EMBEDDING_TIMEOUT_S,
        help=f"DashScope embedding request timeout seconds (default: {daily.DEFAULT_EMBEDDING_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--embedding-sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between embedding batches (default: 0).",
    )
    parser.add_argument(
        "--embedding-max-chars",
        type=int,
        default=daily.DEFAULT_EMBEDDING_MAX_CHARS,
        help=f"Max chars per embedding input text (default: {daily.DEFAULT_EMBEDDING_MAX_CHARS}).",
    )

    args = parser.parse_args(argv)

    try:
        list_date = date.fromisoformat(str(args.date))
    except ValueError as e:
        raise SystemExit(f"Invalid --date {args.date!r} (expected YYYY-MM-DD): {e}") from e

    categories = [c.strip() for c in str(args.categories).split(",") if c.strip()]
    if not categories:
        raise SystemExit("No categories specified (use --categories).")

    cfg = daily.FetchConfig()
    embed_config = None
    if not args.skip_embedding:
        embed_config = daily._load_embedding_config_from_env(
            batch_size=args.embedding_batch_size,
            timeout_s=args.embedding_timeout,
            sleep_s=args.embedding_sleep,
            max_chars=args.embedding_max_chars,
        )
        if embed_config is None:
            print(
                "[Step5] embeddings skipped (missing EMBEDDING_MODEL/EMBEDDING_API_KEY).",
                file=sys.stderr,
                flush=True,
            )
    show_size = max(1, int(args.show_size))
    max_pages = max(1, int(args.max_pages))

    # Step1: fetch list from papers.cool, merge categories & minimal metadata into results.
    id_to_cats: Dict[str, set[str]] = {}
    cool_map: Dict[str, Dict[str, Any]] = {}
    for category in categories:
        items = fetch_papers_cool(
            category=category,
            list_date=list_date,
            cfg=cfg,
            show_size=show_size,
            max_pages=max_pages,
        )
        for arxiv_id, payload in items.items():
            id_to_cats.setdefault(arxiv_id, set()).add(category)
            existing = cool_map.get(arxiv_id) or {}
            merged = dict(existing)
            for k, v in payload.items():
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                if k == "authors":
                    prev = merged.get("authors") if isinstance(merged.get("authors"), list) else []
                    now = v if isinstance(v, list) else []
                    merged["authors"] = list(dict.fromkeys([*prev, *now]))
                else:
                    merged.setdefault(k, v)
            cool_map[arxiv_id] = merged

    ids = sorted(cool_map.keys())
    date_str = list_date.isoformat()
    if not ids:
        print(
            f"[Step1] {date_str}: no ids; skip writing results and thumbnails.",
            flush=True,
        )
        return 0
    results_path = daily.RESULTS_DIR / f"{date_str}.json"

    results: Dict[str, Dict[str, Any]] = {}
    if results_path.exists():
        try:
            with results_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                results = loaded
        except Exception:
            results = {}

    for arxiv_id in ids:
        cats_now = sorted(id_to_cats.get(arxiv_id, set()))
        defaults = {
            "arxiv_id": arxiv_id,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "list_date": date_str,
            "list_category": cats_now[0] if cats_now else "",
            "list_categories": cats_now,
            "list_source": "papers.cool",
            "list_url": _papers_cool_url(cats_now[0], list_date=date_str, skip=0, show=show_size) if cats_now else "",
            "list_urls": {c: _papers_cool_url(c, list_date=date_str, skip=0, show=show_size) for c in cats_now},
            "created_at": daily._now_iso(),
            "updated_at": daily._now_iso(),
            "metadata_fetched": False,
            "translated": False,
            "summary_generated": False,
            "thumbnails_generated": False,
            "embedding": None,
            "errors": {},
        }
        entry = results.setdefault(arxiv_id, defaults)
        _merge_list_fields(entry, list_date=date_str, categories=cats_now, show_size=show_size)
        entry.setdefault("embedding", None)
        _fill_missing_from_papers_cool(entry, cool_map.get(arxiv_id) or {})

    daily._json_dump_atomic(results, results_path)
    print(f"[Step1] {date_str}: saved {len(ids)} ids -> {results_path}", flush=True)

    # Step2: fetch arXiv API metadata (title/abstract/authors/categories/comments/...), and persist.
    ids_need_meta = [i for i in ids if not results.get(i, {}).get("metadata_fetched")]
    for chunk in daily._chunked(ids_need_meta, 50):
        meta_map = daily._fetch_arxiv_metadata(chunk, cfg)
        for arxiv_id in chunk:
            entry = results.setdefault(arxiv_id, {"arxiv_id": arxiv_id})
            entry.setdefault("embedding", None)
            entry["updated_at"] = daily._now_iso()
            try:
                meta = meta_map[arxiv_id]
                entry.update(meta)
                entry["metadata_fetched"] = True
                entry.setdefault("errors", {}).pop("metadata", None)
            except Exception as e:  # noqa: BLE001
                entry["metadata_fetched"] = False
                entry.setdefault("errors", {})["metadata"] = str(e)

            # Ensure we still have at least title/abstract from papers.cool for Codex fill.
            _fill_missing_from_papers_cool(entry, cool_map.get(arxiv_id) or {})

            daily._json_dump_atomic(results, results_path)
        time.sleep(0.3)
    print(f"[Step2] {date_str}: metadata fetched -> {results_path}", flush=True)

    pdf_cfg = daily.FetchConfig(
        timeout_s=max(cfg.timeout_s, 60),
        retries=cfg.retries,
        backoff_s=cfg.backoff_s,
        user_agent=cfg.user_agent,
    )

    # Step4 (Codex fill) can run in parallel with Step3 (thumbnails).
    codex_proc: Optional[subprocess.Popen[str]] = None
    codex_rc = 0
    if not args.skip_codex:
        codex_fill_path = Path(args.codex_fill).resolve()
        if not codex_fill_path.exists():
            print(
                f"[Step4] {date_str}: ERROR codex_fill_zh not found: {codex_fill_path}",
                file=sys.stderr,
                flush=True,
            )
            codex_rc = 2
        else:
            cmd = daily._build_codex_fill_cmd(args, input_path=results_path)
            print(f"[Step4] {date_str}: start codex fill -> {results_path}", flush=True)
            codex_proc = subprocess.Popen(cmd)

    # Step3: thumbnails
    thumb_updates: Dict[str, daily.ThumbnailUpdate] = {}
    to_generate: List[Tuple[str, str]] = []
    for arxiv_id in ids:
        entry = results.get(arxiv_id) or {}
        pdf_url = (entry.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf").strip()
        large_png, small_png = daily._thumbnail_paths(date_str, arxiv_id)
        up_to_date = (
            entry.get("thumbnails_generated") is True
            and entry.get("thumbnail_version") == daily.THUMBNAIL_VERSION
        )
        if not args.thumb_overwrite and up_to_date and large_png.exists() and small_png.exists():
            thumb_updates[arxiv_id] = daily.ThumbnailUpdate(
                arxiv_id=arxiv_id, ok=True, large_png=large_png, small_png=small_png
            )
        else:
            to_generate.append((arxiv_id, pdf_url))

    workers = max(1, int(args.thumb_workers))
    if to_generate:
        print(
            f"[Step3] {date_str}: thumbnails start (workers={workers}, tasks={len(to_generate)})",
            flush=True,
        )
    else:
        print(f"[Step3] {date_str}: thumbnails start (nothing to generate)", flush=True)

    embedding_updates: Dict[str, daily.EmbeddingUpdate] = {}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_id = {
            ex.submit(
                daily._thumbnail_worker,
                arxiv_id=arxiv_id,
                pdf_url=pdf_url,
                date_str=date_str,
                pdf_cfg=pdf_cfg,
                overwrite=args.thumb_overwrite,
                log_timings=args.thumb_timing,
            ): arxiv_id
            for arxiv_id, pdf_url in to_generate
        }
        embedding_updates = daily._compute_embedding_updates(
            date_str=date_str,
            ids=ids,
            results=results,
            cfg=cfg,
            embed_config=embed_config,
        )
        for fut in concurrent.futures.as_completed(fut_to_id):
            arxiv_id = fut_to_id[fut]
            try:
                upd = fut.result()
            except Exception as e:  # noqa: BLE001
                large_png, small_png = daily._thumbnail_paths(date_str, arxiv_id)
                upd = daily.ThumbnailUpdate(
                    arxiv_id=arxiv_id, ok=False, large_png=large_png, small_png=small_png, error=str(e)
                )
            thumb_updates[arxiv_id] = upd
            completed += 1
            if completed % 10 == 0 or completed == len(to_generate):
                print(f"[Step3] {date_str}: thumbnails {completed}/{len(to_generate)}", flush=True)
    print(f"[Step3] {date_str}: thumbnails done -> {daily.IMAGES_DIR / date_str}", flush=True)

    if codex_proc is not None:
        codex_rc = codex_proc.wait()
    if not args.skip_codex:
        print(f"[Step4] {date_str}: codex fill done rc={codex_rc} -> {results_path}", flush=True)

    # Reload results to include Codex updates, then apply thumbnail updates in one atomic write.
    with results_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Unexpected results format in {results_path} (expected dict).")
    results_latest: Dict[str, Dict[str, Any]] = loaded

    thumb_changed = 0
    for arxiv_id, upd in thumb_updates.items():
        entry = results_latest.get(arxiv_id) or {"arxiv_id": arxiv_id}
        if daily._apply_thumbnail_update(entry, upd):
            thumb_changed += 1
        results_latest[arxiv_id] = entry

    embed_changed = 0
    for arxiv_id, upd in embedding_updates.items():
        entry = results_latest.get(arxiv_id) or {"arxiv_id": arxiv_id}
        if upd.ok and daily._has_valid_embedding(entry.get("embedding")):
            continue
        if not upd.ok and daily._has_valid_embedding(entry.get("embedding")):
            continue
        if daily._apply_embedding_update(entry, upd):
            embed_changed += 1
        results_latest[arxiv_id] = entry

    changed_total = thumb_changed + embed_changed
    if changed_total:
        daily._json_dump_atomic(results_latest, results_path)
        if thumb_changed:
            print(
                f"[Step3] {date_str}: thumbnails saved (updated {thumb_changed}) -> {results_path}",
                flush=True,
            )
        if embed_changed:
            print(
                f"[Step5] {date_str}: embeddings saved (updated {embed_changed}) -> {results_path}",
                flush=True,
            )

    return codex_rc


if __name__ == "__main__":
    raise SystemExit(main())
