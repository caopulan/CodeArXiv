#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from thumbnail import generate_thumbnails


LIST_CATEGORIES = ["cs.CV", "cs.AI", "cs.CG", "cs.CL"]
LIST_SHOW = 2000
LIST_URL_TEMPLATE = "https://arxiv.org/list/{category}/pastweek?show={show}"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
RESULTS_DIR = Path("CodeArXiv-data")
IMAGES_DIR = RESULTS_DIR / "images"
THUMB_PAGES = 5
THUMB_RENDER_DPI = 300
THUMB_LOWRES_MAX_WIDTH = 1200
THUMBNAIL_VERSION = 2
DEFAULT_THUMB_WORKERS = 10


@dataclass(frozen=True)
class ThumbnailUpdate:
    arxiv_id: str
    ok: bool
    large_png: Path
    small_png: Path
    error: Optional[str] = None


@dataclass(frozen=True)
class FetchConfig:
    timeout_s: int = 30
    retries: int = 5
    backoff_s: float = 1.5
    user_agent: str = "CodeXivBot/0.1 (+https://arxiv.org)"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_dump_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _fetch_bytes(url: str, cfg: FetchConfig, data: Optional[bytes] = None) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(cfg.retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "User-Agent": cfg.user_agent,
                    "Accept": "*/*",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 - capture & retry network failures
            last_err = e
            sleep_s = cfg.backoff_s * (attempt + 1)
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def _fetch_text(url: str, cfg: FetchConfig) -> str:
    return _fetch_bytes(url, cfg).decode("utf-8", "replace")


def _list_url(category: str) -> str:
    return LIST_URL_TEMPLATE.format(category=category, show=LIST_SHOW)


def _parse_latest_date_and_ids(list_html: str) -> Tuple[str, List[str]]:
    h3s = list(re.finditer(r"<h3[^>]*>([^<]+)</h3>", list_html))
    if not h3s:
        raise RuntimeError("Failed to locate date headings (<h3>) on listing page.")

    heading = h3s[0].group(1).strip()
    heading_date = heading.split(" (showing", 1)[0].strip()
    dt = datetime.strptime(heading_date, "%a, %d %b %Y")
    date_str = dt.strftime("%Y-%m-%d")

    start = h3s[0].end()
    end = h3s[1].start() if len(h3s) > 1 else len(list_html)
    section = list_html[start:end]
    ids = re.findall(r'href\s*=\s*"/abs/([^"]+)"', section)
    base_ids: List[str] = []
    seen: set[str] = set()
    for raw in ids:
        base = re.sub(r"v\d+$", "", raw)
        if base not in seen:
            seen.add(base)
            base_ids.append(base)
    return date_str, base_ids


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _parse_arxiv_api_feed(xml_text: str) -> Dict[str, Dict[str, Any]]:
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(xml_text)
    out: Dict[str, Dict[str, Any]] = {}

    for entry in root.findall("atom:entry", ns):
        entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        m = re.search(r"/abs/([^/]+)$", entry_id)
        if not m:
            continue
        versioned_id = m.group(1)
        base_id = re.sub(r"v\d+$", "", versioned_id)

        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        title = re.sub(r"\\s+", " ", title)

        authors = []
        for author in entry.findall("atom:author", ns):
            name = (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)

        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        summary = re.sub(r"\\s+", " ", summary)

        comment = (entry.findtext("arxiv:comment", default="", namespaces=ns) or "").strip()

        primary_category_el = entry.find("arxiv:primary_category", ns)
        primary_category = (
            (primary_category_el.attrib.get("term") if primary_category_el is not None else "") or ""
        ).strip()

        categories = []
        for cat in entry.findall("atom:category", ns):
            term = (cat.attrib.get("term") or "").strip()
            if term:
                categories.append(term)

        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()

        out[base_id] = {
            "arxiv_id": base_id,
            "arxiv_id_versioned": versioned_id,
            "title_en": title,
            "authors": authors,
            "abstract_en": summary,
            "comments": comment,
            "primary_category": primary_category,
            "categories": categories,
            "published": published,
            "updated": updated,
        }
    return out


def _fetch_arxiv_metadata(ids: List[str], cfg: FetchConfig) -> Dict[str, Dict[str, Any]]:
    params = {"id_list": ",".join(ids), "max_results": str(len(ids))}
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    xml_text = _fetch_text(url, cfg)
    return _parse_arxiv_api_feed(xml_text)


def _thumbnail_paths(date_str: str, arxiv_id: str) -> Tuple[Path, Path]:
    safe_id = arxiv_id.replace("/", "_")
    out_dir = IMAGES_DIR / date_str
    return out_dir / f"{safe_id}.png", out_dir / f"{safe_id}_small.png"


def _generate_thumbnails_for_pdf(pdf_path: Path, date_str: str, arxiv_id: str) -> Tuple[Path, Path]:
    large_png, small_png = _thumbnail_paths(date_str, arxiv_id)
    generate_thumbnails(
        pdf_path,
        large_png,
        small_png,
        max_pages=THUMB_PAGES,
        dpi=THUMB_RENDER_DPI,
        lowres_max_width=THUMB_LOWRES_MAX_WIDTH,
    )
    return large_png, small_png


def _build_codex_fill_cmd(args: argparse.Namespace, *, input_path: Path) -> List[str]:
    cmd = [sys.executable, str(Path(args.codex_fill).resolve()), "--input", str(input_path)]
    if args.codex_model:
        cmd += ["--model", str(args.codex_model)]
    if args.codex_batch_size:
        cmd += ["--batch-size", str(int(args.codex_batch_size))]
    if args.codex_timeout:
        cmd += ["--timeout", str(int(args.codex_timeout))]
    if args.codex_sleep is not None:
        cmd += ["--sleep", str(float(args.codex_sleep))]
    if args.codex_overwrite:
        cmd += ["--overwrite"]
    return cmd


def _thumbnail_worker(*, arxiv_id: str, pdf_url: str, date_str: str, pdf_cfg: FetchConfig) -> ThumbnailUpdate:
    large_png, small_png = _thumbnail_paths(date_str, arxiv_id)
    try:
        with tempfile.TemporaryDirectory(prefix="codexiv_pdf_") as tmp:
            pdf_path = Path(tmp) / "paper.pdf"
            try:
                pdf_path.write_bytes(_fetch_bytes(pdf_url, pdf_cfg))
            except Exception:
                # Let thumbnail generation fall back to a placeholder image.
                pass
            _generate_thumbnails_for_pdf(pdf_path, date_str, arxiv_id)
        return ThumbnailUpdate(arxiv_id=arxiv_id, ok=True, large_png=large_png, small_png=small_png)
    except Exception as e:  # noqa: BLE001
        return ThumbnailUpdate(arxiv_id=arxiv_id, ok=False, large_png=large_png, small_png=small_png, error=str(e))


def _apply_thumbnail_update(entry: Dict[str, Any], update: ThumbnailUpdate) -> bool:
    if not isinstance(entry.get("errors"), dict):
        entry["errors"] = {}
    errors: Dict[str, Any] = entry["errors"]
    changed = False

    if update.ok:
        if entry.get("thumbnails_generated") is not True:
            entry["thumbnails_generated"] = True
            changed = True
        if entry.get("thumbnail_300_path") != str(update.large_png):
            entry["thumbnail_300_path"] = str(update.large_png)
            changed = True
        if entry.get("thumbnail_small_path") != str(update.small_png):
            entry["thumbnail_small_path"] = str(update.small_png)
            changed = True
        if entry.get("thumbnail_100_path") != str(update.small_png):
            entry["thumbnail_100_path"] = str(update.small_png)
            changed = True
        if entry.get("thumbnail_pages") != THUMB_PAGES:
            entry["thumbnail_pages"] = THUMB_PAGES
            changed = True
        if entry.get("thumbnail_render_dpi") != THUMB_RENDER_DPI:
            entry["thumbnail_render_dpi"] = THUMB_RENDER_DPI
            changed = True
        if entry.get("thumbnail_small_max_width") != THUMB_LOWRES_MAX_WIDTH:
            entry["thumbnail_small_max_width"] = THUMB_LOWRES_MAX_WIDTH
            changed = True
        if entry.get("thumbnail_version") != THUMBNAIL_VERSION:
            entry["thumbnail_version"] = THUMBNAIL_VERSION
            changed = True

        if errors.get("thumbnail"):
            errors.pop("thumbnail", None)
            changed = True
    else:
        if entry.get("thumbnails_generated") is not False:
            entry["thumbnails_generated"] = False
            changed = True
        err = update.error or "Unknown error"
        if errors.get("thumbnail") != err:
            errors["thumbnail"] = err
            changed = True

    if changed:
        entry["updated_at"] = _now_iso()
    return changed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch arXiv list + metadata, then run thumbnail generation and (optional) Codex JSON fill in parallel."
        )
    )
    parser.add_argument(
        "--thumb-workers",
        type=int,
        default=DEFAULT_THUMB_WORKERS,
        help=f"Thumbnail generation concurrency (default: {DEFAULT_THUMB_WORKERS}).",
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
    parser.add_argument("--codex-model", type=str, default=None, help="Forwarded to codex_fill_zh.py --model.")
    parser.add_argument(
        "--codex-batch-size", type=int, default=5, help="Forwarded to codex_fill_zh.py --batch-size."
    )
    parser.add_argument("--codex-timeout", type=int, default=300, help="Forwarded to codex_fill_zh.py --timeout.")
    parser.add_argument("--codex-sleep", type=float, default=0.2, help="Forwarded to codex_fill_zh.py --sleep.")
    parser.add_argument(
        "--codex-overwrite",
        action="store_true",
        help="Forwarded to codex_fill_zh.py --overwrite.",
    )

    args = parser.parse_args(argv)
    cfg = FetchConfig()
    overall_rc = 0

    grouped: Dict[str, Dict[str, set[str]]] = {}
    for category in LIST_CATEGORIES:
        url = _list_url(category)
        list_html = _fetch_text(url, cfg)
        date_str, ids = _parse_latest_date_and_ids(list_html)
        grouped.setdefault(date_str, {}).setdefault(category, set()).update(ids)

    for date_str in sorted(grouped.keys(), reverse=True):
        cat_map = grouped[date_str]
        ids_set: set[str] = set()
        id_to_cats: Dict[str, set[str]] = {}
        for category, ids in cat_map.items():
            ids_set.update(ids)
            for arxiv_id in ids:
                id_to_cats.setdefault(arxiv_id, set()).add(category)
        ids = sorted(ids_set)

        results_path = RESULTS_DIR / f"{date_str}.json"
        results: Dict[str, Dict[str, Any]] = {}
        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                results = loaded

        for arxiv_id in ids:
            cats_now = sorted(id_to_cats.get(arxiv_id, set()))
            defaults = {
                "arxiv_id": arxiv_id,
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "list_date": date_str,
                "list_category": cats_now[0] if cats_now else "",
                "list_categories": cats_now,
                "list_url": _list_url(cats_now[0]) if cats_now else "",
                "list_urls": {c: _list_url(c) for c in cats_now},
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "metadata_fetched": False,
                "translated": False,
                "summary_generated": False,
                "thumbnails_generated": False,
                "errors": {},
            }
            entry = results.setdefault(arxiv_id, defaults)
            entry["list_date"] = date_str
            entry.setdefault("errors", {})

            cats_existing = set(entry.get("list_categories") or [])
            if entry.get("list_category"):
                cats_existing.add(str(entry["list_category"]))
            cats_existing.update(cats_now)
            entry["list_categories"] = sorted(cats_existing)

            if not entry.get("list_category") and entry["list_categories"]:
                entry["list_category"] = entry["list_categories"][0]
            if not entry.get("list_url") and entry.get("list_category"):
                entry["list_url"] = _list_url(str(entry["list_category"]))

            urls_existing = entry.get("list_urls") if isinstance(entry.get("list_urls"), dict) else {}
            for c in entry["list_categories"]:
                urls_existing[c] = _list_url(c)
            entry["list_urls"] = urls_existing

        _json_dump_atomic(results, results_path)
        print(f"[Step1] {date_str}: saved {len(ids)} ids -> {results_path}")

        ids_need_meta = [i for i in ids if not results.get(i, {}).get("metadata_fetched")]
        for chunk in _chunked(ids_need_meta, 50):
            meta_map = _fetch_arxiv_metadata(chunk, cfg)
            for arxiv_id in chunk:
                entry = results.setdefault(arxiv_id, {"arxiv_id": arxiv_id})
                entry["updated_at"] = _now_iso()
                try:
                    meta = meta_map[arxiv_id]
                    entry.update(meta)
                    entry["metadata_fetched"] = True
                    entry.setdefault("errors", {}).pop("metadata", None)
                except Exception as e:  # noqa: BLE001
                    entry["metadata_fetched"] = False
                    entry.setdefault("errors", {})["metadata"] = str(e)
                _json_dump_atomic(results, results_path)
            time.sleep(0.3)
        print(f"[Step2] {date_str}: metadata fetched -> {results_path}")

        pdf_cfg = FetchConfig(
            timeout_s=max(cfg.timeout_s, 60),
            retries=cfg.retries,
            backoff_s=cfg.backoff_s,
            user_agent=cfg.user_agent,
        )

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
                cmd = _build_codex_fill_cmd(args, input_path=results_path)
                print(f"[Step4] {date_str}: start codex fill -> {results_path}", flush=True)
                codex_proc = subprocess.Popen(cmd)

        thumb_updates: Dict[str, ThumbnailUpdate] = {}
        to_generate: List[Tuple[str, str]] = []
        for arxiv_id in ids:
            entry = results.get(arxiv_id) or {}
            pdf_url = (entry.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf").strip()
            large_png, small_png = _thumbnail_paths(date_str, arxiv_id)
            if large_png.exists() and small_png.exists():
                thumb_updates[arxiv_id] = ThumbnailUpdate(
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

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_id = {
                ex.submit(_thumbnail_worker, arxiv_id=arxiv_id, pdf_url=pdf_url, date_str=date_str, pdf_cfg=pdf_cfg): (
                    arxiv_id
                )
                for arxiv_id, pdf_url in to_generate
            }
            for fut in concurrent.futures.as_completed(fut_to_id):
                arxiv_id = fut_to_id[fut]
                try:
                    upd = fut.result()
                except Exception as e:  # noqa: BLE001
                    large_png, small_png = _thumbnail_paths(date_str, arxiv_id)
                    upd = ThumbnailUpdate(
                        arxiv_id=arxiv_id, ok=False, large_png=large_png, small_png=small_png, error=str(e)
                    )
                thumb_updates[arxiv_id] = upd
                completed += 1
                if completed % 10 == 0 or completed == len(to_generate):
                    print(f"[Step3] {date_str}: thumbnails {completed}/{len(to_generate)}", flush=True)
        print(f"[Step3] {date_str}: thumbnails done -> {IMAGES_DIR / date_str}", flush=True)

        if codex_proc is not None:
            codex_rc = codex_proc.wait()
        if not args.skip_codex:
            print(f"[Step4] {date_str}: codex fill done rc={codex_rc} -> {results_path}", flush=True)

        if codex_rc != 0 and overall_rc == 0:
            overall_rc = codex_rc

        # Reload results to include Codex updates, then apply thumbnail updates in one atomic write.
        with results_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Unexpected results format in {results_path} (expected dict).")
        results_latest: Dict[str, Dict[str, Any]] = loaded

        changed_total = 0
        for arxiv_id, upd in thumb_updates.items():
            entry = results_latest.get(arxiv_id) or {"arxiv_id": arxiv_id}
            if _apply_thumbnail_update(entry, upd):
                changed_total += 1
            results_latest[arxiv_id] = entry

        if changed_total:
            _json_dump_atomic(results_latest, results_path)
            print(f"[Step3] {date_str}: thumbnails saved (updated {changed_total}) -> {results_path}", flush=True)

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
