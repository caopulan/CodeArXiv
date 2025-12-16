#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from thumbnail import generate_thumbnails, generate_thumbnails_from_pdf_bytes


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


DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
DASHSCOPE_EMBEDDING_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
DEFAULT_EMBEDDING_BATCH_SIZE = 1
DEFAULT_EMBEDDING_TIMEOUT_S = 60
DEFAULT_EMBEDDING_MAX_CHARS = 8000


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    api_key: str
    endpoint: str
    timeout_s: int = DEFAULT_EMBEDDING_TIMEOUT_S
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    sleep_s: float = 0.0
    max_chars: int = DEFAULT_EMBEDDING_MAX_CHARS


@dataclass(frozen=True)
class EmbeddingUpdate:
    arxiv_id: str
    ok: bool
    embedding: Optional[str] = None  # JSON string of a float list.
    error: Optional[str] = None


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_dump_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _resolve_dashscope_embedding_endpoint(base_url: str) -> str:
    """
    Resolve EMBEDDING_BASE_URL to DashScope's embedding endpoint.

    Accepts either:
    - full endpoint URL (already contains /services/embeddings/...), or
    - a base host (https://dashscope.aliyuncs.com), or
    - an api root (https://dashscope.aliyuncs.com/api/v1), or
    - DashScope OpenAI-compatible base (https://dashscope.aliyuncs.com/compatible-mode/v1).
    """
    raw = (base_url or "").strip()
    if not raw:
        return f"{DASHSCOPE_DEFAULT_BASE_URL}{DASHSCOPE_EMBEDDING_PATH}"

    parsed = urllib.parse.urlparse(raw)
    if not parsed.scheme:
        raw = f"https://{raw}"
        parsed = urllib.parse.urlparse(raw)

    path = parsed.path or ""
    if "/services/embeddings/" in path:
        return raw.rstrip("/")

    cleaned = raw.rstrip("/")
    path_clean = path.rstrip("/")

    compat_suffix = "/compatible-mode/v1"
    if path_clean.endswith(compat_suffix):
        prefix_path = path_clean[: -len(compat_suffix)]
        base = (
            urllib.parse.urlunparse(parsed._replace(path=prefix_path, params="", query="", fragment="")).rstrip("/")
        )
        return f"{base}{DASHSCOPE_EMBEDDING_PATH}"

    if path_clean == "/api/v1":
        return f"{cleaned}/services/embeddings/text-embedding/text-embedding"

    return f"{cleaned}{DASHSCOPE_EMBEDDING_PATH}"


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


def _load_embedding_config_from_env(
    *,
    batch_size: int,
    timeout_s: int,
    sleep_s: float,
    max_chars: int,
) -> Optional[EmbeddingConfig]:
    load_dotenv()
    model = (os.getenv("EMBEDDING_MODEL") or "").strip()
    api_key = (os.getenv("EMBEDDING_API_KEY") or "").strip()
    base_url = (os.getenv("EMBEDDING_BASE_URL") or "").strip()
    if api_key:
        api_key = os.path.expandvars(api_key).strip()
    if base_url:
        base_url = os.path.expandvars(base_url).strip()
    if not model or not api_key:
        return None
    endpoint = _resolve_dashscope_embedding_endpoint(base_url)
    return EmbeddingConfig(
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        timeout_s=max(1, int(timeout_s)),
        batch_size=max(1, int(batch_size)),
        sleep_s=max(0.0, float(sleep_s)),
        max_chars=max(0, int(max_chars)),
    )


def _has_valid_embedding(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, (list, tuple)):
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if not value or value in ("[]", "null", "None"):
            return False
        if value.startswith("[") and value.endswith("]"):
            return True
        return False
    return False


def _compose_embedding_text(entry: Dict[str, Any], *, max_chars: int) -> Optional[str]:
    title = (entry.get("title_en") or "").strip() or (entry.get("title_zh") or "").strip()
    abstract = (entry.get("abstract_en") or "").strip() or (entry.get("abstract_zh") or "").strip()
    parts = [p for p in (title, abstract) if p]
    if not parts:
        return None
    text = "\n".join(parts)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def _dashscope_embed_texts(texts: List[str], *, config: EmbeddingConfig, cfg: FetchConfig) -> List[List[float]]:
    payload = {"model": config.model, "input": {"texts": texts}}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": cfg.user_agent,
    }

    last_err: Optional[Exception] = None
    for attempt in range(cfg.retries):
        try:
            req = urllib.request.Request(config.endpoint, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
                raw = resp.read().decode("utf-8", "replace")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise RuntimeError("Unexpected DashScope response (expected JSON object).")
            if obj.get("code"):
                raise RuntimeError(f"DashScope error {obj.get('code')}: {obj.get('message')}")
            output = obj.get("output")
            if not isinstance(output, dict):
                raise RuntimeError("Unexpected DashScope response (missing output).")
            embeddings = output.get("embeddings")
            if not isinstance(embeddings, list):
                raise RuntimeError("Unexpected DashScope response (missing output.embeddings).")

            out: List[Optional[List[float]]] = [None] * len(texts)
            for item in embeddings:
                if not isinstance(item, dict):
                    continue
                vec_raw = item.get("embedding")
                if not isinstance(vec_raw, list):
                    continue
                vec = [float(x) for x in vec_raw]
                idx = item.get("text_index")
                if isinstance(idx, int) and 0 <= idx < len(out):
                    out[idx] = vec

            if any(v is None for v in out):
                # Some models may omit text_index and return embeddings in order.
                ordered = []
                for item in embeddings:
                    if not isinstance(item, dict):
                        continue
                    vec_raw = item.get("embedding")
                    if isinstance(vec_raw, list):
                        ordered.append([float(x) for x in vec_raw])
                if len(ordered) == len(out):
                    out = ordered  # type: ignore[assignment]

            missing = [i for i, v in enumerate(out) if v is None]
            if missing:
                raise RuntimeError(f"DashScope response missing embeddings for indices: {missing}")

            return [v for v in out if v is not None]
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", "replace").strip()
            except Exception:
                err_body = ""
            last_err = RuntimeError(f"DashScope HTTP {e.code} for {config.endpoint}: {err_body or e.reason}")
            if 400 <= e.code < 500 and e.code not in (408, 429):
                break
        except Exception as e:  # noqa: BLE001 - best-effort retries
            last_err = e

        time.sleep(cfg.backoff_s * (attempt + 1))

    assert last_err is not None
    raise last_err


def _compute_embedding_updates(
    *,
    date_str: str,
    ids: List[str],
    results: Dict[str, Dict[str, Any]],
    cfg: FetchConfig,
    embed_config: Optional[EmbeddingConfig],
) -> Dict[str, EmbeddingUpdate]:
    if embed_config is None:
        return {}

    pending: List[Tuple[str, str]] = []
    updates: Dict[str, EmbeddingUpdate] = {}

    for arxiv_id in ids:
        entry = results.get(arxiv_id) or {}
        if _has_valid_embedding(entry.get("embedding")):
            continue
        text = _compose_embedding_text(entry, max_chars=embed_config.max_chars)
        if not text:
            updates[arxiv_id] = EmbeddingUpdate(
                arxiv_id=arxiv_id,
                ok=False,
                error="Missing title/abstract for embedding.",
            )
            continue
        pending.append((arxiv_id, text))

    if not pending:
        print(f"[Step5] {date_str}: embeddings start (nothing to generate)", flush=True)
        return updates

    print(
        f"[Step5] {date_str}: embeddings start (batch={embed_config.batch_size}, tasks={len(pending)})",
        flush=True,
    )

    updated = 0
    for start in range(0, len(pending), embed_config.batch_size):
        batch = pending[start : start + embed_config.batch_size]
        texts = [t for _, t in batch]
        try:
            vecs = _dashscope_embed_texts(texts, config=embed_config, cfg=cfg)
            if len(vecs) != len(batch):
                raise RuntimeError(f"DashScope returned {len(vecs)} embeddings for {len(batch)} texts.")
            for (arxiv_id, _), vec in zip(batch, vecs):
                updates[arxiv_id] = EmbeddingUpdate(
                    arxiv_id=arxiv_id,
                    ok=True,
                    embedding=json.dumps(vec, ensure_ascii=False, separators=(",", ":")),
                )
                updated += 1
        except Exception as e:  # noqa: BLE001 - per-batch capture
            err = str(e)
            for arxiv_id, _ in batch:
                updates[arxiv_id] = EmbeddingUpdate(arxiv_id=arxiv_id, ok=False, error=err)

        if updated and updated % 50 == 0:
            print(f"[Step5] {date_str}: embeddings {updated}/{len(pending)}", flush=True)

        if embed_config.sleep_s:
            time.sleep(embed_config.sleep_s)

    print(f"[Step5] {date_str}: embeddings done (updated {updated})", flush=True)
    return updates


def _apply_embedding_update(entry: Dict[str, Any], update: EmbeddingUpdate) -> bool:
    if not isinstance(entry.get("errors"), dict):
        entry["errors"] = {}
    errors: Dict[str, Any] = entry["errors"]
    changed = False

    if update.ok:
        if update.embedding and entry.get("embedding") != update.embedding:
            entry["embedding"] = update.embedding
            changed = True
        if errors.get("embedding"):
            errors.pop("embedding", None)
            changed = True
    else:
        err = update.error or "Unknown error"
        if errors.get("embedding") != err:
            errors["embedding"] = err
            changed = True

    if changed:
        entry["updated_at"] = _now_iso()
    return changed


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


def _generate_thumbnails_for_pdf_bytes(pdf_bytes: bytes, date_str: str, arxiv_id: str) -> Tuple[Path, Path]:
    large_png, small_png = _thumbnail_paths(date_str, arxiv_id)
    generate_thumbnails_from_pdf_bytes(
        pdf_bytes,
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
        try:
            pdf_bytes = _fetch_bytes(pdf_url, pdf_cfg)
        except Exception:
            # Let thumbnail generation fall back to a placeholder image.
            pdf_bytes = b""
        _generate_thumbnails_for_pdf_bytes(pdf_bytes, date_str, arxiv_id)
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
    load_dotenv()
    default_codex_model = _env_str("CODEX_MODEL")
    default_codex_batch_size = max(1, _env_int("CODEX_BATCH_SIZE", 5))
    default_codex_timeout = max(1, _env_int("CODEX_TIMEOUT", 300))
    default_codex_sleep = max(0.0, _env_float("CODEX_SLEEP", 0.2))
    default_codex_overwrite = _env_bool("CODEX_OVERWRITE", False)

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
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=f"DashScope embedding batch size (default: {DEFAULT_EMBEDDING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--embedding-timeout",
        type=int,
        default=DEFAULT_EMBEDDING_TIMEOUT_S,
        help=f"DashScope embedding request timeout seconds (default: {DEFAULT_EMBEDDING_TIMEOUT_S}).",
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
        default=DEFAULT_EMBEDDING_MAX_CHARS,
        help=f"Max chars per embedding input text (default: {DEFAULT_EMBEDDING_MAX_CHARS}).",
    )

    args = parser.parse_args(argv)
    cfg = FetchConfig()
    overall_rc = 0
    embed_config = None
    if not args.skip_embedding:
        embed_config = _load_embedding_config_from_env(
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
                "embedding": None,
                "errors": {},
            }
            entry = results.setdefault(arxiv_id, defaults)
            entry["list_date"] = date_str
            entry.setdefault("embedding", None)
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
                entry.setdefault("embedding", None)
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

        embedding_updates: Dict[str, EmbeddingUpdate] = {}
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_id = {
                ex.submit(
                    _thumbnail_worker,
                    arxiv_id=arxiv_id,
                    pdf_url=pdf_url,
                    date_str=date_str,
                    pdf_cfg=pdf_cfg,
                ): arxiv_id
                for arxiv_id, pdf_url in to_generate
            }
            embedding_updates = _compute_embedding_updates(
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
                    large_png, small_png = _thumbnail_paths(date_str, arxiv_id)
                    upd = ThumbnailUpdate(
                        arxiv_id=arxiv_id,
                        ok=False,
                        large_png=large_png,
                        small_png=small_png,
                        error=str(e),
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

        thumb_changed = 0
        for arxiv_id, upd in thumb_updates.items():
            entry = results_latest.get(arxiv_id) or {"arxiv_id": arxiv_id}
            if _apply_thumbnail_update(entry, upd):
                thumb_changed += 1
            results_latest[arxiv_id] = entry

        embed_changed = 0
        for arxiv_id, upd in embedding_updates.items():
            entry = results_latest.get(arxiv_id) or {"arxiv_id": arxiv_id}
            if _has_valid_embedding(entry.get("embedding")):
                continue
            if _apply_embedding_update(entry, upd):
                embed_changed += 1
            results_latest[arxiv_id] = entry

        changed_total = thumb_changed + embed_changed
        if changed_total:
            _json_dump_atomic(results_latest, results_path)
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

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
