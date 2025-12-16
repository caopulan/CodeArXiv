"""
Lightweight JSON-backed paper storage.

Files live in PAPERS_DATA_DIR with names like YYYY-MM-DD.json. Each file contains
either:

- a JSON array of paper dicts, or
- a JSON object mapping paper_id -> paper_dict (the format produced by `run_daily.py`).
"""

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from flask import current_app


IMAGE_PATH_KEYS = (
    "image_path",
    "thumbnail_300_path",
    "thumbnail_path",
    "thumbnail_small_path",
    "thumbnail_100_path",
)


def _data_dir() -> Path:
    base_cfg = current_app.config.get("PAPERS_DATA_DIR", "CodeArXiv-data")
    base = Path(base_cfg).expanduser()
    if not base.is_absolute():
        base = (Path(current_app.root_path).parent / base).resolve()
    else:
        base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _is_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value, flags=re.IGNORECASE))


def _normalize_local_asset_path(raw_path: Any) -> Optional[str]:
    if not raw_path:
        return None
    value = str(raw_path).strip()
    if not value:
        return None
    if _is_url(value):
        return value

    # Be tolerant of values like:
    # - CodeArXiv-data/images/YYYY-MM-DD/xxxx.png
    # - /abs/path/.../CodeArXiv-data/images/YYYY-MM-DD/xxxx.png
    # - app/static/...
    value = value.replace("\\", "/")
    path_obj = Path(value)
    parts = path_obj.parts

    if "static" in parts:
        idx = parts.index("static")
        rel = Path(*parts[idx + 1 :])
        return str(rel)

    if "images" in parts:
        idx = parts.index("images")
        rel = Path(*parts[idx:])
        return str(rel)

    return value


def _local_file_exists(data_dir: Path, path_str: str) -> bool:
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj.exists()
    if (data_dir / path_obj).exists():
        return True
    # Support values that still include the data dir prefix (e.g. CodeArXiv-data/images/...).
    if (data_dir.parent / path_obj).exists():
        return True
    return False


def _infer_image_path_from_filesystem(data_dir: Path, *, date_str: str, paper_id: str) -> Optional[str]:
    if not paper_id:
        return None
    safe_id = str(paper_id).replace("/", "_").strip()
    if not safe_id:
        return None
    date_dir = data_dir / "images" / date_str
    if not date_dir.exists():
        return None

    large = date_dir / f"{safe_id}.png"
    if large.exists():
        return str(Path("images") / date_str / large.name)

    small = date_dir / f"{safe_id}_small.png"
    if small.exists():
        return str(Path("images") / date_str / small.name)
    return None


def _date_path(date_val: Union[dt.date, str]) -> Path:
    date_str = date_val if isinstance(date_val, str) else date_val.isoformat()
    return _data_dir() / f"{date_str}.json"


def list_dates() -> List[dt.date]:
    dates: List[dt.date] = []
    for path in _data_dir().glob("*.json"):
        try:
            dates.append(dt.date.fromisoformat(path.stem))
        except ValueError:
            continue
    return sorted(dates)


def latest_date() -> Optional[dt.date]:
    dates = list_dates()
    return dates[-1] if dates else None


def _load_raw(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            # Common wrappers like {"papers": [...]}.
            papers = data.get("papers")
            if isinstance(papers, list):
                return [item for item in papers if isinstance(item, dict)]

            # run_daily.py format: {"2512.12345": {...}, ...}
            if data and all(isinstance(v, dict) for v in data.values()):
                items: List[Dict[str, Any]] = []
                for paper_id, payload in data.items():
                    item = dict(payload)
                    if not item.get("id") and not item.get("paper_id") and not item.get("arxiv_id"):
                        item["id"] = str(paper_id)
                    item.setdefault("arxiv_id", str(paper_id))
                    items.append(item)
                return items
    except Exception:
        return []
    return []


def _normalize_paper(raw: Dict[str, Any]) -> Dict[str, Any]:
    paper = dict(raw)
    pid = (
        paper.get("id")
        or paper.get("paper_id")
        or paper.get("arxiv_id")
        or paper.get("arxiv_id_versioned")
        or ""
    )
    pid = str(pid).strip()
    pid = re.sub(r"v\d+$", "", pid)
    paper["id"] = pid
    if pid and not paper.get("arxiv_id"):
        paper["arxiv_id"] = pid

    if not paper.get("comment") and paper.get("comments"):
        paper["comment"] = paper.get("comments")
    if not paper.get("pub_date") and paper.get("published"):
        paper["pub_date"] = paper.get("published")
    if not paper.get("category"):
        derived_category = paper.get("list_category") or paper.get("primary_category")
        list_categories = paper.get("list_categories")
        if isinstance(list_categories, list) and list_categories:
            derived_category = ", ".join(str(c).strip() for c in list_categories if str(c).strip()) or derived_category
        paper["category"] = str(derived_category or "").strip()

    for key in IMAGE_PATH_KEYS:
        if paper.get(key):
            paper[key] = _normalize_local_asset_path(paper[key])

    if not paper.get("image_path"):
        for key in ("thumbnail_300_path", "thumbnail_path", "thumbnail_small_path", "thumbnail_100_path"):
            candidate = paper.get(key)
            if candidate:
                paper["image_path"] = candidate
                break
    # Normalize tags
    tags_raw = paper.get("tags")
    if isinstance(tags_raw, str):
        try:
            tags_raw = json.loads(tags_raw)
        except Exception:
            tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]
    if tags_raw is None:
        tags_raw = []
    if not isinstance(tags_raw, list):
        tags_raw = [tags_raw]
    paper["tags"] = [str(t).strip() for t in tags_raw if str(t).strip()]
    embedding = paper.get("embedding")
    if isinstance(embedding, tuple):
        embedding = list(embedding)
    paper["embedding"] = embedding
    if paper.get("pub_date"):
        paper["pub_date"] = str(paper["pub_date"])
    if paper.get("created_at"):
        paper["created_at"] = str(paper["created_at"])
    return paper


def _attach_images(date_str: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data_dir = _data_dir()
    for paper in papers:
        for key in IMAGE_PATH_KEYS:
            if paper.get(key):
                paper[key] = _normalize_local_asset_path(paper[key])

        image_path = paper.get("image_path")
        if image_path and not _is_url(str(image_path)):
            if not _local_file_exists(data_dir, str(image_path)):
                paper["image_path"] = None
                image_path = None

        if not image_path:
            inferred = _infer_image_path_from_filesystem(
                data_dir,
                date_str=date_str,
                paper_id=str(paper.get("id") or paper.get("arxiv_id") or "").strip(),
            )
            if inferred:
                paper["image_path"] = inferred
    return papers


def load_date(date_val: Union[dt.date, str], *, with_images: bool = True) -> List[Dict[str, Any]]:
    path = _date_path(date_val)
    raw = _load_raw(path)
    papers = [_normalize_paper(item) for item in raw if isinstance(item, dict)]
    if not with_images:
        return papers
    date_str = date_val if isinstance(date_val, str) else date_val.isoformat()
    return _attach_images(date_str, papers)


def save_date(date_val: Union[dt.date, str], papers: Iterable[Dict[str, Any]]) -> None:
    path = _date_path(date_val)
    serializable = list(papers)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def merge_papers(date_val: Union[dt.date, str], new_papers: Iterable[Dict[str, Any]]) -> int:
    """
    Merge papers for the given date. Deduplicates by id and preserves existing data.

    Returns the number of newly added ids.
    """
    existing = {p["id"]: p for p in load_date(date_val, with_images=False)}
    added = 0
    for paper in new_papers:
        normalized = _normalize_paper(paper)
        pid = normalized.get("id")
        if not pid:
            continue
        if pid in existing:
            merged = existing[pid].copy()
            for key, val in normalized.items():
                if val is None or val == "":
                    continue
                merged[key] = val
            existing[pid] = merged
        else:
            existing[pid] = normalized
            added += 1
    save_date(date_val, existing.values())
    return added


def find_by_id(paper_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[dt.date]]:
    """Search all date files (newest first) for a paper id."""
    pid = str(paper_id).strip()
    for date_val in reversed(list_dates()):
        items = load_date(date_val, with_images=False)
        for paper in items:
            if paper.get("id") == pid:
                return _attach_images(date_val.isoformat(), [paper])[0], date_val
    return None, None
