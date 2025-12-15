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


def _data_dir() -> Path:
    base_cfg = current_app.config.get("PAPERS_DATA_DIR", "CodeArXiv-data")
    base = Path(base_cfg).expanduser()
    if not base.is_absolute():
        base = (Path(current_app.root_path).parent / base).resolve()
    else:
        base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


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


def _parse_embedding(raw: Any):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            return raw
    return raw


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
    if not paper.get("image_path"):
        paper["image_path"] = (
            paper.get("thumbnail_300_path")
            or paper.get("thumbnail_path")
            or paper.get("thumbnail_small_path")
            or paper.get("thumbnail_100_path")
        )
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
    paper["embedding"] = _parse_embedding(paper.get("embedding"))
    if paper.get("pub_date"):
        paper["pub_date"] = str(paper["pub_date"])
    if paper.get("created_at"):
        paper["created_at"] = str(paper["created_at"])
    return paper


def load_date(date_val: Union[dt.date, str]) -> List[Dict[str, Any]]:
    path = _date_path(date_val)
    raw = _load_raw(path)
    return [_normalize_paper(item) for item in raw if isinstance(item, dict)]


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
    existing = {p["id"]: p for p in load_date(date_val)}
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
        items = load_date(date_val)
        for paper in items:
            if paper.get("id") == pid:
                return paper, date_val
    return None, None
