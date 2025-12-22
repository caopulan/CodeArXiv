#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv


RESULTS_DIR = Path("results")
TAG_PROMPT_PATH = Path(__file__).resolve().parent / "tag_prompt.md"
TAG_KEYS = ("task", "method", "property", "special")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_dump_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _pick_latest_results_path(results_dir: Path) -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"{results_dir} does not exist")

    candidates = []
    for p in results_dir.glob("*.json"):
        m = re.fullmatch(r"(\\d{4}-\\d{2}-\\d{2})\\.json", p.name)
        if m:
            candidates.append((m.group(1), p))
    if not candidates:
        raise FileNotFoundError(f"No YYYY-MM-DD.json found under {results_dir}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Best-effort fallback: extract the first {...} block.
    m = re.search(r"\\{.*\\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in Codex output.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Codex output JSON is not an object.")
    return obj


def _load_tag_prompt(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"Tag prompt is empty: {path}")
    return content.strip()


def _normalize_tag_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except Exception:
            raw = [t.strip() for t in raw.split(",") if t.strip()]
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    return [str(t).strip() for t in raw if str(t).strip()]


def _parse_tag_payload(payload: Dict[str, Any]) -> Dict[str, list[str]]:
    missing = [k for k in TAG_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Codex output missing tag keys: {', '.join(missing)}")
    return {k: _normalize_tag_list(payload.get(k)) for k in TAG_KEYS}


def _flatten_tag_payload(tag_payload: Dict[str, list[str]]) -> list[str]:
    seen = set()
    merged: list[str] = []
    for key in TAG_KEYS:
        for tag in tag_payload.get(key, []):
            if tag not in seen:
                seen.add(tag)
                merged.append(tag)
    return merged


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


def _build_codex_exec_cmd(
    *,
    model: Optional[str],
    reasoning_effort: str,
    reasoning_summary: str,
    tmp_path: Path,
) -> list[str]:
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--disable",
        "shell_tool",
        "--color",
        "never",
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        f'model_reasoning_summary="{reasoning_summary}"',
        "--output-last-message",
        str(tmp_path),
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append("-")
    return cmd


def _codex_translate_and_summarize(
    *,
    title_en: str,
    abstract_en: str,
    model: Optional[str],
    reasoning_effort: str,
    reasoning_summary: str,
    timeout_s: int,
) -> Tuple[str, str, str, str]:
    prompt = (
        "你是一个学术论文助手。请严格只输出 JSON（不要输出解释/Markdown/代码块）。\\n"
        "JSON 必须包含四个字段：title_zh, abstract_zh, summary_zh, summary_en。\\n"
        "- title_zh：把 title_en 翻译成中文，保留必要的术语/缩写/方法名（可在中文中保留英文括号）。\\n"
        "- abstract_zh：把 abstract_en 尽量忠实完整地翻译成中文，保持一段或多段原有结构，不要遗漏关键信息。\\n"
        "- summary_zh：用一句中文概述论文做了什么/贡献是什么（1 句，尽量以“提出/实现/构建/统一/证明/系统化”等动词开头，末尾用“。”）。\\n"
        "- summary_en：用一句英文概述论文做了什么/贡献是什么（1 sentence, English only, end with a period).\\n"
        "\\n"
        "输入：\\n"
        f"title_en: {title_en.strip()}\\n"
        f"abstract_en: {abstract_en.strip()}\\n"
        "\\n"
        "输出 JSON 示例：\\n"
        "{\"title_zh\":\"...\",\"abstract_zh\":\"...\",\"summary_zh\":\"...\",\"summary_en\":\"...\"}\\n"
    )

    tmp_path = Path(f".codex_last_message.{os.getpid()}.txt")
    try:
        cmd = _build_codex_exec_cmd(
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            tmp_path=tmp_path,
        )

        subprocess.run(
            cmd,
            input=prompt,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )

        raw = tmp_path.read_text(encoding="utf-8", errors="replace")
        obj = _extract_json_object(raw)
        title_zh = str(obj.get("title_zh", "")).strip()
        abstract_zh = str(obj.get("abstract_zh", "")).strip()
        summary_zh = str(obj.get("summary_zh", "")).strip()
        summary_en = str(obj.get("summary_en", "")).strip()
        if not (title_zh and abstract_zh and summary_zh and summary_en):
            raise ValueError("Codex output missing required fields.")
        return title_zh, abstract_zh, summary_zh, summary_en
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def _codex_translate_and_summarize_batch(
    *,
    items: Dict[str, Dict[str, str]],
    model: Optional[str],
    reasoning_effort: str,
    reasoning_summary: str,
    timeout_s: int,
) -> Dict[str, Dict[str, str]]:
    """
    items: {arxiv_id: {"title_en": "...", "abstract_en": "..."}}
    returns: {arxiv_id: {"title_zh": "...", "abstract_zh": "...", "summary_zh": "...", "summary_en": "..."}}
    """
    payload = []
    for arxiv_id, v in items.items():
        payload.append(
            {
                "arxiv_id": arxiv_id,
                "title_en": (v.get("title_en") or "").strip(),
                "abstract_en": (v.get("abstract_en") or "").strip(),
            }
        )

    prompt = (
        "你是一个学术论文助手。请严格只输出 JSON（不要输出解释/Markdown/代码块）。\\n"
        "输出必须是一个 JSON 对象：key 为 arxiv_id，value 为包含 title_zh, abstract_zh, summary_zh, summary_en 的对象。\\n"
        "- title_zh：把 title_en 翻译成中文，保留必要的术语/缩写/方法名（可在中文中保留英文括号）。\\n"
        "- abstract_zh：把 abstract_en 尽量忠实完整地翻译成中文，保持原有段落结构，不要遗漏关键信息。\\n"
        "- summary_zh：用一句中文概述论文做了什么/贡献是什么（1 句，尽量以“提出/实现/构建/统一/证明/系统化”等动词开头，末尾用“。”）。\\n"
        "- summary_en：用一句英文概述论文做了什么/贡献是什么（1 sentence, English only, end with a period).\\n"
        "\\n"
        "输入 JSON：\\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\\n"
        "\\n"
        "输出 JSON 示例：\\n"
        "{\"2512.00001\":{\"title_zh\":\"...\",\"abstract_zh\":\"...\",\"summary_zh\":\"...\",\"summary_en\":\"...\"}}\\n"
    )

    tmp_path = Path(f".codex_last_message.{os.getpid()}.txt")
    try:
        cmd = _build_codex_exec_cmd(
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            tmp_path=tmp_path,
        )

        subprocess.run(
            cmd,
            input=prompt,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )

        raw = tmp_path.read_text(encoding="utf-8", errors="replace")
        obj = _extract_json_object(raw)
        out: Dict[str, Dict[str, str]] = {}
        for arxiv_id in items.keys():
            v = obj.get(arxiv_id)
            if not isinstance(v, dict):
                continue
            title_zh = str(v.get("title_zh", "")).strip()
            abstract_zh = str(v.get("abstract_zh", "")).strip()
            summary_zh = str(v.get("summary_zh", "")).strip()
            summary_en = str(v.get("summary_en", "")).strip()
            if title_zh and abstract_zh and summary_zh and summary_en:
                out[arxiv_id] = {
                    "title_zh": title_zh,
                    "abstract_zh": abstract_zh,
                    "summary_zh": summary_zh,
                    "summary_en": summary_en,
                }
        return out
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def _codex_tag_batch(
    *,
    items: Dict[str, Dict[str, str]],
    tag_prompt: str,
    model: Optional[str],
    reasoning_effort: str,
    reasoning_summary: str,
    timeout_s: int,
) -> Dict[str, Dict[str, list[str]]]:
    payload = []
    for arxiv_id, v in items.items():
        payload.append(
            {
                "arxiv_id": arxiv_id,
                "title": (v.get("title") or "").strip(),
                "abstract": (v.get("abstract") or "").strip(),
            }
        )

    prompt = (
        f"{tag_prompt.strip()}\n\n"
        "你将收到一个 JSON 数组，每个元素包含 arxiv_id、title、abstract（对应 Title/Abstract）。\n"
        "请对每个元素按照上面的标签体系打标，并输出一个 JSON 对象：key 为 arxiv_id，"
        "value 为包含 task/method/property/special 的对象。\n"
        "输出必须是严格 JSON，不要包含解释或 Markdown。\n"
        "\n"
        "输入 JSON：\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\n"
        "输出 JSON 示例：\n"
        "{\"2512.00001\":{\"task\":[],\"method\":[],\"property\":[],\"special\":[]}}\n"
    )

    tmp_path = Path(f".codex_last_message.{os.getpid()}.txt")
    try:
        cmd = _build_codex_exec_cmd(
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            tmp_path=tmp_path,
        )

        subprocess.run(
            cmd,
            input=prompt,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )

        raw = tmp_path.read_text(encoding="utf-8", errors="replace")
        obj = _extract_json_object(raw)
        out: Dict[str, Dict[str, list[str]]] = {}

        if len(items) == 1 and all(k in obj for k in TAG_KEYS):
            only_id = next(iter(items.keys()))
            out[only_id] = _parse_tag_payload(obj)
            return out

        for arxiv_id in items.keys():
            v = obj.get(arxiv_id)
            if not isinstance(v, dict):
                continue
            out[arxiv_id] = _parse_tag_payload(v)
        return out
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def main() -> int:
    load_dotenv()
    default_model = _env_str("CODEX_MODEL")
    default_batch_size = max(1, _env_int("CODEX_BATCH_SIZE", 5))
    default_timeout = max(1, _env_int("CODEX_TIMEOUT", 300))
    default_sleep = max(0.0, _env_float("CODEX_SLEEP", 0.2))
    default_overwrite = _env_bool("CODEX_OVERWRITE", False)
    default_reasoning_effort = (os.getenv("CODEX_REASONING_EFFORT") or "").strip() or "low"
    default_reasoning_summary = (os.getenv("CODEX_REASONING_SUMMARY") or "").strip() or "concise"

    parser = argparse.ArgumentParser(
        description=(
            "Fill title_zh/abstract_zh/summary_zh/summary_en and tags in results JSON via Codex (LLM)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to results/<date>.json (default: latest under results/).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=default_overwrite,
        help="Overwrite existing translated/summary/tag fields even if present.",
    )
    parser.add_argument(
        "--skip-tags",
        action="store_true",
        help="Skip tag generation via tag_prompt.md.",
    )
    parser.add_argument(
        "--tag-prompt",
        type=Path,
        default=TAG_PROMPT_PATH,
        help="Path to tag_prompt.md.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="For testing: only process first N papers (0 = all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help="Batch size per Codex request (1 = per-paper).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Codex model name (default: Codex CLI default).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=default_reasoning_effort,
        help="Codex reasoning effort (default: CODEX_REASONING_EFFORT or low).",
    )
    parser.add_argument(
        "--reasoning-summary",
        type=str,
        default=default_reasoning_summary,
        help="Codex reasoning summary (default: CODEX_REASONING_SUMMARY or concise).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=default_sleep,
        help="Sleep seconds between papers to avoid rate limits.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=default_timeout,
        help="Timeout seconds per Codex request.",
    )
    args = parser.parse_args()

    input_path: Path = args.input or _pick_latest_results_path(RESULTS_DIR)
    data: Dict[str, Dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
    tag_prompt = None if args.skip_tags else _load_tag_prompt(args.tag_prompt)

    arxiv_ids = sorted(data.keys())
    if args.limit and args.limit > 0:
        arxiv_ids = arxiv_ids[: args.limit]

    total = len(arxiv_ids)
    batch_size = max(1, int(args.batch_size))
    processed = 0
    processed_ids: set[str] = set()
    idx = 0
    while idx < total:
        batch_ids = arxiv_ids[idx : idx + batch_size]
        idx_end = idx + len(batch_ids)

        items: Dict[str, Dict[str, str]] = {}
        for arxiv_id in batch_ids:
            entry = data.get(arxiv_id) or {}
            title_en = (entry.get("title_en") or "").strip()
            abstract_en = (entry.get("abstract_en") or "").strip()
            if not title_en or not abstract_en:
                entry.setdefault("errors", {})["codex_fill_zh"] = "Missing title_en or abstract_en."
                data[arxiv_id] = entry
                continue

            have_title_zh = bool((entry.get("title_zh") or "").strip())
            have_abstract_zh = bool((entry.get("abstract_zh") or "").strip())
            have_summary_zh = bool((entry.get("summary_zh") or "").strip())
            have_summary_en = bool((entry.get("summary_en") or "").strip())
            already_done = have_title_zh and have_abstract_zh and have_summary_zh and have_summary_en
            if already_done and not args.overwrite:
                continue

            items[arxiv_id] = {"title_en": title_en, "abstract_en": abstract_en}

        if items:
            try:
                out_map = _codex_translate_and_summarize_batch(
                    items=items,
                    model=args.model,
                    reasoning_effort=str(args.reasoning_effort),
                    reasoning_summary=str(args.reasoning_summary),
                    timeout_s=args.timeout,
                )
                for arxiv_id in items.keys():
                    entry = data.get(arxiv_id) or {}
                    out = out_map.get(arxiv_id)
                    if not out:
                        entry.setdefault("errors", {})["codex_fill_zh"] = "Missing arxiv_id in Codex output."
                        data[arxiv_id] = entry
                        continue

                    changed = False
                    updated_by_codex = False

                    if args.overwrite or not (entry.get("title_zh") or "").strip():
                        entry["title_zh"] = out["title_zh"]
                        changed = True
                        updated_by_codex = True
                    if args.overwrite or not (entry.get("abstract_zh") or "").strip():
                        entry["abstract_zh"] = out["abstract_zh"]
                        changed = True
                        updated_by_codex = True
                    if args.overwrite or not (entry.get("summary_zh") or "").strip():
                        entry["summary_zh"] = out["summary_zh"]
                        changed = True
                        updated_by_codex = True
                    if args.overwrite or not (entry.get("summary_en") or "").strip():
                        entry["summary_en"] = out["summary_en"]
                        changed = True

                    if entry.get("title_zh") and entry.get("abstract_zh") and entry.get("translated") is not True:
                        entry["translated"] = True
                        changed = True
                    if entry.get("summary_zh") and entry.get("summary_generated") is not True:
                        entry["summary_generated"] = True
                        changed = True

                    if updated_by_codex and entry.get("translation_backend") != "codex":
                        entry["translation_backend"] = "codex"
                        changed = True

                    entry.setdefault("errors", {}).pop("codex_fill_zh", None)
                    if changed:
                        entry["updated_at"] = _now_iso()
                        if arxiv_id not in processed_ids:
                            processed_ids.add(arxiv_id)
                            processed += 1
                    data[arxiv_id] = entry
            except Exception as e:  # noqa: BLE001
                for arxiv_id in items.keys():
                    entry = data.get(arxiv_id) or {}
                    entry.setdefault("errors", {})["codex_fill_zh"] = str(e)
                    data[arxiv_id] = entry

            _json_dump_atomic(data, input_path)

        if tag_prompt:
            tag_items: Dict[str, Dict[str, str]] = {}
            for arxiv_id in batch_ids:
                entry = data.get(arxiv_id) or {}
                existing_tags = _normalize_tag_list(entry.get("tags"))
                if existing_tags and not args.overwrite:
                    continue
                if entry.get("tags_generated") is True and not args.overwrite:
                    continue

                title = (entry.get("title_en") or entry.get("title_zh") or "").strip()
                abstract = (entry.get("abstract_en") or entry.get("abstract_zh") or "").strip()
                if not title or not abstract:
                    entry.setdefault("errors", {})["codex_tag"] = "Missing title or abstract."
                    data[arxiv_id] = entry
                    continue

                tag_items[arxiv_id] = {"title": title, "abstract": abstract}

            if tag_items:
                try:
                    out_map = _codex_tag_batch(
                        items=tag_items,
                        tag_prompt=tag_prompt,
                        model=args.model,
                        reasoning_effort=str(args.reasoning_effort),
                        reasoning_summary=str(args.reasoning_summary),
                        timeout_s=args.timeout,
                    )
                    for arxiv_id in tag_items.keys():
                        entry = data.get(arxiv_id) or {}
                        out = out_map.get(arxiv_id)
                        if not out:
                            entry.setdefault("errors", {})["codex_tag"] = "Missing arxiv_id in Codex output."
                            data[arxiv_id] = entry
                            continue

                        merged_tags = _flatten_tag_payload(out)
                        existing_tags = _normalize_tag_list(entry.get("tags"))
                        changed = False
                        if args.overwrite or not existing_tags:
                            if merged_tags != existing_tags:
                                entry["tags"] = merged_tags
                                changed = True
                        if entry.get("tags_generated") is not True:
                            entry["tags_generated"] = True
                            changed = True

                        entry.setdefault("errors", {}).pop("codex_tag", None)
                        if changed:
                            entry["updated_at"] = _now_iso()
                            if arxiv_id not in processed_ids:
                                processed_ids.add(arxiv_id)
                                processed += 1
                        data[arxiv_id] = entry
                except Exception as e:  # noqa: BLE001
                    for arxiv_id in tag_items.keys():
                        entry = data.get(arxiv_id) or {}
                        entry.setdefault("errors", {})["codex_tag"] = str(e)
                        data[arxiv_id] = entry

                _json_dump_atomic(data, input_path)

        print(f"[codex_fill_zh] {idx_end}/{total} (updated {processed}) -> {input_path}")
        idx = idx_end
        time.sleep(max(0.0, float(args.sleep)))

    print(f"[codex_fill_zh] Done. Updated {processed}/{total} -> {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
