#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
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
        raise ValueError("No JSON object found in LLM output.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM output JSON is not an object.")
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
        raise ValueError(f"LLM output missing tag keys: {', '.join(missing)}")
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


@dataclass(frozen=True)
class _ApiConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: int
    max_tokens: Optional[int] = None
    temperature: float = 0.0


def _env_int_opt(name: str) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_float_opt(name: str) -> Optional[float]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _normalize_openai_compat_base_url(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""

    value = os.path.expandvars(value).strip()
    parsed = urllib.parse.urlparse(value)
    if not parsed.scheme:
        value = f"https://{value}"
        parsed = urllib.parse.urlparse(value)

    # Most OpenAI-compatible endpoints live under /v1; append if caller passed only the host.
    path = (parsed.path or "").rstrip("/")
    if not path:
        value = value.rstrip("/") + "/v1"
    return value.rstrip("/")


def _load_api_config_from_env(
    *,
    model: Optional[str],
    timeout_s: int,
    base_url: Optional[str] = None,
) -> _ApiConfig:
    api_key = (os.getenv("LLM_API_KEY") or "").strip() or (os.getenv("KIMI_API_KEY") or "").strip()
    if api_key:
        api_key = os.path.expandvars(api_key).strip()
    if not api_key:
        raise ValueError("Missing LLM_API_KEY (or legacy KIMI_API_KEY) in environment for backend=api.")

    raw_base = (base_url or os.getenv("LLM_BASE_URL") or os.getenv("KIMI_BASE_URL") or "").strip()
    base = _normalize_openai_compat_base_url(raw_base)
    if not base:
        raise ValueError("Missing LLM_BASE_URL (or legacy KIMI_BASE_URL) in environment for backend=api.")

    model_name = (
        (model or "").strip()
        or (os.getenv("LLM_MODEL_NAME") or "").strip()
        or (os.getenv("LLM_MODEL") or "").strip()
        or (os.getenv("KIMI_MODEL") or "").strip()
    )
    model_name = os.path.expandvars(model_name).strip()
    if not model_name:
        raise ValueError(
            "Missing LLM_MODEL_NAME (or LLM_MODEL / legacy KIMI_MODEL) in environment for backend=api."
        )

    max_tokens = _env_int_opt("LLM_MAX_OUTPUT_TOKENS") or _env_int_opt("KIMI_MAX_OUTPUT_TOKENS")
    temperature = _env_float_opt("LLM_TEMPERATURE") or _env_float_opt("KIMI_TEMPERATURE")

    return _ApiConfig(
        base_url=base,
        api_key=api_key,
        model=model_name,
        timeout_s=max(1, int(timeout_s)),
        max_tokens=max_tokens if (max_tokens is None or max_tokens > 0) else None,
        temperature=float(temperature) if temperature is not None else 0.0,
    )


def _openai_compat_chat_completions(prompt: str, *, config: _ApiConfig) -> str:
    url = f"{config.base_url}/chat/completions"
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        payload["max_tokens"] = int(config.max_tokens)

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "CodeArXiv/0.1",
    }

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", "replace").strip()
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {e.code} for {url}: {err_body or e.reason}") from None
    except Exception as e:  # noqa: BLE001 - surface minimal error
        raise RuntimeError(f"Request failed for {url}: {e}") from None

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"API returned non-JSON response: {raw[:2000]!r}") from None

    if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
        err = obj["error"]
        msg = err.get("message") or err.get("type") or str(err)
        raise RuntimeError(f"API error: {msg}")

    if not isinstance(obj, dict):
        raise RuntimeError("API response is not a JSON object.")
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("API response missing choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("API response choices[0] is not an object.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("API response missing choices[0].message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("API response missing choices[0].message.content.")
    return content


def _llm_exec(
    *,
    backend: str,
    prompt: str,
    model: Optional[str],
    reasoning_effort: str,
    reasoning_summary: str,
    timeout_s: int,
) -> str:
    if backend == "codex":
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

            return tmp_path.read_text(encoding="utf-8", errors="replace")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    if backend in ("api", "kimi"):
        api_cfg = _load_api_config_from_env(model=model, timeout_s=timeout_s)
        return _openai_compat_chat_completions(prompt, config=api_cfg)

    raise ValueError(f"Unknown backend: {backend!r}")


def _codex_translate_and_summarize(
    *,
    title_en: str,
    abstract_en: str,
    backend: str,
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

    raw = _llm_exec(
        backend=backend,
        prompt=prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        timeout_s=timeout_s,
    )

    obj = _extract_json_object(raw)
    title_zh = str(obj.get("title_zh", "")).strip()
    abstract_zh = str(obj.get("abstract_zh", "")).strip()
    summary_zh = str(obj.get("summary_zh", "")).strip()
    summary_en = str(obj.get("summary_en", "")).strip()
    if not (title_zh and abstract_zh and summary_zh and summary_en):
        raise ValueError("LLM output missing required fields.")
    return title_zh, abstract_zh, summary_zh, summary_en


def _codex_translate_and_summarize_batch(
    *,
    items: Dict[str, Dict[str, str]],
    backend: str,
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

    raw = _llm_exec(
        backend=backend,
        prompt=prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        timeout_s=timeout_s,
    )

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


def _codex_tag_batch(
    *,
    items: Dict[str, Dict[str, str]],
    tag_prompt: str,
    backend: str,
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

    raw = _llm_exec(
        backend=backend,
        prompt=prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        timeout_s=timeout_s,
    )

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


def main() -> int:
    load_dotenv()
    raw_backend = (os.getenv("LLM_PROVIDER") or os.getenv("CODEX_BACKEND") or "").strip().lower()
    default_backend = raw_backend if raw_backend in ("codex", "api", "kimi") else "codex"
    default_batch_size = max(1, _env_int("CODEX_BATCH_SIZE", 5))
    default_timeout = max(1, _env_int("CODEX_TIMEOUT", 300))
    default_sleep = max(0.0, _env_float("CODEX_SLEEP", 0.2))
    default_overwrite = _env_bool("CODEX_OVERWRITE", False)
    default_reasoning_effort = (os.getenv("CODEX_REASONING_EFFORT") or "").strip() or "low"
    default_reasoning_summary = (os.getenv("CODEX_REASONING_SUMMARY") or "").strip() or "concise"

    parser = argparse.ArgumentParser(
        description=(
            "Fill title_zh/abstract_zh/summary_zh/summary_en and tags in results JSON via an LLM (Codex CLI or OpenAI-compatible API)."
        )
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=default_backend,
        choices=("codex", "api", "kimi"),
        help="LLM backend: codex (Codex CLI) or api (OpenAI-compatible HTTP API). "
        "Note: kimi is kept as an alias of api for compatibility. "
        "Default: LLM_PROVIDER or CODEX_BACKEND or codex.",
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
        help="Batch size per LLM request (1 = per-paper).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. codex: defaults to CODEX_MODEL (or Codex CLI default). "
        "api/kimi: defaults to LLM_MODEL_NAME (or LLM_MODEL / legacy KIMI_MODEL).",
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

    backend = str(args.backend or "").strip().lower() or "codex"
    model: Optional[str] = (str(args.model).strip() if args.model else None) or None
    if backend == "codex" and not model:
        model = _env_str("CODEX_MODEL")
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
                    backend=backend,
                    model=model,
                    reasoning_effort=str(args.reasoning_effort),
                    reasoning_summary=str(args.reasoning_summary),
                    timeout_s=args.timeout,
                )
                for arxiv_id in items.keys():
                    entry = data.get(arxiv_id) or {}
                    out = out_map.get(arxiv_id)
                    if not out:
                        entry.setdefault("errors", {})["codex_fill_zh"] = "Missing arxiv_id in LLM output."
                        data[arxiv_id] = entry
                        continue

                    changed = False
                    updated_by_llm = False

                    if args.overwrite or not (entry.get("title_zh") or "").strip():
                        entry["title_zh"] = out["title_zh"]
                        changed = True
                        updated_by_llm = True
                    if args.overwrite or not (entry.get("abstract_zh") or "").strip():
                        entry["abstract_zh"] = out["abstract_zh"]
                        changed = True
                        updated_by_llm = True
                    if args.overwrite or not (entry.get("summary_zh") or "").strip():
                        entry["summary_zh"] = out["summary_zh"]
                        changed = True
                        updated_by_llm = True
                    if args.overwrite or not (entry.get("summary_en") or "").strip():
                        entry["summary_en"] = out["summary_en"]
                        changed = True

                    if entry.get("title_zh") and entry.get("abstract_zh") and entry.get("translated") is not True:
                        entry["translated"] = True
                        changed = True
                    if entry.get("summary_zh") and entry.get("summary_generated") is not True:
                        entry["summary_generated"] = True
                        changed = True

                    backend_name = "codex" if backend == "codex" else backend
                    if updated_by_llm and entry.get("translation_backend") != backend_name:
                        entry["translation_backend"] = backend_name
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
                        backend=backend,
                        model=model,
                        reasoning_effort=str(args.reasoning_effort),
                        reasoning_summary=str(args.reasoning_summary),
                        timeout_s=args.timeout,
                    )
                    for arxiv_id in tag_items.keys():
                        entry = data.get(arxiv_id) or {}
                        out = out_map.get(arxiv_id)
                        if not out:
                            entry.setdefault("errors", {})["codex_tag"] = "Missing arxiv_id in LLM output."
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
