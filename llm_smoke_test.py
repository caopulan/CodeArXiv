#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

import codex_fill_zh as fill


DEFAULT_SAMPLE_TITLE = "A Simple Test Title"
DEFAULT_SAMPLE_ABSTRACT = (
    "We present a small test abstract to verify JSON-only responses from the model. "
    "The system should return valid JSON with the required keys."
)


def _env_str(name: str) -> Optional[str]:
    raw = (os.getenv(name) or "").strip()
    return raw or None


def _load_entry_from_json(path: Path, *, arxiv_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Input JSON must be an object: {path}")
    if arxiv_id:
        entry = obj.get(arxiv_id)
        if not isinstance(entry, dict):
            raise KeyError(f"arxiv_id not found or not an object: {arxiv_id}")
        return arxiv_id, entry
    # Pick first deterministic key for convenience.
    for k in sorted(obj.keys()):
        v = obj.get(k)
        if isinstance(v, dict):
            return str(k), v
    raise ValueError(f"No object entries found in: {path}")


def _build_translate_prompt(*, title_en: str, abstract_en: str) -> str:
    return (
        "你是一个学术论文助手。请严格只输出 JSON（不要输出解释/Markdown/代码块）。\n"
        "JSON 必须包含四个字段：title_zh, abstract_zh, summary_zh, summary_en。\n"
        "- title_zh：把 title_en 翻译成中文，保留必要的术语/缩写/方法名（可在中文中保留英文括号）。\n"
        "- abstract_zh：把 abstract_en 尽量忠实完整地翻译成中文，保持一段或多段原有结构，不要遗漏关键信息。\n"
        "- summary_zh：用一句中文概述论文做了什么/贡献是什么（1 句，尽量以“提出/实现/构建/统一/证明/系统化”等动词开头，末尾用“。”）。\n"
        "- summary_en：用一句英文概述论文做了什么/贡献是什么（1 sentence, English only, end with a period).\n"
        "\n"
        "输入：\n"
        f"title_en: {title_en.strip()}\n"
        f"abstract_en: {abstract_en.strip()}\n"
        "\n"
        "输出 JSON 示例：\n"
        '{"title_zh":"...","abstract_zh":"...","summary_zh":"...","summary_en":"..."}\n'
    )


def _build_tag_prompt(*, arxiv_id: str, title: str, abstract: str, tag_prompt: str) -> str:
    payload = [{"arxiv_id": arxiv_id, "title": title.strip(), "abstract": abstract.strip()}]
    return (
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
        '{"2512.00001":{"task":[],"method":[],"property":[],"special":[]}}\n'
    )


def _print_env_summary() -> None:
    provider = _env_str("LLM_PROVIDER") or _env_str("CODEX_BACKEND") or "codex (default)"
    print(f"[env] LLM_PROVIDER={provider!r}")
    print(f"[env] LLM_BASE_URL present={bool(_env_str('LLM_BASE_URL') or _env_str('KIMI_BASE_URL'))}")
    print(f"[env] LLM_API_KEY present={bool(_env_str('LLM_API_KEY') or _env_str('KIMI_API_KEY'))}")
    print(
        f"[env] LLM_MODEL_NAME={(_env_str('LLM_MODEL_NAME') or _env_str('LLM_MODEL') or _env_str('KIMI_MODEL'))!r}"
    )
    print(f"[env] CODEX_MODEL={_env_str('CODEX_MODEL')!r}")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for CodeArXiv LLM backends: codex (Codex CLI) or api (OpenAI-compatible HTTP API)."
    )
    parser.add_argument("--backend", choices=("codex", "api", "kimi"), default=None, help="LLM backend to test.")
    parser.add_argument("--model", type=str, default=None, help="Override model name for this run.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout seconds for the LLM call.")
    parser.add_argument("--mode", choices=("translate", "tag", "both"), default="translate", help="What to test.")
    parser.add_argument("--input-json", type=Path, default=None, help="Use the first entry from a results JSON file.")
    parser.add_argument("--arxiv-id", type=str, default=None, help="Pick a specific arxiv_id from --input-json.")
    parser.add_argument("--title-en", type=str, default=DEFAULT_SAMPLE_TITLE, help="Sample English title.")
    parser.add_argument("--abstract-en", type=str, default=DEFAULT_SAMPLE_ABSTRACT, help="Sample English abstract.")
    parser.add_argument("--print-prompt", action="store_true", help="Print the prompt.")
    parser.add_argument("--print-raw", action="store_true", help="Print the raw model output.")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="If set, write prompt/raw/parsed JSON into this directory for debugging.",
    )
    args = parser.parse_args(argv)

    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    backend = (str(args.backend or "").strip().lower() or _env_str("LLM_PROVIDER") or _env_str("CODEX_BACKEND") or "codex")
    if backend not in ("codex", "api", "kimi"):
        print(f"[error] Invalid backend {backend!r}; expected codex/api/kimi", file=sys.stderr)
        return 2

    model = (str(args.model).strip() if args.model else None) or None
    timeout_s = max(1, int(args.timeout))

    _print_env_summary()
    print(f"[run] backend={backend!r} model_override={model!r} timeout_s={timeout_s} mode={args.mode!r}")

    arxiv_id = "2512.00001"
    title_en = args.title_en
    abstract_en = args.abstract_en
    title_for_tags = title_en
    abstract_for_tags = abstract_en

    if args.input_json:
        picked_id, entry = _load_entry_from_json(Path(args.input_json), arxiv_id=args.arxiv_id)
        arxiv_id = picked_id
        title_en = str(entry.get("title_en") or entry.get("title") or title_en)
        abstract_en = str(entry.get("abstract_en") or entry.get("abstract") or abstract_en)
        title_for_tags = str(entry.get("title_en") or entry.get("title_zh") or entry.get("title") or title_for_tags)
        abstract_for_tags = str(entry.get("abstract_en") or entry.get("abstract_zh") or entry.get("abstract") or abstract_for_tags)
        print(f"[input] arxiv_id={arxiv_id!r} (from {args.input_json})")

    reasoning_effort = (os.getenv("CODEX_REASONING_EFFORT") or "").strip() or "low"
    reasoning_summary = (os.getenv("CODEX_REASONING_SUMMARY") or "").strip() or "concise"

    def _call_llm(prompt: str) -> Tuple[str, Dict[str, Any]]:
        raw = fill._llm_exec(  # noqa: SLF001 - debug helper uses internal function
            backend=backend,
            prompt=prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            timeout_s=timeout_s,
        )
        parsed = fill._extract_json_object(raw)  # noqa: SLF001 - debug helper uses internal function
        return raw, parsed

    rc = 0

    if args.mode in ("translate", "both"):
        print("\n=== translate ===")
        prompt = _build_translate_prompt(title_en=title_en, abstract_en=abstract_en)
        if args.print_prompt:
            print(prompt)
        if args.dump_dir:
            _write_text(Path(args.dump_dir) / "translate.prompt.txt", prompt)
        try:
            raw, parsed = _call_llm(prompt)
            if args.print_raw:
                print(raw)
            if args.dump_dir:
                _write_text(Path(args.dump_dir) / "translate.raw.txt", raw)
                _write_text(
                    Path(args.dump_dir) / "translate.parsed.json",
                    json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                )
            missing = [k for k in ("title_zh", "abstract_zh", "summary_zh", "summary_en") if not str(parsed.get(k, "")).strip()]
            if missing:
                print(f"[translate] FAIL missing keys: {missing}", file=sys.stderr)
                rc = 1
            else:
                print("[translate] OK")
                print(f"title_zh: {str(parsed.get('title_zh'))[:200]!r}")
                print(f"summary_en: {str(parsed.get('summary_en'))[:200]!r}")
        except Exception as e:  # noqa: BLE001 - debug script
            print(f"[translate] ERROR: {e}", file=sys.stderr)
            rc = 1

    if args.mode in ("tag", "both"):
        print("\n=== tag ===")
        tag_prompt_path = Path(__file__).resolve().parent / "tag_prompt.md"
        tag_prompt = fill._load_tag_prompt(tag_prompt_path)  # noqa: SLF001 - debug helper uses internal function
        prompt = _build_tag_prompt(arxiv_id=arxiv_id, title=title_for_tags, abstract=abstract_for_tags, tag_prompt=tag_prompt)
        if args.print_prompt:
            print(prompt)
        if args.dump_dir:
            _write_text(Path(args.dump_dir) / "tag.prompt.txt", prompt)
        try:
            raw, parsed = _call_llm(prompt)
            if args.print_raw:
                print(raw)
            if args.dump_dir:
                _write_text(Path(args.dump_dir) / "tag.raw.txt", raw)
                _write_text(
                    Path(args.dump_dir) / "tag.parsed.json",
                    json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                )

            # Accept either a top-level payload (single-item shortcut) or {arxiv_id: payload}.
            payload_obj: Optional[Dict[str, Any]] = None
            if isinstance(parsed, dict) and all(k in parsed for k in fill.TAG_KEYS):
                payload_obj = parsed
            else:
                v = parsed.get(arxiv_id) if isinstance(parsed, dict) else None
                if isinstance(v, dict):
                    payload_obj = v

            if not payload_obj:
                print("[tag] FAIL missing payload for arxiv_id", file=sys.stderr)
                rc = 1
            else:
                tag_payload = fill._parse_tag_payload(payload_obj)  # noqa: SLF001 - debug helper uses internal function
                merged = fill._flatten_tag_payload(tag_payload)  # noqa: SLF001 - debug helper uses internal function
                print("[tag] OK")
                print(f"[tag] merged_tags({len(merged)}): {merged}")
        except Exception as e:  # noqa: BLE001 - debug script
            print(f"[tag] ERROR: {e}", file=sys.stderr)
            rc = 1

    if args.dump_dir:
        print(f"\n[dump] wrote debug files under: {Path(args.dump_dir).resolve()}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
