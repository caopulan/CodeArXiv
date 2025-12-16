#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


DEFAULT_INTERVAL_HOURS = 1.0
DEFAULT_RETRY_MINUTES = 30.0
DEFAULT_THUMB_WORKERS = 10

_stop_requested = False


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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _on_signal(signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True
    print(f"[server_daily] {_now_iso()} received signal {signum}; will stop after current step.", flush=True)


def _sleep_interruptible(seconds: float) -> None:
    deadline = time.time() + max(0.0, seconds)
    while not _stop_requested:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def _run_cmd(cmd: List[str], *, cwd: Path, timeout_s: Optional[int] = None) -> int:
    print(f"[server_daily] {_now_iso()} $ {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd), timeout=timeout_s)
        return 0
    except subprocess.TimeoutExpired:
        print(f"[server_daily] {_now_iso()} ERROR timeout after {timeout_s}s", file=sys.stderr, flush=True)
        return 124
    except subprocess.CalledProcessError as e:
        print(
            f"[server_daily] {_now_iso()} ERROR exit code {e.returncode}: {' '.join(cmd)}",
            file=sys.stderr,
            flush=True,
        )
        return int(e.returncode)
    except FileNotFoundError as e:
        print(f"[server_daily] {_now_iso()} ERROR {e}", file=sys.stderr, flush=True)
        return 127


def _run_one_cycle(args: argparse.Namespace) -> bool:
    cwd = Path(args.workdir).resolve()
    run_daily_path = Path(args.run_daily).resolve()
    codex_fill_path = Path(args.codex_fill).resolve()

    ok = True
    cmd = [sys.executable, str(run_daily_path), "--thumb-workers", str(int(args.thumb_workers))]
    if args.thumb_overwrite:
        cmd.append("--thumb-overwrite")
    if args.skip_codex:
        cmd.append("--skip-codex")
    cmd += ["--codex-fill", str(codex_fill_path)]
    if args.codex_model:
        cmd += ["--codex-model", str(args.codex_model)]
    if args.codex_batch_size:
        cmd += ["--codex-batch-size", str(int(args.codex_batch_size))]
    if args.codex_timeout:
        cmd += ["--codex-timeout", str(int(args.codex_timeout))]
    if args.codex_sleep is not None:
        cmd += ["--codex-sleep", str(float(args.codex_sleep))]
    if args.codex_overwrite:
        cmd += ["--codex-overwrite"]

    rc = _run_cmd(cmd, cwd=cwd)
    ok = ok and (rc == 0)

    return ok


def main() -> int:
    load_dotenv()
    default_codex_model = _env_str("CODEX_MODEL")
    default_codex_batch_size = max(1, _env_int("CODEX_BATCH_SIZE", 5))
    default_codex_timeout = max(1, _env_int("CODEX_TIMEOUT", 300))
    default_codex_sleep = max(0.0, _env_float("CODEX_SLEEP", 0.2))
    default_codex_overwrite = _env_bool("CODEX_OVERWRITE", False)

    parser = argparse.ArgumentParser(
        description="Long-running daily updater: run run_daily.py on a fixed interval."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repo working directory (default: this file's directory).",
    )
    parser.add_argument(
        "--run-daily",
        type=Path,
        default=Path(__file__).resolve().parent / "run_daily.py",
        help="Path to run_daily.py.",
    )
    parser.add_argument(
        "--codex-fill",
        type=Path,
        default=Path(__file__).resolve().parent / "codex_fill_zh.py",
        help="Path to codex_fill_zh.py (used by run_daily.py).",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=DEFAULT_INTERVAL_HOURS,
        help=f"Run interval in hours (default: {DEFAULT_INTERVAL_HOURS}).",
    )
    parser.add_argument(
        "--retry-minutes",
        type=float,
        default=DEFAULT_RETRY_MINUTES,
        help=f"Retry delay in minutes after failure (default: {DEFAULT_RETRY_MINUTES}).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (for testing / cron).",
    )
    parser.add_argument(
        "--skip-codex",
        action="store_true",
        help="Skip running codex fill (only run list/metadata + thumbnails).",
    )

    parser.add_argument(
        "--thumb-workers",
        type=int,
        default=DEFAULT_THUMB_WORKERS,
        help=f"Forwarded to run_daily.py --thumb-workers (default: {DEFAULT_THUMB_WORKERS}).",
    )
    parser.add_argument(
        "--thumb-overwrite",
        action="store_true",
        help="Forwarded to run_daily.py --thumb-overwrite.",
    )

    parser.add_argument(
        "--codex-model",
        type=str,
        default=default_codex_model,
        help="Forwarded to run_daily.py --codex-model (default: CODEX_MODEL).",
    )
    parser.add_argument(
        "--codex-batch-size",
        type=int,
        default=default_codex_batch_size,
        help="Forwarded to run_daily.py --codex-batch-size (default: CODEX_BATCH_SIZE).",
    )
    parser.add_argument(
        "--codex-timeout",
        type=int,
        default=default_codex_timeout,
        help="Forwarded to run_daily.py --codex-timeout (default: CODEX_TIMEOUT).",
    )
    parser.add_argument(
        "--codex-sleep",
        type=float,
        default=default_codex_sleep,
        help="Forwarded to run_daily.py --codex-sleep (default: CODEX_SLEEP).",
    )
    parser.add_argument(
        "--codex-overwrite",
        action=argparse.BooleanOptionalAction,
        default=default_codex_overwrite,
        help="Forwarded to run_daily.py --codex-overwrite.",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    workdir = Path(args.workdir).resolve()
    os.chdir(workdir)

    while not _stop_requested:
        print(f"[server_daily] {_now_iso()} cycle start", flush=True)
        ok = _run_one_cycle(args)
        print(f"[server_daily] {_now_iso()} cycle done ok={ok}", flush=True)

        if args.once:
            return 0 if ok else 1

        if _stop_requested:
            break

        if ok:
            sleep_s = max(60.0, float(args.interval_hours) * 3600.0)
        else:
            sleep_s = max(60.0, float(args.retry_minutes) * 60.0)

        print(f"[server_daily] {_now_iso()} sleep {int(sleep_s)}s", flush=True)
        _sleep_interruptible(sleep_s)

    print(f"[server_daily] {_now_iso()} stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
