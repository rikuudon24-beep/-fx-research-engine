#!/usr/bin/env python3
"""Apply only allow-listed, deterministic repairs to known FX engine failures."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_log(path: pathlib.Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def fix_known_import_bug(log: str) -> bool:
    target = ROOT / "scripts" / "repair_market_gaps.py"
    text = target.read_text()
    if "ModuleNotFoundError: No module named 'scripts'" not in log:
        return False
    old = "from scripts.native_candles import download_native_candles"
    new = "from native_candles import download_native_candles"
    if old not in text:
        return False
    target.write_text(text.replace(old, new, 1))
    print("[AUTO-FIX] repaired native_candles import path")
    return True


def verify() -> None:
    targets = [
        ROOT / "scripts" / "repair_market_gaps.py",
        ROOT / "scripts" / "native_candles.py",
    ]
    for target in targets:
        subprocess.run([sys.executable, "-m", "py_compile", str(target)], check=True)
    print("[AUTO-FIX] Python syntax verification passed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    log = read_log(pathlib.Path(args.log))
    changed = fix_known_import_bug(log)

    if not changed:
        print("[AUTO-FIX] no allow-listed fix matched")
        return 2

    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
