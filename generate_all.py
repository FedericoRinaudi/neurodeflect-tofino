#!/usr/bin/env python3
"""
generate_all.py

Runs catboost_to_tofino.py for every JSON in jsons/catboost/
and xgboost_to_tofino.py for every JSON in jsons/xgboost/.

All resulting D{N}_CATBOOST / D{N}_XGBOOST folders are placed
inside a single output directory (default: "generated/").

Usage
-----
  python3 generate_all.py
  python3 generate_all.py --out my_output --merge-strategy greedy
  python3 generate_all.py --light
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(script: Path, model_json: Path, out_dir: Path, merge_strategy: str, light: bool) -> bool:
    cmd = [
        sys.executable, str(script),
        str(model_json),
        "--mode", "combined",
        "--merge-strategy", merge_strategy,
    ]
    if light:
        cmd.append("--light")
    print(f"\n{'='*60}")
    print(f"  {script.name}  ←  {model_json.name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=out_dir)
    return result.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all Tofino output folders")
    ap.add_argument("--out", default="generated", help="Output directory (default: generated/)")
    ap.add_argument("--merge-strategy", choices=["greedy", "optimal"], default="greedy")
    ap.add_argument("--light", action="store_true",
                    help="Pick smaller raw side before merging, skip merging the larger side")
    args = ap.parse_args()

    root       = Path(__file__).parent
    out_dir    = root / args.out
    jsons_root = root / "jsons"

    out_dir.mkdir(exist_ok=True)

    tasks = [
        (root / "catboost_to_tofino.py", jsons_root / "catboost"),
        (root / "xgboost_to_tofino.py",  jsons_root / "xgboost"),
    ]

    ok = failed = 0
    for script, json_dir in tasks:
        for json_file in sorted(json_dir.glob("*.json")):
            if run(script, json_file, out_dir, args.merge_strategy, args.light):
                ok += 1
            else:
                failed += 1
                print(f"  ERROR: {json_file.name}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"  Done: {ok} succeeded, {failed} failed")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
