#!/usr/bin/env python3
"""
catboost_to_tofino.py

CatBoost JSON (oblivious trees, binary Logloss) → Tofino bfrt script + P4 spec.

CatBoost oblivious tree structure:
  - depth d splits, ALL nodes at level i share the same split condition
  - 2^d leaves, indexed by a d-bit integer
  - bit j of leaf_index = 1 if feature > splits[j].border, else 0
  - prediction = sigmoid(scale * sum(leaf_values) + bias)
  - deflect if sum(leaf_values) >= 0  (scale=1, bias=0 in default export)

Pipeline (same as gate_dt_to_tofino.py):
  Stage 1: encode_fN tables  (raw float-scaled int → bucket index)
  Stage 2: per-tree OR single combined model_decision table

Floats are scaled to integers here directly (no separate quantization step).

Usage
-----
  # Per-tree:
  python3 catboost_to_tofino.py model.json \\
      --out bfrt.py --p4-spec spec.txt

  # Single combined table (recommended for small models):
  python3 catboost_to_tofino.py model.json --mode combined \\
      --merge-strategy optimal --out bfrt.py --p4-spec spec.txt
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, ".")
from xgboost_to_tofino import (
    EncodingTable,
    FeatureIndex,
    IntThresholds,
    Names,
    TreeRule,
    build_encoding_tables,
    intersect_all_trees,
    merge_rules,
    emit_bfrt,
    emit_bfrt_combined,
    emit_p4_spec,
    emit_p4_spec_combined,
    emit_p4_code_combined,
    emit_headers_p4,
    emit_build_sh,
    _thr_to_bucket_range,
    _DEFINES_P4,
    _PARSER_P4,
)

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(path: str, threshold_scale: int, leaf_scale: int):
    """
    Returns:
      trees            : list of dicts with 'tree_index', 'nodes' in gate_dt format
      names            : {feature_index: feature_name}
      base_score_int   : scaled bias (typically 0)
      int_thresholds   : {fi: sorted list of int thresholds}
    """
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    fi_info = data["features_info"]["float_features"]
    names: Names = {f["flat_feature_index"]: f["feature_name"] for f in fi_info}

    scale, bias = data["scale_and_bias"]
    base_score_int = round(float(bias[0]) * leaf_scale)

    # Global thresholds per feature from features_info
    int_thresholds: IntThresholds = {}
    for f in fi_info:
        fi = f["flat_feature_index"]
        borders_int = sorted({round(b * threshold_scale) for b in f["borders"]})
        int_thresholds[fi] = borders_int

    # Add any borders used in splits but not in features_info (shouldn't happen but be safe)
    for tree_data in data["oblivious_trees"]:
        for sp in tree_data["splits"]:
            fi = sp["float_feature_index"]
            thr = round(sp["border"] * threshold_scale)
            if fi not in int_thresholds:
                int_thresholds[fi] = []
            if thr not in int_thresholds[fi]:
                int_thresholds[fi].append(thr)
                int_thresholds[fi].sort()

    # Convert oblivious trees to list of (tree_index, rules) using gate_dt TreeRule format
    tree_rules: List[Tuple[int, List[TreeRule]]] = []
    for t_idx, tree_data in enumerate(data["oblivious_trees"]):
        rules = _extract_tree_rules(tree_data, int_thresholds, threshold_scale, leaf_scale)
        tree_rules.append((t_idx, rules))

    return tree_rules, names, base_score_int, int_thresholds, scale


def _extract_tree_rules(
    tree_data: dict,
    int_thresholds: IntThresholds,
    threshold_scale: int,
    leaf_scale: int,
) -> List[TreeRule]:
    """
    Extract rules from one CatBoost oblivious tree.

    For each leaf index (0 .. 2^depth - 1):
      bit j of leaf_index = 1  →  feature > splits[j].border  →  op ">="
      bit j of leaf_index = 0  →  feature <= splits[j].border →  op "<"
    """
    splits = tree_data["splits"]
    depth  = len(splits)
    rules: List[TreeRule] = []

    for leaf_idx in range(2 ** depth):
        leaf_val_int = round(tree_data["leaf_values"][leaf_idx] * leaf_scale)

        # Build conditions: one per split level
        conds: List[Tuple[int, str, int]] = []
        for level, sp in enumerate(splits):
            fi  = sp["float_feature_index"]
            thr = round(sp["border"] * threshold_scale)
            op  = ">=" if (leaf_idx >> level) & 1 else "<"
            conds.append((fi, op, thr))

        # Convert conditions to bucket ranges
        ranges: Dict[int, Tuple[int, int]] = {
            fi: (0, len(int_thresholds[fi]))
            for fi in {fi for fi, _, _ in conds}
        }
        valid = True
        for fi, op, thr in conds:
            r = _thr_to_bucket_range(int_thresholds[fi], thr, op)
            if r is None:
                valid = False
                break
            lo = max(ranges[fi][0], r[0])
            hi = min(ranges[fi][1], r[1])
            if lo > hi:
                valid = False
                break
            ranges[fi] = (lo, hi)

        if valid:
            rules.append(TreeRule(ranges, leaf_val_int))

    return rules


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="CatBoost JSON (oblivious trees) → Tofino bfrt script + P4 spec"
    )
    ap.add_argument("model_json")
    ap.add_argument("--out",       default=None,          help="bfrt .py output path")
    ap.add_argument("--p4-spec",   default=None,          help="P4 spec .txt output path")
    ap.add_argument("--program",   default=None,           help="P4 program name (default: d{max_depth})")
    ap.add_argument("--pipe",      default="pipe",        help="Pipeline prefix")
    ap.add_argument("--field-max", type=int, default=0xFFFF,
                    help="Max raw feature field value (default: 65535)")
    ap.add_argument("--threshold-scale", type=int, default=1_000_000,
                    help="Multiply float borders by this (default: 1000000)")
    ap.add_argument("--leaf-scale", type=int, default=100_000,
                    help="Multiply float leaf values by this (default: 100000)")
    ap.add_argument("--mode", choices=["per-tree", "combined"], default="per-tree",
                    help="per-tree: one table per tree; combined: single decision table")
    ap.add_argument("--merge-strategy", choices=["greedy", "optimal"], default="greedy",
                    help="combined mode: greedy (fast) or optimal (QMC+set-cover)")
    ap.add_argument("--light", action="store_true",
                    help="combined mode only — pick smaller raw side before merging, skip merging the larger side")
    ap.add_argument("--p4",        default=None,      help="P4 code .p4 output path (combined mode)")
    ap.add_argument("--key-field", action="append",  default=[], metavar="NAME:FIELD",
                    help="Override P4 key field for an encode table, e.g. occ_vs_min_cand:occ_vs")
    args = ap.parse_args()
    key_field_map = {
        "occ_vs_min_cand":  "occ_vs",
        "occ_vs_mean_cand": "occ_vs",
    }
    for kf in args.key_field:
        name, _, field = kf.partition(":")
        key_field_map[name.strip()] = field.strip()

    tree_rules, names, base_score_int, int_thresholds, scale = load_model(
        args.model_json, args.threshold_scale, args.leaf_scale
    )

    if args.program is None:
        import json
        with open(args.model_json, encoding="utf-8") as _f:
            _data = json.load(_f)
        max_depth = max(len(t["splits"]) for t in _data["oblivious_trees"])
        args.program = f"d{max_depth}"

    depth_str = args.program[1:] if args.program[0] == "d" else args.program
    if args.out is None:
        args.out = f"bfrt{depth_str}.py"
    if args.p4_spec is None:
        args.p4_spec = f"spec{depth_str}.txt"
    if args.p4 is None and args.mode == "combined":
        args.p4 = f"{args.program}.p4"

    print(f"Loaded: {len(tree_rules)} trees  base_score_int={base_score_int}  program={args.program}")
    print(f"threshold_scale={args.threshold_scale}  leaf_scale={args.leaf_scale}  model_scale={scale}")
    print(f"Features: {[names[fi] for fi in sorted(names)]}")
    print(f"Buckets per feature: { {names[fi]: len(v)+1 for fi, v in int_thresholds.items()} }")

    for tidx, rules in tree_rules:
        print(f"  tree {tidx}: {len(rules)} rules ({sum(1 for r in rules if r.leaf_int != 0)} non-zero leaves)")

    encoding_tables = build_encoding_tables(int_thresholds, args.field_max)

    if args.mode == "combined":
        combined = intersect_all_trees(tree_rules, int_thresholds, base_score_int)
        deflect_raw    = [r for r in combined if r.leaf_int >= 0]
        nondflect_raw  = [r for r in combined if r.leaf_int < 0]

        print(f"Combined: {len(combined)} intersections → "
              f"deflect={len(deflect_raw)}  non-deflect={len(nondflect_raw)}")

        if args.light:
            if len(deflect_raw) <= len(nondflect_raw):
                chosen_raw, default_action = deflect_raw, "no_deflect"
            else:
                chosen_raw, default_action = nondflect_raw, "deflect"
            side = "deflect" if default_action == "no_deflect" else "non-deflect"
            print(f"Light: merging {side} side only ({len(chosen_raw)} entries, {args.merge_strategy})")
            chosen_rules = merge_rules(chosen_raw, strategy=args.merge_strategy)
            print(f"→ {len(chosen_rules)} entries after merge, default={default_action}")
        else:
            deflect_merged   = merge_rules(deflect_raw,   strategy=args.merge_strategy)
            nondflect_merged = merge_rules(nondflect_raw, strategy=args.merge_strategy)
            print(f"Merged: deflect={len(deflect_merged)}  non-deflect={len(nondflect_merged)}  ({args.merge_strategy})")

            if len(deflect_merged) <= len(nondflect_merged):
                chosen_rules, default_action = deflect_merged, "no_deflect"
                print(f"→ using deflect rules ({len(chosen_rules)} entries), default=no_deflect")
            else:
                chosen_rules, default_action = nondflect_merged, "deflect"
                print(f"→ using non-deflect rules ({len(chosen_rules)} entries), default=deflect")

        import os
        folder = f"D{depth_str.upper()}_CATBOOST"
        includes_dir = os.path.join(folder, "includes")
        os.makedirs(includes_dir, exist_ok=True)

        p4_path    = os.path.join(folder, f"{args.program}.p4")
        bfrt_path  = os.path.join(folder, "bfrt_shell.py")
        build_path = os.path.join(folder, "build.sh")

        code = emit_p4_code_combined(encoding_tables, chosen_rules, names,
                                     default_action=default_action,
                                     key_field_map=key_field_map)
        with open(p4_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"P4 code     → {p4_path}")

        script = emit_bfrt_combined(encoding_tables, chosen_rules, names,
                                    args.program, args.pipe, default_action,
                                    key_field_map=key_field_map)
        with open(bfrt_path, "w", encoding="utf-8") as f:
            f.write(script)
        print(f"bfrt script → {bfrt_path}")

        with open(build_path, "w", encoding="utf-8") as f:
            f.write(emit_build_sh(args.program))
        print(f"build.sh    → {build_path}")

        with open(os.path.join(includes_dir, "headers.p4"), "w", encoding="utf-8") as f:
            f.write(emit_headers_p4(encoding_tables, names))
        print(f"headers.p4  → {includes_dir}/headers.p4")

        with open(os.path.join(includes_dir, "defines.p4"), "w", encoding="utf-8") as f:
            f.write(_DEFINES_P4)
        print(f"defines.p4  → {includes_dir}/defines.p4")

        with open(os.path.join(includes_dir, "parser.p4"), "w", encoding="utf-8") as f:
            f.write(_PARSER_P4)
        print(f"parser.p4   → {includes_dir}/parser.p4")

        if args.p4_spec:
            spec = emit_p4_spec_combined(encoding_tables, chosen_rules, names,
                                         default_action=default_action)
            with open(args.p4_spec, "w", encoding="utf-8") as f:
                f.write(spec)
            print(f"spec        → {args.p4_spec}")

    else:  # per-tree
        script = emit_bfrt(encoding_tables, tree_rules, base_score_int, names,
                           args.program, args.pipe)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(script)
        print(f"bfrt    → {args.out}")

        if args.p4_spec:
            spec = emit_p4_spec(encoding_tables, tree_rules, base_score_int, names)
            with open(args.p4_spec, "w", encoding="utf-8") as f:
                f.write(spec)
            print(f"spec    → {args.p4_spec}")


if __name__ == "__main__":
    main()
