#!/usr/bin/env python3
"""
xgboost_to_tofino.py

gate_dt JSON (XGBoost binary logistic)  →  Tofino bfrt script + P4 spec.

The gate_dt format already contains integer thresholds (threshold_int)
and integer leaf values (leaf_value_int) — no quantization needed.

model_kind     : binary_logistic_dt
prediction     : score = base_score_logit_int + sum(leaf_value_int per tree)
                 deflect if score >= 0

Usage
-----
  # Per-tree:
  python3 xgboost_to_tofino.py model.json \\
      --out bfrt.py --p4-spec spec.txt

  # Single combined table (recommended for small models):
  python3 xgboost_to_tofino.py model.json --mode combined \\
      --merge-strategy optimal
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from typing import Dict, List, Optional, Set, Tuple

# ── Types ──────────────────────────────────────────────────────────────────────

FeatureIndex = int
BucketIndex  = int
BucketRange  = Tuple[BucketIndex, BucketIndex]
IntThresholds = Dict[FeatureIndex, List[int]]
Names         = Dict[FeatureIndex, str]

_Condition = Tuple[FeatureIndex, str, int]    # (fi, "<" | ">=", threshold_int)
_LeafPath  = Tuple[List[_Condition], int]     # (conditions, leaf_value_int)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("model_kind") != "binary_logistic_dt":
        raise ValueError(f"Unsupported model_kind: {data.get('model_kind')}")

    names: Names = {i: n for i, n in enumerate(data["feature_names"])}
    trees         = data["trees"]
    base_score    = int(data["base_score_logit_int"])
    threshold_scale = int(data["threshold_scale"])
    leaf_scale      = int(data["leaf_scale"])

    return trees, names, base_score, threshold_scale, leaf_scale


# ── Threshold collection ───────────────────────────────────────────────────────

def collect_int_thresholds(trees: List[dict]) -> IntThresholds:
    """Collect all unique threshold_int values per feature across all trees."""
    splits: Dict[int, Set[int]] = collections.defaultdict(set)
    for tree in trees:
        for node in tree["nodes"]:
            if not node["is_leaf"]:
                splits[node["feature_index"]].add(node["threshold_int"])
    return {fi: sorted(vals) for fi, vals in sorted(splits.items())}


# ── Encoding tables ────────────────────────────────────────────────────────────

class EncodingTable:
    def __init__(self, feature: FeatureIndex, thresholds: List[int], field_max: int):
        self.feature   = feature
        self.field_max = field_max
        edges = [0] + thresholds + [field_max + 1]
        self.entries: List[Tuple[int, int, int]] = [
            (lo, hi_excl - 1, bucket)
            for bucket, (lo, hi_excl) in enumerate(zip(edges, edges[1:]))
        ]
        self.n_buckets   = len(self.entries)
        self.raw_bits    = math.ceil(math.log2(field_max + 1)) if field_max > 0 else 1
        self.bucket_bits = math.ceil(math.log2(self.n_buckets)) if self.n_buckets > 1 else 1


def build_encoding_tables(int_thresholds: IntThresholds, field_max: int) -> List[EncodingTable]:
    return [EncodingTable(fi, ths, field_max) for fi, ths in int_thresholds.items()]


# ── Per-tree rule extraction ───────────────────────────────────────────────────

def _leaf_paths(tree: dict) -> List[_LeafPath]:
    """DFS using explicit node list indexed by node_id."""
    node_map = {n["node_id"]: n for n in tree["nodes"]}
    paths: List[_LeafPath] = []

    def dfs(node_id: int, conds: List[_Condition]) -> None:
        node = node_map[node_id]
        if node["is_leaf"]:
            paths.append((list(conds), node["leaf_value_int"]))
            return
        fi  = node["feature_index"]
        thr = node["threshold_int"]
        dfs(node["left"],  conds + [(fi, "<",  thr)])
        dfs(node["right"], conds + [(fi, ">=", thr)])

    dfs(0, [])
    return paths


def _thr_to_bucket_range(thresholds: List[int], thr: int, op: str) -> Optional[BucketRange]:
    n = len(thresholds)
    if op == "<":
        hi = -1
        for k, t in enumerate(thresholds):
            if t <= thr:
                hi = k
            else:
                break
        return (0, hi) if hi >= 0 else None
    else:
        for k in range(n + 1):
            lower = thresholds[k - 1] if k > 0 else 0
            if lower >= thr:
                return (k, n)
        return (n, n)


class TreeRule:
    def __init__(self, match: Dict[FeatureIndex, BucketRange], leaf_int: int):
        self.match    = match
        self.leaf_int = leaf_int


def extract_tree_rules(tree: dict, int_thresholds: IntThresholds) -> List[TreeRule]:
    rules: List[TreeRule] = []
    for conds, leaf_int in _leaf_paths(tree):
        ranges: Dict[int, BucketRange] = {
            fi: (0, len(int_thresholds[fi])) for fi in {fi for fi, _, _ in conds}
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
            rules.append(TreeRule(ranges, leaf_int))
    return rules


# ── Combined table (cross-intersection of all trees) ──────────────────────────

def intersect_all_trees(
    tree_rules: List[Tuple[int, List[TreeRule]]],
    int_thresholds: IntThresholds,
    base_score_int: int,
) -> List[TreeRule]:
    """
    Cross-intersect rules from all trees into a single flat list.
    Each resulting rule covers a region where every tree's leaf is constant,
    and carries the precomputed final score (base_score + sum of all leaf values).
    """
    # Seed with first tree
    _, first_rules = tree_rules[0]
    combined: List[Tuple[Dict[FeatureIndex, BucketRange], int]] = [
        (dict(rule.match), base_score_int + rule.leaf_int)
        for rule in first_rules
    ]

    for _, rules in tree_rules[1:]:
        new_combined = []
        for match1, score1 in combined:
            for rule2 in rules:
                all_fi = set(match1) | set(rule2.match)
                new_match: Dict[FeatureIndex, BucketRange] = {}
                valid = True
                for fi in all_fi:
                    lo1, hi1 = match1.get(fi, (0, len(int_thresholds[fi])))
                    lo2, hi2 = rule2.match.get(fi, (0, len(int_thresholds[fi])))
                    lo, hi = max(lo1, lo2), min(hi1, hi2)
                    if lo > hi:
                        valid = False
                        break
                    new_match[fi] = (lo, hi)
                if valid:
                    new_combined.append((new_match, score1 + rule2.leaf_int))
        combined = new_combined

    return [TreeRule(match, score) for match, score in combined]


# ── Rule merging ───────────────────────────────────────────────────────────────

def _rule_key(r: TreeRule) -> tuple:
    return tuple(sorted((fi, lo, hi) for fi, (lo, hi) in r.match.items()))


def _try_merge(r1: TreeRule, r2: TreeRule) -> Optional[TreeRule]:
    if r1.match.keys() != r2.match.keys():
        return None
    diff = [fi for fi in r1.match if r1.match[fi] != r2.match[fi]]
    if len(diff) != 1:
        return None
    fi = diff[0]
    lo1, hi1 = r1.match[fi]
    lo2, hi2 = r2.match[fi]
    if hi1 + 1 != lo2 and hi2 + 1 != lo1:
        return None
    new_match = dict(r1.match)
    new_match[fi] = (min(lo1, lo2), max(hi1, hi2))
    return TreeRule(new_match, 0)


def merge_rules_greedy(rules: List[TreeRule]) -> List[TreeRule]:
    """
    Greedy iterative merge: pairs that differ in one dimension and are adjacent.
    Fast (O(n²·iterations)) but may not find the global minimum.
    """
    current = list(rules)
    changed = True
    while changed:
        changed = False
        used: Set[int] = set()
        merged: List[TreeRule] = []
        for i in range(len(current)):
            if i in used:
                continue
            for j in range(i + 1, len(current)):
                if j in used:
                    continue
                m = _try_merge(current[i], current[j])
                if m:
                    merged.append(m)
                    used.add(i)
                    used.add(j)
                    changed = True
                    break
            if i not in used:
                merged.append(current[i])
        current = merged
    return current


def _other_key(r: TreeRule, skip_fi: int) -> tuple:
    """Match key excluding feature skip_fi — used to group candidates for merging along skip_fi."""
    return tuple(sorted((fi, lo, hi) for fi, (lo, hi) in r.match.items() if fi != skip_fi))


def _generate_prime_implicants(rules: List[TreeRule]) -> List[TreeRule]:
    """
    Generate all prime implicants (maximal rectangles in the deflect region).

    Uses a saturating pool with feature-grouped merging:
      For each feature fi, group rules by their match on ALL OTHER features.
      Within each group, only rules with identical non-fi ranges can merge
      along fi — so we only compare within groups.
      This turns the O(n²) global scan into many small O(k²) scans.
    """
    pool: Dict[tuple, TreeRule] = {_rule_key(r): r for r in rules}

    changed = True
    while changed:
        changed = False
        pool_list = list(pool.values())
        all_features = sorted({fi for r in pool_list for fi in r.match})

        for merge_fi in all_features:
            groups: Dict[tuple, List[TreeRule]] = {}
            for r in pool_list:
                if merge_fi in r.match:
                    groups.setdefault(_other_key(r, merge_fi), []).append(r)

            for group in groups.values():
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        m = _try_merge(group[i], group[j])
                        if m:
                            k = _rule_key(m)
                            if k not in pool:
                                pool[k] = m
                                changed = True

    # Prime implicants: no other rule in the pool is adjacent along any feature
    pool_list = list(pool.values())
    all_features = sorted({fi for r in pool_list for fi in r.match})
    not_prime: Set[tuple] = set()
    for merge_fi in all_features:
        groups: Dict[tuple, List[TreeRule]] = {}
        for r in pool_list:
            if merge_fi in r.match:
                groups.setdefault(_other_key(r, merge_fi), []).append(r)
        for group in groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if _try_merge(group[i], group[j]):
                        not_prime.add(_rule_key(group[i]))
                        not_prime.add(_rule_key(group[j]))

    return [r for r in pool_list if _rule_key(r) not in not_prime]


def _subsumes(pi: TreeRule, cell: TreeRule) -> bool:
    """True if prime implicant pi fully contains atomic cell."""
    for fi, (lo_c, hi_c) in cell.match.items():
        if fi not in pi.match:
            return False
        lo_p, hi_p = pi.match[fi]
        if lo_c < lo_p or hi_c > hi_p:
            return False
    return True


def _min_set_cover(n_cells: int, coverage: List[frozenset]) -> List[int]:
    """
    Exact minimum set cover via branch-and-bound with most-constrained-cell pivot.

    Strategy: at each node pick the cell covered by fewest PIs (hardest to cover),
    then branch on which PI to use for it. Branching factor = n_PIs_for_that_cell,
    typically 1-5, vs n_total_PIs in a naive approach.
    """
    # Step 1: extract essential PIs (only PI covering some cell)
    cell_to_pis: List[List[int]] = [[] for _ in range(n_cells)]
    for j, cov in enumerate(coverage):
        for i in cov:
            cell_to_pis[i].append(j)

    essential: Set[int] = set()
    for i in range(n_cells):
        if len(cell_to_pis[i]) == 1:
            essential.add(cell_to_pis[i][0])

    covered_by_essential = frozenset().union(*(coverage[j] for j in essential))
    init_remaining = frozenset(range(n_cells)) - covered_by_essential

    if not init_remaining:
        return sorted(essential)

    available = [j for j in range(len(coverage)) if j not in essential]
    av_cov    = [coverage[j] for j in available]  # pre-index for speed

    best: List[List[int]] = [list(range(len(available)))]  # worst: all available

    def bnb(remaining: frozenset, chosen: List[int]) -> None:
        if not remaining:
            if len(chosen) < len(best[0]):
                best[0] = list(chosen)
            return
        if len(chosen) >= len(best[0]) - 1:
            return

        # Lower bound via max coverage of remaining available PIs
        max_gain = max((len(av_cov[k] & remaining) for k in range(len(available))
                        if k not in chosen_set), default=0)
        if max_gain == 0:
            return
        if len(chosen) + math.ceil(len(remaining) / max_gain) >= len(best[0]):
            return

        # Pivot: pick the cell covered by fewest available (not yet chosen) PIs
        pivot_cell = min(
            remaining,
            key=lambda i: sum(1 for k, c in enumerate(av_cov)
                              if k not in chosen_set and i in c)
        )
        candidates = [k for k, c in enumerate(av_cov)
                      if k not in chosen_set and pivot_cell in c]
        if not candidates:
            return  # uncoverable

        for k in sorted(candidates, key=lambda k: -len(av_cov[k] & remaining)):
            chosen_set.add(k)
            bnb(remaining - av_cov[k], chosen + [k])
            chosen_set.discard(k)

    chosen_set: Set[int] = set()
    bnb(init_remaining, [])

    return sorted(essential) + sorted(available[k] for k in best[0])


def merge_rules_optimal(rules: List[TreeRule]) -> List[TreeRule]:
    """
    Optimal minimum cover via Quine-McCluskey prime implicant generation
    followed by exact branch-and-bound set cover.
    May be slow for large rule sets (> ~500 cells).
    """
    if not rules:
        return []

    prime_implicants = _generate_prime_implicants(rules)

    coverage = [
        frozenset(i for i, cell in enumerate(rules) if _subsumes(pi, cell))
        for pi in prime_implicants
    ]

    chosen_idx = _min_set_cover(len(rules), coverage)
    return [prime_implicants[i] for i in chosen_idx]


def merge_rules(rules: List[TreeRule], strategy: str = "greedy") -> List[TreeRule]:
    if strategy == "optimal":
        return merge_rules_optimal(rules)
    return merge_rules_greedy(rules)


# ── bfrt script (combined mode) ───────────────────────────────────────────────

def emit_bfrt_combined(
    encoding_tables: List[EncodingTable],
    rules: List[TreeRule],           # already filtered and merged
    names: Names,
    p4_program: str,
    pipe: str,
    default_action: str = "no_deflect",  # "no_deflect": hit=do_deflect; "deflect": hit=no_deflect
    key_field_map: Optional[Dict[str, str]] = None,  # feature_name -> p4 key field name
) -> str:
    """bfshell bfrt_python script (run via $SDE/run_bfshell.sh -b script.py).
    default_action='no_deflect': entries fire do_deflect, miss = no deflect.
    default_action='deflect':    entries fire no_deflect, miss = deflect.
    key_field_map: overrides the p4 key field for an encode table
                   e.g. {'occ_vs_min_cand': 'occ_vs', 'occ_vs_mean_cand': 'occ_vs'}
    """
    hit_action = "do_deflect" if default_action == "no_deflect" else "no_deflect"
    miss_label = "no deflect" if default_action == "no_deflect" else "deflect"
    kfm = key_field_map or {}
    L: List[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    def fname(fi: FeatureIndex) -> str:
        return names.get(fi, f"f{fi}")

    def kfield(n: str) -> str:
        return kfm.get(n, n)

    w("#!/usr/bin/env python3")
    w(f"# bfshell bfrt_python control plane — {p4_program}  (combined, {len(rules)} entries, default={default_action})")
    w("# Run with: $SDE/run_bfshell.sh -b <this_file>")
    w()
    w(f"ig = bfrt.{p4_program}.pipe.SwitchIngress")
    w()

    w("# ── Stage 1: Encoding tables ─────────────────────────────────────────────")
    for enc in encoding_tables:
        n = fname(enc.feature)
        k = kfield(n)
        w()
        w(f"# {n}: {enc.n_buckets} buckets")
        w(f"_t = ig.encode_{n}")
        w(f"_t.clear()")
        for lo, hi, bucket in enc.entries:
            w(f"_t.add_with_set_bucket_{n}({k}_start={lo}, {k}_end={hi}, bucket={bucket})")

    w()
    w(f"# ── Combined decision table (hit = {hit_action}, miss = {miss_label}) ───────────")
    w()
    w(f"# model_decision: {len(rules)} entries")
    w(f"_t = ig.model_decision")
    w(f"_t.clear()")
    for rule in rules:
        kv = ", ".join(
            f"bucket_{fname(fi)}_start={lo}, bucket_{fname(fi)}_end={hi}"
            for fi, (lo, hi) in sorted(rule.match.items())
        )
        w(f"_t.add_with_{hit_action}({kv})")
    w()

    w("print('All tables populated.')")
    return "\n".join(L)


# ── P4 spec (combined mode) ────────────────────────────────────────────────────

def emit_p4_spec_combined(
    encoding_tables: List[EncodingTable],
    rules: List[TreeRule],           # already filtered and merged
    names: Names,
    leaf_bits: int = 32,
    default_action: str = "no_deflect",
) -> str:
    L: List[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    def fname(fi: FeatureIndex) -> str:
        return names.get(fi, f"f{fi}")

    hit_action  = "do_deflect" if default_action == "no_deflect" else "no_deflect"
    p4_default  = "NoAction()" if default_action == "no_deflect" else "do_deflect()"
    decision_desc = "hit = deflect, miss = no deflect" if default_action == "no_deflect" \
                    else "hit = no deflect, miss = deflect"
    enc_map = {enc.feature: enc for enc in encoding_tables}
    all_features = sorted({fi for rule in rules for fi in rule.match})

    w("=" * 72)
    w("P4 TABLE SPEC  —  auto-generated from gate_dt model  (combined mode)")
    w("=" * 72)
    w()

    w("── METADATA STRUCT ──────────────────────────────────────────────────────")
    w()
    w("struct gate_dt_meta_t {")
    w()
    w("    // Stage 1 inputs: raw feature values")
    for enc in encoding_tables:
        n = fname(enc.feature)
        w(f"    bit<{enc.raw_bits:2d}>  {n};")
    w()
    w("    // Stage 1 outputs: bucket indices")
    for enc in encoding_tables:
        n = fname(enc.feature)
        w(f"    bit<{enc.bucket_bits:2d}>  bucket_{n};       // {enc.n_buckets} buckets")
    w()
    w(f"    bit< 1>  deflect;            // set by model_decision table")
    w("}")
    w()

    w("── STAGE 1: ENCODING TABLES ─────────────────────────────────────────────")
    w()
    for enc in encoding_tables:
        n = fname(enc.feature)
        w(f"table encode_{n} {{")
        w(f"    key     = {{ ig_md.{n} : range; }}")
        w(f"    actions = {{ set_bucket_{n}; }}")
        w(f"    size    = {enc.n_buckets};")
        w(f"    // action set_bucket_{n}(bit<{enc.bucket_bits}> bucket):")
        w(f"    //   ig_md.bucket_{n} = bucket;")
        w(f"}}")
        w()

    w("── STAGE 2: COMBINED DECISION TABLE ─────────────────────────────────────")
    w()
    key_lines = "\n".join(
        f"        ig_md.bucket_{fname(fi)} : range;"
        for fi in all_features if fi in enc_map
    )
    key_bits = sum(enc_map[fi].bucket_bits for fi in all_features if fi in enc_map)
    w(f"table model_decision {{")
    w(f"    key = {{")
    w(f"{key_lines}")
    w(f"    }}")
    w(f"    actions     = {{ do_deflect; no_deflect; }}")
    w(f"    default_action = {p4_default};")
    w(f"    size        = {len(rules)};")
    w(f"    // total match key width: {key_bits} bits")
    w(f"    // action {hit_action}():")
    w(f"    //   ig_md.deflect = {'1' if hit_action == 'do_deflect' else '0'};")
    w(f"}}")
    w()

    w("── APPLY BLOCK ──────────────────────────────────────────────────────────")
    w()
    init_val = "0" if default_action == "no_deflect" else "1"
    w(f"    ig_md.deflect = {init_val};")
    for enc in encoding_tables:
        w(f"    encode_{fname(enc.feature)}.apply();")
    w()
    w("    model_decision.apply();")
    w()

    w("── SUMMARY ──────────────────────────────────────────────────────────────")
    w()
    w(f"  Encoding tables  : {len(encoding_tables)}  "
      f"(total {sum(e.n_buckets for e in encoding_tables)} entries)")
    w(f"  Decision table   : 1  ({len(rules)} entries after merge)")
    w(f"  Decision         : {decision_desc}")
    w()

    return "\n".join(L)


# ── P4 code (combined mode) ────────────────────────────────────────────────────

def emit_p4_code_combined(
    encoding_tables: List[EncodingTable],
    rules: List[TreeRule],
    names: Names,
    default_action: str = "no_deflect",
    key_field_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Emit complete P4 file identical to prova.p4 except for model-specific tables.
    default_action='no_deflect': hit=do_deflect, check == 1
    default_action='deflect':    hit=no_deflect, check != 1
    key_field_map: maps feature_name -> p4 metadata key field
                   e.g. {'occ_vs_min_cand': 'occ_vs', 'occ_vs_mean_cand': 'occ_vs'}
    """
    hit_action    = "do_deflect" if default_action == "no_deflect" else "no_deflect"
    deflect_check = "== 1"       if default_action == "no_deflect" else "!= 1"
    kfm = key_field_map or {}

    def fname(fi: FeatureIndex) -> str:
        return names.get(fi, f"f{fi}")

    def kfield(n: str) -> str:
        return kfm.get(n, n)

    enc_map      = {enc.feature: enc for enc in encoding_tables}
    all_features = sorted({fi for rule in rules for fi in rule.match})

    # ── Model-specific: encode actions + tables ───────────────────────────────
    model_lines: List[str] = []
    for enc in encoding_tables:
        n = fname(enc.feature)
        k = kfield(n)
        model_lines.append(f"    action set_bucket_{n}(bit<{enc.bucket_bits}> bucket) {{")
        model_lines.append(f"        ig_md.bucket_{n} = bucket;")
        model_lines.append(f"    }}")
        model_lines.append(f"")
        model_lines.append(f"    table encode_{n} {{")
        model_lines.append(f"        key     = {{ ig_md.{k} : range; }}")
        model_lines.append(f"        actions = {{ set_bucket_{n}; }}")
        model_lines.append(f"        size    = {enc.n_buckets};")
        model_lines.append(f"    }}")
        model_lines.append(f"")

    model_lines.append(f"    action {hit_action}() {{")
    model_lines.append(f"        ig_md.deflect = 1;")
    model_lines.append(f"    }}")
    model_lines.append(f"")
    model_lines.append(f"    table model_decision {{")
    model_lines.append(f"        key = {{")
    for fi in all_features:
        if fi in enc_map:
            n = fname(fi)
            model_lines.append(f"            ig_md.bucket_{n} : range;")
    model_lines.append(f"        }}")
    model_lines.append(f"        actions     = {{ {hit_action}; NoAction; }}")
    model_lines.append(f"        default_action = NoAction();")
    model_lines.append(f"        size        = {len(rules)};")
    model_lines.append(f"    }}")
    model_section = "\n".join(model_lines)

    # ── Apply: encode calls ───────────────────────────────────────────────────
    encode_apply = "\n".join(
        f"            encode_{fname(enc.feature)}.apply();" for enc in encoding_tables
    )

    return f"""\
#include <core.p4>
#include <tna.p4>

#include "includes/defines.p4"
#include "includes/headers.p4"
#include "includes/parser.p4"

/*
    Switch Ingress Pipeline
*/

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {{

    // --- Actions ---

    action drop() {{
        ig_intr_dprsr_md.drop_ctl = 1;
    }}

    action set_deflect_port(bit<16> port) {{
        ig_intr_tm_md.ucast_egress_port = (bit<9>)port;
    }}

    action nop() {{ }}

    // Standard L3 Forwarding
    action ipv4_forward(bit<16> port, bit<2> port_logical, bit<48> dstAddr, bit<48> srcAddr) {{
        ig_intr_tm_md.ucast_egress_port = (bit<9>)port;
        ig_md.port_idx = port_logical;
        hdr.ethernet.srcAddr = srcAddr;
        hdr.ethernet.dstAddr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }}


    #define DECLARE_IG_QUEUE_LEN_REG(N)                                        \\
        Register<bit<16>, bit<1>>(1) ig_queue##N##_len_reg;                   \\
        RegisterAction<bit<16>, bit<1>, bit<16>>(ig_queue##N##_len_reg)        \\
        r_ig_queue##N##_len_alu = {{                                            \\
            void apply(inout bit<16> register_data, out bit<16> result) {{      \\
                result = register_data;                                         \\
            }}                                                                   \\
        }};                                                                     \\
        RegisterAction<bit<16>, bit<1>, void>(ig_queue##N##_len_reg)           \\
        w_ig_queue##N##_len_alu = {{                                            \\
            void apply(inout bit<16> register_data) {{                          \\
                register_data = (bit<16>)(hdr.bee.queue_occ);                  \\
            }}                                                                   \\
        }};                                                                       \\
        action read_ig_queue##N##_len() {{                                               \\
            ig_md.q##N##_occ = r_ig_queue##N##_len_alu.execute(0);              \\
        }}                                                                        \\
        action write_ig_queue##N##_len() {{                                              \\
            w_ig_queue##N##_len_alu.execute(0);                                              \\
        }}                                                                          \\
        table write_ig_queue##N##_len_tbl {{                                                   \\
            actions = {{ write_ig_queue##N##_len;}}                                   \\
            default_action = write_ig_queue##N##_len();                                                        \\
            size = 1;                                                                      \\
        }}                                                                         \\
        table read_ig_queue##N##_len_tbl {{                                                   \\
            actions = {{ read_ig_queue##N##_len;}}                                   \\
            default_action = read_ig_queue##N##_len();                                                        \\
            size = 1;                                                                      \\
        }}

    DECLARE_IG_QUEUE_LEN_REG(0)
    DECLARE_IG_QUEUE_LEN_REG(1)
    DECLARE_IG_QUEUE_LEN_REG(2)
    DECLARE_IG_QUEUE_LEN_REG(3)

    // 1. Forwarding Table
    table ipv4_lpm {{
        key = {{
            hdr.ipv4.dstAddr : lpm;
        }}
        actions = {{
            ipv4_forward;
            //manage_bee;
            drop;
        }}
        size = TABLE_SIZE;
        default_action = drop();
    }}

    // 2. Convert logical deflection port to physical
    table deflect_port_table {{
        key = {{
            ig_md.min_id : exact;
        }}
        actions = {{
            set_deflect_port;
            drop;
        }}
        size = TABLE_SIZE;
        default_action = drop();
    }}

{model_section}

    // Mux: select the local-port occupancy based on port_idx set by ipv4_lpm.
    action set_local_occ_q0() {{ ig_md.local_queue_occ = ig_md.q0_occ; }}
    action set_local_occ_q1() {{ ig_md.local_queue_occ = ig_md.q1_occ; }}
    action set_local_occ_q2() {{ ig_md.local_queue_occ = ig_md.q2_occ; }}
    action set_local_occ_q3() {{ ig_md.local_queue_occ = ig_md.q3_occ; }}
    table select_local_occ_tbl {{
        key = {{ ig_md.port_idx : exact; }}
        actions = {{ set_local_occ_q0; set_local_occ_q1;
                    set_local_occ_q2; set_local_occ_q3; nop; }}
        const entries = {{
            2w0 : set_local_occ_q0();
            2w1 : set_local_occ_q1();
            2w2 : set_local_occ_q2();
            2w3 : set_local_occ_q3();
        }}
        default_action = nop();
        size = 4;
    }}

    // Sorting-network level 1: pairwise comparisons + partial mean-sums.
    // Slot-reuse (overwritten in level2/level3):
    //   min_cand_occ ← min(q0,q1)   max1 ← max(q0,q1)
    //   cand2_occ    ← min(q2,q3)   cand_occ_spread ← max(q2,q3)
    //   sum          ← (q0>>2)+(q1>>2)   lo_hi ← (q2>>2)+(q3>>2)
    // All 6 outputs independent → single stage.
    action compute_level1() {{
        ig_md.min_cand_occ    = min(ig_md.q0_occ, ig_md.q1_occ);
        ig_md.max1            = max(ig_md.q0_occ, ig_md.q1_occ);
        ig_md.cand2_occ       = min(ig_md.q2_occ, ig_md.q3_occ);
        ig_md.cand_occ_spread = max(ig_md.q2_occ, ig_md.q3_occ);
        ig_md.sum             = ig_md.q0_occ + ig_md.q1_occ;
        ig_md.lo_hi           = ig_md.q2_occ + ig_md.q3_occ;
        ig_md.occ_vs          = ig_md.q0_occ |-| ig_md.q1_occ;  // scratch: >0 iff q0 > q1
    }}
    table level1_tbl {{
        actions        = {{ compute_level1; }}
        default_action = compute_level1();
        size           = 1;
    }}

    // Sorting-network level 2: merge pairs into final min/max + intermediates.
    // sum computed first so it reads OLD lo_hi (= q2+q3 partial sum from level1).
    // All 5 outputs read DIFFERENT fields than any prior write → no intra-action deps.
    action compute_level2() {{
        ig_md.sum          = ig_md.sum + ig_md.lo_hi;                  // (q0+q1)+(q2+q3) = sum*4
        ig_md.lo_hi        = max(ig_md.min_cand_occ, ig_md.cand2_occ); // max(min01,min23)
        ig_md.hi_lo        = min(ig_md.max1, ig_md.cand_occ_spread);   // min(max01,max23)
        ig_md.min_cand_occ = min(ig_md.min_cand_occ, ig_md.cand2_occ); // final overall min
        ig_md.max1         = max(ig_md.max1, ig_md.cand_occ_spread);   // final overall max
    }}
    table level2_tbl {{
        actions        = {{ compute_level2; }}
        default_action = compute_level2();
        size           = 1;
    }}

    apply {{
        if(hdr.bee.isValid()){{
            ig_intr_tm_md.ucast_egress_port[8:7] = ig_intr_md.ingress_port[8:7];
            ig_intr_tm_md.ucast_egress_port[6:0] = 68;
            if(hdr.bee.port_id == 0){{
                write_ig_queue0_len_tbl.apply();
            }} else if(hdr.bee.port_id == 1){{
                write_ig_queue1_len_tbl.apply();
            }} else if(hdr.bee.port_id == 2){{
                write_ig_queue2_len_tbl.apply();
            }} else if(hdr.bee.port_id == 3){{
                write_ig_queue3_len_tbl.apply();
            }}
        }} else if(hdr.ipv4.isValid()){{
            ipv4_lpm.apply();

            // All 4 reads are independent (no shared tmp) → compiler can parallelise.
            read_ig_queue0_len_tbl.apply();
            read_ig_queue1_len_tbl.apply();
            read_ig_queue2_len_tbl.apply();
            read_ig_queue3_len_tbl.apply();

            // Level-1: pairwise comparisons + partial sums + min_id scratch (one stage).
            select_local_occ_tbl.apply();
            level1_tbl.apply();

            // min_id: occ_vs = q0 |-| q1 from level1; >0 means q1 < q0 → port 1 is less loaded.
            if (ig_md.occ_vs != 0) {{ ig_md.min_id = 1; }}

            // Level-2: merge pairs (single stage, no intra-action deps).
            level2_tbl.apply();

            // Level-3: second smallest = min of the two "middle" sorted elements.
            ig_md.cand2_occ = min(ig_md.lo_hi, ig_md.hi_lo);

            // Shift sum to get mean (single op → own auto-gen stage).
            ig_md.sum = ig_md.sum >> 2;
            // occ_vs_mean = local |-| mean.
            ig_md.occ_vs = ig_md.local_queue_occ |-| ig_md.sum;
            encode_occ_vs_mean_cand.apply();

            ig_md.cand_occ_spread = ig_md.max1 - ig_md.min_cand_occ;

            encode_cand2_occ.apply();
            encode_cand_occ_spread.apply();
            encode_min_cand_occ.apply();
            encode_local_queue_occ.apply();

            ig_md.occ_vs = ig_md.local_queue_occ |-| ig_md.min_cand_occ;
            encode_occ_vs_min_cand.apply();

            model_decision.apply();

            if (ig_md.deflect {deflect_check}) {{
                deflect_port_table.apply();
            }}
        }}
    }}
}}

/*
    Switch Egress Pipeline
*/
control SwitchEgress(
        inout header_t hdr,
        inout metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprsr,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_md_for_oport) {{

    action nop() {{ }}

    action set_port_idx_in_reg(bit<2> logical_port) {{
        eg_md.port_idx = logical_port;
    }}


    #define DECLARE_EG_QUEUE_LEN_REG(N)                                        \\
        Register<bit<16>, bit<1>>(1) eg_queue##N##_len_reg;                   \\
        RegisterAction<bit<16>, bit<1>, bit<16>>(eg_queue##N##_len_reg)        \\
        r_eg_queue##N##_len_alu = {{                                            \\
            void apply(inout bit<16> register_data, out bit<16> result) {{      \\
                result = register_data;                                         \\
            }}                                                                   \\
        }};                                                                     \\
        RegisterAction<bit<16>, bit<1>, void>(eg_queue##N##_len_reg)           \\
        w_eg_queue##N##_len_alu = {{                                            \\
            void apply(inout bit<16> register_data) {{                          \\
                register_data = (bit<16>)(eg_md.local_queue_occ);             \\
            }}                                                                   \\
        }};

    //TODO: da decidere logica, shiftare? Cappare la dim massima della coda?

    DECLARE_EG_QUEUE_LEN_REG(0)
    DECLARE_EG_QUEUE_LEN_REG(1)
    DECLARE_EG_QUEUE_LEN_REG(2)
    DECLARE_EG_QUEUE_LEN_REG(3)

    table set_port_idx_table {{
        key = {{
            eg_intr_md.egress_port : exact;

        }}
        actions = {{
            set_port_idx_in_reg;
            nop;
        }}
        size = TABLE_SIZE;
        default_action = nop();
    }}

    apply {{
        if(hdr.bee.isValid()) {{
            // Bee packet: load current egress queue depth into header so
            // ingress can update its register on the next loop iteration.
            if(hdr.bee.port_id == 0){{
                hdr.bee.queue_occ = r_eg_queue0_len_alu.execute(0);
            }} else if(hdr.bee.port_id == 1){{
                hdr.bee.queue_occ = r_eg_queue1_len_alu.execute(0);
            }} else if(hdr.bee.port_id == 2){{
                hdr.bee.queue_occ = r_eg_queue2_len_alu.execute(0);
            }} else if(hdr.bee.port_id == 3){{
                hdr.bee.queue_occ = r_eg_queue3_len_alu.execute(0);
            }}
        }} else {{
            // Data packet: snapshot egress queue depth into registers.
            set_port_idx_table.apply();
            eg_md.local_queue_occ = (bit<16>) eg_intr_md.deq_qdepth << 5;
            if(eg_intr_md.egress_port == 0){{
                w_eg_queue0_len_alu.execute(0);
            }} else if(eg_intr_md.egress_port == 1){{
                w_eg_queue1_len_alu.execute(0);
            }} else if(eg_intr_md.egress_port == 2){{
                w_eg_queue2_len_alu.execute(0);
            }} else if(eg_intr_md.egress_port == 3){{
                w_eg_queue3_len_alu.execute(0);
            }}
        }}
    }}
}}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;

Switch(pipe) main;
"""


# ── bfrt script (per-tree mode) ────────────────────────────────────────────────

def emit_bfrt(
    encoding_tables: List[EncodingTable],
    tree_rules: List[Tuple[int, List[TreeRule]]],   # (tree_idx, rules)
    base_score_int: int,
    names: Names,
    p4_program: str,
    pipe: str,
) -> str:
    L: List[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    def fname(fi: FeatureIndex) -> str:
        return names.get(fi, f"f{fi}")

    w("#!/usr/bin/env python3")
    w(f"# bfshell bfrt_python control plane — {p4_program}  (per-tree, base_score={base_score_int})")
    w("# Run with: $SDE/run_bfshell.sh -b <this_file>")
    w()
    w(f"ig = bfrt.{p4_program}.pipe.SwitchIngress")
    w()

    w("# ── Stage 1: Encoding tables ─────────────────────────────────────────────")
    for enc in encoding_tables:
        n = fname(enc.feature)
        w()
        w(f"# {n}: {enc.n_buckets} buckets")
        w(f"_t = ig.encode_{n}")
        w(f"_t.clear()")
        for lo, hi, bucket in enc.entries:
            w(f"_t.add_with_set_bucket_{n}({n}_start={lo}, {n}_end={hi}, bucket={bucket})")

    w()
    w("# ── Stage 2: Tree match-action tables ───────────────────────────────────")
    for tidx, rules in tree_rules:
        w()
        w(f"# tree_{tidx}: {len(rules)} entries")
        w(f"_t = ig.tree_{tidx}")
        w(f"_t.clear()")
        for rule in rules:
            kv = ", ".join(
                f"bucket_{fname(fi)}_start={lo}, bucket_{fname(fi)}_end={hi}"
                for fi, (lo, hi) in sorted(rule.match.items())
            )
            w(f"_t.add_with_set_leaf_{tidx}({kv}, leaf_value={rule.leaf_int})")
    w()

    w("print('All tables populated.')")
    return "\n".join(L)


# ── P4 spec ────────────────────────────────────────────────────────────────────

def emit_p4_spec(
    encoding_tables: List[EncodingTable],
    tree_rules: List[Tuple[int, List[TreeRule]]],
    base_score_int: int,
    names: Names,
    leaf_bits: int = 32,
    score_bits: int = 32,
) -> str:
    L: List[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    def fname(fi: FeatureIndex) -> str:
        return names.get(fi, f"f{fi}")

    enc_map = {enc.feature: enc for enc in encoding_tables}

    w("=" * 72)
    w("P4 TABLE SPEC  —  auto-generated from gate_dt model")
    w("=" * 72)
    w()

    # ── Metadata struct ───────────────────────────────────────────────────────
    w("── METADATA STRUCT ──────────────────────────────────────────────────────")
    w()
    w("struct gate_dt_meta_t {")
    w()
    w("    // Stage 1 inputs: raw feature values (integer-scaled)")
    for enc in encoding_tables:
        n = fname(enc.feature)
        w(f"    bit<{enc.raw_bits:2d}>  {n};")
    w()
    w("    // Stage 1 outputs: bucket indices")
    for enc in encoding_tables:
        n = fname(enc.feature)
        w(f"    bit<{enc.bucket_bits:2d}>  bucket_{n};       // {enc.n_buckets} buckets")
    w()
    w("    // Stage 2 outputs: leaf values per tree")
    for tidx, _ in tree_rules:
        w(f"    int<{leaf_bits:2d}>  leaf_{tidx};")
    w()
    w(f"    int<{score_bits:2d}>  score;              // sum of all leaf values + base_score")
    w(f"    bit< 1>  deflect;            // 1 if score >= 0")
    w("}")
    w()

    # ── Encoding tables ───────────────────────────────────────────────────────
    w("── STAGE 1: ENCODING TABLES ─────────────────────────────────────────────")
    w()
    for enc in encoding_tables:
        n = fname(enc.feature)
        w(f"table encode_{n} {{")
        w(f"    key     = {{ ig_md.{n} : range; }}")
        w(f"    actions = {{ set_bucket_{n}; }}")
        w(f"    size    = {enc.n_buckets};")
        w(f"    // action set_bucket_{n}(bit<{enc.bucket_bits}> bucket):")
        w(f"    //   ig_md.bucket_{n} = bucket;")
        w(f"}}")
        w()

    # ── Tree tables ───────────────────────────────────────────────────────────
    w("── STAGE 2: TREE MATCH-ACTION TABLES ────────────────────────────────────")
    w()
    for tidx, rules in tree_rules:
        features_used = sorted({fi for rule in rules for fi in rule.match})
        key_bits = sum(enc_map[fi].bucket_bits for fi in features_used if fi in enc_map)
        key_lines = "\n".join(
            f"        ig_md.bucket_{fname(fi)} : range;"
            for fi in features_used
        )
        w(f"table tree_{tidx} {{")
        w(f"    key = {{")
        w(f"{key_lines}")
        w(f"    }}")
        w(f"    actions = {{ set_leaf_{tidx}; }}")
        w(f"    size    = {len(rules)};")
        w(f"    // total match key width: {key_bits} bits")
        w(f"    // action set_leaf_{tidx}(int<{leaf_bits}> leaf_value):")
        w(f"    //   ig_md.leaf_{tidx} = leaf_value;")
        w(f"}}")
        w()

    # ── Combination logic ─────────────────────────────────────────────────────
    w("── COMBINATION LOGIC (apply block) ──────────────────────────────────────")
    w()
    w("    // Stage 1: encode raw features → bucket indices")
    for enc in encoding_tables:
        w(f"    encode_{fname(enc.feature)}.apply();")
    w()
    w("    // Stage 2: look up leaf value per tree")
    for tidx, _ in tree_rules:
        w(f"    tree_{tidx}.apply();")
    w()
    w("    // Stage 3: sum all leaf values + base_score")
    terms = " + ".join(f"ig_md.leaf_{tidx}" for tidx, _ in tree_rules)
    w(f"    ig_md.score = {base_score_int} + {terms};")
    w()
    w("    // Stage 4: deflect if score >= 0")
    w("    ig_md.deflect = (ig_md.score >= 0) ? 1w1 : 1w0;")
    w()

    # ── Summary ───────────────────────────────────────────────────────────────
    w("── SUMMARY ──────────────────────────────────────────────────────────────")
    w()
    w(f"  Encoding tables : {len(encoding_tables)}  "
      f"(total {sum(e.n_buckets for e in encoding_tables)} entries)")
    w(f"  Tree tables     : {len(tree_rules)}  "
      f"(total {sum(len(r) for _, r in tree_rules)} entries)")
    w(f"  base_score_int  : {base_score_int}")
    w(f"  Decision        : deflect if (base_score_int + sum(leaf values)) >= 0")
    w()

    return "\n".join(L)


def emit_headers_p4(encoding_tables: "List[EncodingTable]", names: "Names") -> str:
    """Generate includes/headers.p4 with correct bucket bit-widths for this model."""
    def fname(fi: "FeatureIndex") -> str:
        return names.get(fi, f"f{fi}")
    bucket_lines = "\n".join(
        f"    bit<{enc.bucket_bits}>  bucket_{fname(enc.feature)};"
        for enc in encoding_tables
    )
    return f"""\
#ifndef _D8_HEADERS_
#define _D8_HEADERS_

#include "defines.p4"

// Ethernet Header
header ethernet_t {{
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}}

// IPv4 Header
header ipv4_t {{
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}}

// Bee Header (Worker Packet)
header bee_t {{
    bit<8> port_id;
    bit<16> queue_occ;
}}

struct header_t {{
    ethernet_t ethernet;
    ipv4_t     ipv4;
    bee_t      bee;
}}

// User Metadata
struct metadata_t {{
    // Forwarding
    bit<2> port_idx;
    bit<2> min_id;
    bit<1>  deflect;

    bit<16> local_queue_occ;
    bit<16> min_cand_occ;
    bit<16> max1;
    bit<16> cand2_occ;
    bit<16> cand_occ_spread;
    bit<16> occ_vs;
    bit<16> q0_occ;
    bit<16> q1_occ;
    bit<16> q2_occ;
    bit<16> q3_occ;
    bit<16> sum;
    bit<16> lo_hi;
    bit<16> hi_lo;

    // Bucketed features for model_decision table
{bucket_lines}
}}

#endif
"""


def emit_build_sh(program_name: str) -> str:
    """Generate build.sh for bf-p4c compilation."""
    # The heredoc inside the shell script contains Python f-string syntax;
    # those braces are escaped ({{...}}) so the outer f-string leaves them as-is.
    return (
        "#!/bin/bash\n"
        "\n"
        "# Environment Setup\n"
        "export SDE=$HOME/bf-sde-9.13.4\n"
        "export SDE_INSTALL=$SDE/install\n"
        "export PATH=$SDE_INSTALL/bin:$PATH\n"
        "\n"
        "# Configuration\n"
        f'P4_PROGRAM_NAME="{program_name}"\n'
        f'P4_SOURCE="{program_name}.p4"\n'
        'BUILD_DIR="build"\n'
        "\n"
        'echo "=========================================="\n'
        'echo "Compiling $P4_PROGRAM_NAME"\n'
        'echo "=========================================="\n'
        "\n"
        "mkdir -p $BUILD_DIR\n"
        "\n"
        "bf-p4c --target tofino \\\n"
        "       --arch tna \\\n"
        "       --create-graphs \\\n"
        "       --verbose 2 \\\n"
        "       --display-power-budget \\\n"
        "       --parser-timing-reports \\\n"
        "       --program-name $P4_PROGRAM_NAME \\\n"
        "       -o $BUILD_DIR \\\n"
        "       $P4_SOURCE\n"
        "\n"
        "if [ $? -eq 0 ]; then\n"
        '    ABS_BUILD="$PWD/$BUILD_DIR"\n'
        '    python3 - "$BUILD_DIR/$P4_PROGRAM_NAME.conf" "$ABS_BUILD" <<\'PYEOF\'\n'
        "import sys, json\n"
        "conf_path, abs_build = sys.argv[1], sys.argv[2]\n"
        "with open(conf_path) as f: conf = json.load(f)\n"
        "for dev in conf.get('p4_devices', []):\n"
        "    for prog in dev.get('p4_programs', []):\n"
        "        prog.pop('model_json_path', None)\n"
        "        prog['bfrt-config'] = f\"{abs_build}/bfrt.json\"\n"
        "        for pipe in prog.get('p4_pipelines', []):\n"
        "            pipe['context'] = f\"{abs_build}/{pipe['p4_pipeline_name']}/context.json\"\n"
        "            pipe['config']  = f\"{abs_build}/{pipe['p4_pipeline_name']}/tofino.bin\"\n"
        "            pipe['path']    = abs_build\n"
        "with open(conf_path, 'w') as f: json.dump(conf, f, indent=4)\n"
        "PYEOF\n"
        '    cp "$BUILD_DIR/$P4_PROGRAM_NAME.conf" "./$P4_PROGRAM_NAME.conf"\n'
        '    echo "------------------------------------------"\n'
        '    echo "SUCCESS: Compilation finished."\n'
        '    echo "Output directory: $PWD/$BUILD_DIR"\n'
        '    echo "------------------------------------------"\n'
        "else\n"
        '    echo "------------------------------------------"\n'
        '    echo "ERROR: Compilation failed."\n'
        '    echo "------------------------------------------"\n'
        "    exit 1\n"
        "fi\n"
    )


_DEFINES_P4 = """\
#ifndef _D1_TREE_DEFINES_
#define _D1_TREE_DEFINES_

// Constants
#define RECIRCULATE_PORT 68
#define CPU_PORT 64
#define DROP_PORT 511

#define ETHERTYPE_IPV4 0x0800
#define ETHERTYPE_BEE 0x0801
#define ETHERTYPE_MIN_BEE 0x0802

#define IP_PROTOCOLS_TCP 6
#define IP_PROTOCOLS_UDP 17

#define TABLE_SIZE 512
#define INGRESS_CTR_SIZE 65536
#define CTR_WIDTH 64
#define INGRESS_CTR_WIDTH 16

#define NUM_QUEUES 4

// Threshold for deflection (Can be tuned)
#define THRESHOLD_OCC 1000
#define THRESHOLD_OCC2 1200
#define THRESHOLD_OCC3 1100
#define THRESHOLD_OCC_IDX2 1000
#define THRESHOLD_TOT_OCC 3000
#define THRESHOLD_TOT_OCC_IDX0 4000
#define THRESHOLD_TOT_OCC_IDX1 4000
#define THRESHOLD_OCC_IDX2_2 3500
#define THRESHOLD_OCC_IDX2_3 3600

// Widths
// typedef bit<9>  PortId_t; // Defined in tna.p4
// typedef bit<16> QueueId_t; // Defined in tna.p4

#endif
"""

_PARSER_P4 = """\
#ifndef _D8_PARSER_
#define _D8_PARSER_

#include "headers.p4"

parser SwitchIngressParser(packet_in pkt,
    out header_t hdr,
    out metadata_t ig_md,
    out ingress_intrinsic_metadata_t ig_intr_md)
{
    state start {
        pkt.extract(ig_intr_md);

        ig_md.port_idx               = 0;
        ig_md.min_id                 = 0;
        ig_md.deflect                = 0;
        ig_md.local_queue_occ        = 0;
        ig_md.min_cand_occ           = 0;
        ig_md.max1                   = 0;
        ig_md.cand2_occ              = 0;
        ig_md.cand_occ_spread        = 0;
        ig_md.occ_vs                 = 0;
        ig_md.bucket_local_queue_occ = 0;
        ig_md.bucket_min_cand_occ    = 0;
        ig_md.bucket_cand2_occ       = 0;
        ig_md.bucket_cand_occ_spread = 0;
        ig_md.bucket_occ_vs_min_cand = 0;
        ig_md.bucket_occ_vs_mean_cand = 0;

        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            ETHERTYPE_IPV4 : parse_ipv4;
            ETHERTYPE_BEE  : parse_bee;
            default        : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }

    state parse_bee {
        pkt.extract(hdr.bee);
        transition accept;
    }
}

control SwitchIngressDeparser(packet_out pkt,
    inout header_t hdr,
    in metadata_t ig_md,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    in ingress_intrinsic_metadata_t ig_intr_md)
{
    apply {
        pkt.emit(hdr);
    }
}

parser SwitchEgressParser(packet_in pkt,
    out header_t hdr,
    out metadata_t eg_md,
    out egress_intrinsic_metadata_t eg_intr_md)
{
    state start {
        pkt.extract(eg_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            ETHERTYPE_IPV4 : parse_ipv4;
            ETHERTYPE_BEE  : parse_bee;
            default        : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }

    state parse_bee {
        pkt.extract(hdr.bee);
        transition accept;
    }
}

control SwitchEgressDeparser(packet_out pkt,
    inout header_t hdr,
    in metadata_t eg_md,
    in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    in egress_intrinsic_metadata_t eg_intr_md)
{
    apply {
        pkt.emit(hdr);
    }
}

#endif
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="gate_dt JSON → Tofino bfrt script + P4 spec"
    )
    ap.add_argument("model_json",  help="gate_dt JSON file")
    ap.add_argument("--out",       default=None,      help="bfrt .py output path")
    ap.add_argument("--p4-spec",   default=None,      help="P4 spec .txt output path")
    ap.add_argument("--program",   default=None,       help="P4 program name (default: d{max_depth})")
    ap.add_argument("--pipe",      default="pipe",    help="Pipeline prefix")
    ap.add_argument("--field-max", type=int, default=0xFFFF,
                    help="Max integer value of raw feature fields (default: 65535 = bit<16>)")
    ap.add_argument("--mode", choices=["per-tree", "combined"], default="per-tree",
                    help="per-tree: one table per tree (default); combined: single decision table")
    ap.add_argument("--merge-strategy", choices=["greedy", "optimal"], default="greedy",
                    help="combined mode only — greedy: fast iterative merge; optimal: QMC+set-cover minimum (default: greedy)")
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

    trees, names, base_score_int, threshold_scale, leaf_scale = load_model(args.model_json)

    if args.program is None:
        def _tree_depth(tree: dict) -> int:
            node_map = {n["node_id"]: n for n in tree["nodes"]}
            def depth(nid: int) -> int:
                n = node_map[nid]
                return 0 if n["is_leaf"] else 1 + max(depth(n["left"]), depth(n["right"]))
            return depth(0)
        max_depth = max(_tree_depth(t) for t in trees)
        args.program = f"d{max_depth}"

    depth_str = args.program[1:] if args.program[0] == "d" else args.program
    if args.out is None:
        args.out = f"bfrt{depth_str}.py"
    if args.p4_spec is None:
        args.p4_spec = f"spec{depth_str}.txt"
    if args.p4 is None and args.mode == "combined":
        args.p4 = f"{args.program}.p4"

    print(f"Loaded: {len(trees)} trees  base_score_int={base_score_int}  mode={args.mode}  program={args.program}")
    print(f"threshold_scale={threshold_scale}  leaf_scale={leaf_scale}")
    print(f"Features: {list(names.values())}")

    int_thresholds = collect_int_thresholds(trees)
    print(f"Buckets per feature: { {names[fi]: len(v)+1 for fi, v in int_thresholds.items()} }")

    encoding_tables = build_encoding_tables(int_thresholds, args.field_max)

    tree_rules = []
    for tree in trees:
        tidx  = tree["tree_index"]
        rules = extract_tree_rules(tree, int_thresholds)
        tree_rules.append((tidx, rules))
        print(f"  tree {tidx}: {len(rules)} entries")

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
        folder = f"D{depth_str.upper()}_XGBOOST"
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
            print(f"P4 spec     → {args.p4_spec}")

    else:  # per-tree
        script = emit_bfrt(encoding_tables, tree_rules, base_score_int, names, args.program, args.pipe)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(script)
        print(f"bfrt script → {args.out}")

        if args.p4_spec:
            spec = emit_p4_spec(encoding_tables, tree_rules, base_score_int, names)
            with open(args.p4_spec, "w", encoding="utf-8") as f:
                f.write(spec)
            print(f"P4 spec     → {args.p4_spec}")


if __name__ == "__main__":
    main()
