#!/usr/bin/env python3
"""
Generate all paper plots.

Outputs, each as PDF and PNG:
  - plots/regions
  - plots/intersections_vs_final
  - plots/merge
  - plots/tcam
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator

import xgboost_to_tofino as xgb_mod
from xgboost_to_tofino import collect_int_thresholds, intersect_all_trees, merge_rules


ROOT = Path(__file__).parent
DEFAULT_DEPTHS = (4, 8, 12, 16)
DEFAULT_MODEL = "xgboost"

HALF_COL_FIGSIZE = (3.4, 2.75)
LINEWIDTH = 2.1
MARKERSIZE = 6.0
BAR_WIDTH = 2.0
AXIS_LABELSIZE = 14.0

PAPER_BLUE = "#4C78A8"
PAPER_BLUE_DARK = "#2F4B6E"
PAPER_MAGENTA = "#C05A7E"

SERIES_A = {"color": PAPER_BLUE, "linestyle": "--", "marker": "s"}
SERIES_B = {"color": PAPER_MAGENTA, "linestyle": ":", "marker": "^"}


def apply_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 13.0,
        "axes.titlesize": 13.0,
        "axes.labelsize": AXIS_LABELSIZE,
        "legend.fontsize": 12.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.95,
        "xtick.major.width": 0.95,
        "ytick.major.width": 0.95,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
    })


def compact_tick(x: float, _: float) -> str:
    if x >= 1_000_000:
        return f"{int(x / 1_000_000)}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k"
    return f"{int(x)}"


def sparse_log_tick(x: float, _: float) -> str:
    if int(round(x)) in {100, 10_000}:
        return ""
    return compact_tick(x, _)


def style_axis(ax, xlabel: str, ylabel: str, *, log_y: bool = False, y_ticks: int = 5) -> None:
    ax.set_xlabel(xlabel, fontsize=AXIS_LABELSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABELSIZE)
    ax.grid(axis="y", linestyle=":", linewidth=0.75, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=y_ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(sparse_log_tick))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks, integer=True))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(compact_tick))


def style_legend(ax, *, ncol: int = 1) -> None:
    ax.legend(
        frameon=False,
        ncol=ncol,
        handlelength=1.7,
        columnspacing=0.9,
        borderpad=0.2,
        labelspacing=0.3,
        handletextpad=0.45,
    )


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(exist_ok=True)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.16, top=0.995)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")


def xgb_stats(depth: int, xgb_dir: Path) -> dict[str, int]:
    path = xgb_dir / f"gate_dt_depth{depth}_xgboost.json"
    trees, _, base_score, _, _ = xgb_mod.load_model(str(path))
    int_thresholds = collect_int_thresholds(trees)
    tree_rules = [
        (tree["tree_index"], xgb_mod.extract_tree_rules(tree, int_thresholds))
        for tree in trees
    ]
    combined = intersect_all_trees(tree_rules, int_thresholds, base_score)
    deflect_raw = [rule for rule in combined if rule.leaf_int >= 0]
    nondeflect_raw = [rule for rule in combined if rule.leaf_int < 0]
    smaller_branch = min(deflect_raw, nondeflect_raw, key=len)
    merged = merge_rules(smaller_branch, strategy="greedy")

    return {
        "deflect": len(deflect_raw),
        "nondeflect": len(nondeflect_raw),
        "total": len(combined),
        "smaller": len(smaller_branch),
        "merged": len(merged),
    }


def collect_xgb_stats(depths: Iterable[int], xgb_dir: Path) -> dict[int, dict[str, int]]:
    print("Collecting XGBoost model stats...")
    stats_by_depth: dict[int, dict[str, int]] = {}
    for depth in depths:
        print(f"  d{depth}...", end=" ", flush=True)
        stats = xgb_stats(depth, xgb_dir)
        stats_by_depth[depth] = stats
        print(
            "total={total} deflect={deflect} non-deflect={nondeflect} "
            "pre-merge={smaller} post-merge={merged}".format(**stats)
        )
    return stats_by_depth


def plot_regions(depths: list[int], stats_by_depth: dict[int, dict[str, int]], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=HALF_COL_FIGSIZE)
    ax.plot(
        depths,
        [stats_by_depth[d]["deflect"] for d in depths],
        color=SERIES_A["color"],
        ls="-",
        marker=SERIES_A["marker"],
        lw=LINEWIDTH,
        ms=MARKERSIZE,
        label="Deflect Branch",
    )
    ax.plot(
        depths,
        [stats_by_depth[d]["nondeflect"] for d in depths],
        color=SERIES_B["color"],
        ls="--",
        marker=SERIES_B["marker"],
        lw=LINEWIDTH,
        ms=MARKERSIZE,
        label="Non-deflect Branch",
    )
    ax.set_xticks(depths)
    ax.set_xticklabels([str(depth) for depth in depths])
    style_axis(ax, "Max tree depth", "Entries", log_y=True)
    style_legend(ax)
    save_figure(fig, plots_dir / "regions.pdf")


def plot_intersections(depths: list[int], stats_by_depth: dict[int, dict[str, int]], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=HALF_COL_FIGSIZE)
    ax.plot(
        depths,
        [stats_by_depth[d]["total"] for d in depths],
        color=SERIES_A["color"],
        ls="-",
        marker=SERIES_A["marker"],
        lw=LINEWIDTH,
        ms=MARKERSIZE,
        label="All intersections",
    )
    ax.plot(
        depths,
        [stats_by_depth[d]["merged"] for d in depths],
        color=SERIES_B["color"],
        ls="--",
        marker=SERIES_B["marker"],
        lw=LINEWIDTH,
        ms=MARKERSIZE + 0.5,
        markeredgecolor=PAPER_BLUE_DARK,
        markeredgewidth=0.5,
        label="Final entries",
    )
    ax.set_xticks(depths)
    ax.set_xticklabels([str(depth) for depth in depths])
    style_axis(ax, "Max tree depth", "Entries", log_y=True)
    style_legend(ax)
    save_figure(fig, plots_dir / "intersections_vs_final.pdf")


def plot_merge(depths: list[int], stats_by_depth: dict[int, dict[str, int]], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=HALF_COL_FIGSIZE)
    ax.plot(
        depths,
        [stats_by_depth[d]["smaller"] for d in depths],
        color=SERIES_A["color"],
        ls=SERIES_A["linestyle"],
        marker=SERIES_A["marker"],
        lw=LINEWIDTH,
        ms=MARKERSIZE,
        label="Pre-merge",
        zorder=2,
    )
    ax.plot(
        depths,
        [stats_by_depth[d]["merged"] for d in depths],
        color=SERIES_B["color"],
        ls=SERIES_B["linestyle"],
        marker=SERIES_B["marker"],
        lw=LINEWIDTH + 0.1,
        ms=MARKERSIZE + 0.5,
        markeredgecolor=PAPER_BLUE_DARK,
        markeredgewidth=0.45,
        label="Post-merge",
        zorder=4,
    )
    ax.set_xticks(depths)
    ax.set_xticklabels([str(depth) for depth in depths])
    style_axis(ax, "Max tree depth", "Entries", log_y=False)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper left",
        handlelength=1.7,
        columnspacing=0.9,
        borderpad=0.2,
        labelspacing=0.3,
        handletextpad=0.45,
    )
    save_figure(fig, plots_dir / "merge.pdf")


def load_compilation_rows(csv_path: Path, model: str) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            row
            for row in reader
            if row.get("status") == "success" and row.get("model") == model
        ]


def build_series(rows: list[dict[str, str]], column: str) -> tuple[list[int], list[float]]:
    series: dict[int, float] = {}
    for row in rows:
        value = row.get(column, "").strip()
        if value:
            series[int(row["depth"])] = float(value)
    depths = sorted(series)
    return depths, [series[depth] for depth in depths]


def plot_tcam(rows: list[dict[str, str]], plots_dir: Path) -> None:
    depths, values = build_series(rows, "peak_tcam_util_pct")
    if not depths:
        raise ValueError("No successful rows found for peak_tcam_util_pct")

    fig, ax = plt.subplots(figsize=HALF_COL_FIGSIZE)
    ax.bar(
        depths,
        values,
        width=BAR_WIDTH,
        color=PAPER_BLUE,
        edgecolor=PAPER_BLUE_DARK,
        linewidth=0.8,
        hatch="///",
    )
    style_axis(ax, "Max tree depth", "Peak TCAM (%)", log_y=False)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(depth) for depth in depths])
    ax.set_ylim(0, 60.0)
    ax.yaxis.label.set_color(PAPER_BLUE_DARK)
    ax.tick_params(axis="y", colors=PAPER_BLUE_DARK)
    ax.spines["left"].set_color(PAPER_BLUE_DARK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_right = ax.twinx()
    ax_right.set_ylim(0, 10.0)
    ax_right.set_ylabel("Pipeline Stages", fontsize=AXIS_LABELSIZE)
    ax_right.yaxis.label.set_color(PAPER_MAGENTA)
    ax_right.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_right.axhline(9.0, color=PAPER_MAGENTA, linestyle="--", linewidth=1.35, alpha=0.9)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["right"].set_linewidth(1.15)
    ax_right.spines["right"].set_color(PAPER_MAGENTA)
    ax_right.tick_params(axis="y", colors=PAPER_MAGENTA, width=0.95, length=3.6)
    for tick_label in ax_right.get_yticklabels():
        tick_label.set_color(PAPER_MAGENTA)

    save_figure(fig, plots_dir / "tcam.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all paper plots")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEPTHS),
        help="Tree depths to plot (default: 4 8 12 16)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=ROOT / "plots",
        help="Output directory (default: plots/)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "compilation_summary.csv",
        help="Compilation summary CSV for the TCAM plot",
    )
    parser.add_argument(
        "--xgb-dir",
        type=Path,
        default=ROOT / "jsons" / "xgboost",
        help="Directory containing XGBoost JSON models",
    )
    parser.add_argument(
        "--tcam-model",
        default=DEFAULT_MODEL,
        help="Model name to read from the CSV for the TCAM plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depths = sorted(args.depths)

    apply_rcparams()
    stats_by_depth = collect_xgb_stats(depths, args.xgb_dir)
    plot_regions(depths, stats_by_depth, args.plots_dir)
    plot_intersections(depths, stats_by_depth, args.plots_dir)
    plot_merge(depths, stats_by_depth, args.plots_dir)

    rows = load_compilation_rows(args.csv, args.tcam_model)
    plot_tcam(rows, args.plots_dir)


if __name__ == "__main__":
    main()
