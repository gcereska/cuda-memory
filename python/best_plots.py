#!/usr/bin/env python3
"""
bar_charts_by_suite.py — 4 bar charts (suite / fuzzy / linalg / exhaustive)
showing avg ms/iter per allocator from round 2 (warmed-up) data.

Usage:
    python3 bar_charts_by_suite.py [csv_path] [output_dir]

Outputs:
    plots/10_bar_suite.png
    plots/11_bar_fuzzy.png
    plots/12_bar_linalg.png
    plots/13_bar_exhaustive.png
"""

import csv as csvlib
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── warp pool detection ──────────────────────────────────────────────
WARP_ALLOCS = {"warp_first_fit", "warp_best_fit"}

# ── display config ───────────────────────────────────────────────────
CATEGORY_COLORS = {
    "thread_first_fit":        "#378ADD",
    "thread_best_fit":         "#1D9E75",
    "warp_first_fit 1t/pool":  "#D85A30",
    "warp_best_fit 1t/pool":   "#D4537E",
    "warp_first_fit 8t/pool":  "#E8963E",
    "warp_best_fit 8t/pool":   "#A85CA0",
    "freelist":                "#BA7517",
    "global_first_fit":        "#6B8E23",
    "global_best_fit":         "#4682B4",
    "device_malloc":           "#888888",
}

CATEGORY_DISPLAY = {
    "thread_first_fit":        "Thread FF",
    "thread_best_fit":         "Thread BF",
    "warp_first_fit 1t/pool":  "Warp FF\n1t/pool",
    "warp_best_fit 1t/pool":   "Warp BF\n1t/pool",
    "warp_first_fit 8t/pool":  "Warp FF\n8t/pool",
    "warp_best_fit 8t/pool":   "Warp BF\n8t/pool",
    "freelist":                "Freelist",
    "global_first_fit":        "Global FF",
    "global_best_fit":         "Global BF",
    "device_malloc":           "Device\nmalloc",
}

# bar ordering left to right
CATEGORY_ORDER = [
    "thread_first_fit",
    "thread_best_fit",
    "warp_first_fit 1t/pool",
    "warp_best_fit 1t/pool",
    "warp_first_fit 8t/pool",
    "warp_best_fit 8t/pool",
    "freelist",
    "global_first_fit",
    "global_best_fit",
    "device_malloc",
]

SUITE_TITLES = {
    "suite":      "Allocator Test Suite",
    "fuzzy":      "Fuzzy Test",
    "linalg":     "Linear Algebra Suite",
    "exhaustive": "Exhaustive Capacity Test (optin smem)",
}

SUITE_FILES = {
    "suite":      "10_bar_suite.png",
    "fuzzy":      "11_bar_fuzzy.png",
    "linalg":     "12_bar_linalg.png",
    "exhaustive": "13_bar_exhaustive.png",
}

plt.rcParams.update({
    "font.family":     "monospace",
    "axes.titlesize":  11,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":      150,
})


# ── load CSV and extract round-2 final avg_ms_per_iter ───────────────
def load_r2_averages(path):
    """
    Returns dict:  { (category, test_type) : avg_ms_per_iter }
    using the LAST sample row from round 2 for each allocator/test combo.
    """
    # first pass: split into runs using the iteration-reset trick
    rows      = []
    last_iter = {}
    run_num   = {}

    with open(path, newline="") as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            alloc = row["allocator"]
            ttype = row["test_type"]
            key   = (alloc, ttype)
            it    = int(row["iteration"])

            if key not in last_iter:
                last_iter[key] = -1
                run_num[key]   = 1
            elif it <= last_iter[key]:
                run_num[key] += 1

            last_iter[key] = it
            rows.append({
                "allocator":      alloc,
                "test_type":      ttype,
                "iteration":      it,
                "avg_ms_per_iter": float(row["avg_ms_per_iter"]),
                "raw_run":        run_num[key],
            })

    # assign category (annotate warp with thread config)
    warp_cfg_counter = {}  # (alloc, test_type) -> sequential config index
    prev_run         = {}

    for r in rows:
        alloc = r["allocator"]
        ttype = r["test_type"]
        key   = (alloc, ttype)
        raw   = r["raw_run"]

        if alloc in WARP_ALLOCS:
            if key not in prev_run:
                prev_run[key] = raw
                warp_cfg_counter[key] = 0
            elif raw != prev_run[key]:
                warp_cfg_counter[key] += 1
                prev_run[key] = raw

            cfg_idx = warp_cfg_counter[key]
            # configs cycle: 1t, 8t  (32t commented out)
            cfgs = ["1t/pool", "8t/pool"]
            cfg  = cfgs[cfg_idx % len(cfgs)]
            r["category"] = f"{alloc} {cfg}"

            # round within this config: odd raw_runs → r1, even → r2
            # actually: for each config, first occurrence = r1, second = r2
            r["run"] = 1 + cfg_idx // len(cfgs)
        else:
            r["category"] = alloc
            r["run"]      = raw

    # keep only round 2, take the last row (highest iteration) per combo
    result = {}
    for r in rows:
        if r["run"] != 2:
            continue
        key = (r["category"], r["test_type"])
        # keep the row with the highest iteration
        if key not in result or r["iteration"] > result[key]["iteration"]:
            result[key] = r

    return {k: v["avg_ms_per_iter"] for k, v in result.items()}


# ── plotting ─────────────────────────────────────────────────────────
def plot_bar(data, test_type, title, filename):
    # collect categories present for this test_type
    cats = [c for c in CATEGORY_ORDER if (c, test_type) in data]
    if not cats:
        print(f"  SKIP {test_type}: no round-2 data")
        return

    vals   = [data[(c, test_type)] for c in cats]
    colors = [CATEGORY_COLORS.get(c, "#888") for c in cats]
    labels = [CATEGORY_DISPLAY.get(c, c) for c in cats]

    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 1.1), 5))

    x    = np.arange(len(cats))
    bars = ax.bar(x, vals, color=colors, width=0.65, edgecolor="white",
                  linewidth=0.6, zorder=3)

    # value labels on top of each bar
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel("avg ms / iter")
    ax.set_title(f"{title}  —  avg ms/iter (round 2, 100 iters)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # if device_malloc is a massive outlier, note the scale
    if vals and max(vals) / (sorted(vals)[len(vals)//2] or 1) > 10:
        ax.set_yscale("log")
        ax.set_ylabel("avg ms / iter  (log scale)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved -> {out}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────
data = load_r2_averages(CSV_PATH)

print("Round-2 averages loaded:")
for (cat, ttype), avg in sorted(data.items()):
    print(f"  {cat:30s} / {ttype:12s} : {avg:.4f} ms/iter")

print("\nGenerating bar charts...")
for ttype in ["suite", "fuzzy", "linalg", "exhaustive"]:
    plot_bar(data, ttype, SUITE_TITLES[ttype], SUITE_FILES[ttype])

print(f"\nDone. Saved to: {OUT_DIR}/")