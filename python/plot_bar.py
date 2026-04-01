#!/usr/bin/env python3
"""
plot_barchart.py — bar chart of average ms/iteration per allocator

Usage:
    python3 plot_barchart.py [csv_path] [output_dir]

Reads the CSV produced by benchmark_all_csv_modified and plots grouped
bar charts (suite vs fuzzy) showing avg ms/iter for each allocator.
Uses only the last data point of round 2 (warmed-up run) for each allocator.
"""

import csv as csvlib
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "plots"
os.makedirs(OUT_DIR, exist_ok=True)

WARP_ALLOCS = {"warp_first_fit", "warp_best_fit"}

# ── Display names and colors ──
CATEGORY_DISPLAY = {
    "thread_first_fit":       "Thread\nFirst-Fit",
    "thread_best_fit":        "Thread\nBest-Fit",
    "warp_first_fit 1t/pool": "Warp FF\n1t/pool",
    "warp_best_fit 1t/pool":  "Warp BF\n1t/pool",
    "freelist":               "Freelist",
    "global_first_fit":       "Global\nFirst-Fit",
    "global_best_fit":        "Global\nBest-Fit",
    "device_malloc":          "Device\nmalloc",
}

CATEGORY_COLORS = {
    "thread_first_fit":       "#378ADD",
    "thread_best_fit":        "#1D9E75",
    "warp_first_fit 1t/pool": "#D85A30",
    "warp_best_fit 1t/pool":  "#D4537E",
    "freelist":               "#BA7517",
    "global_first_fit":       "#8B5CF6",
    "global_best_fit":        "#E11D48",
    "device_malloc":          "#6B7280",
}

# Display order left to right
CATEGORY_ORDER = [
    "thread_first_fit",
    "thread_best_fit",
    "warp_first_fit 1t/pool",
    "warp_best_fit 1t/pool",
    "freelist",
    "global_first_fit",
    "global_best_fit",
    "device_malloc",
]

# Categories to exclude (set empty to include everything)
EXCLUDE_CATS = {"warp_first_fit 8t/pool", "warp_best_fit 8t/pool",
                "warp_first_fit 32t/pool", "warp_best_fit 32t/pool"}


def load_avg_ms(path):
    """
    Parse the CSV, detect round boundaries (iteration resets),
    and return the avg_ms_per_iter from the LAST row of round 2
    for each (category, test_type).
    """
    rows      = []
    last_iter = {}
    run_num   = {}

    with open(path, newline="") as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            alloc = row["allocator"]
            key   = (alloc, row["test_type"])
            it    = int(row["iteration"])

            if key not in last_iter:
                last_iter[key] = -1
                run_num[key]   = 1
            elif it <= last_iter[key]:
                run_num[key] += 1

            last_iter[key] = it
            rows.append({
                "allocator":      alloc,
                "test_type":      row["test_type"],
                "iteration":      it,
                "avg_ms_per_iter": float(row["avg_ms_per_iter"]),
                "raw_run":        run_num[key],
            })

    # Assign category (tag warp variants with thread count)
    def get_category(r):
        if r["allocator"] in WARP_ALLOCS:
            # odd raw_run = 1t/pool, even = 8t/pool (within each round pair)
            cfg = "1t/pool" if r["raw_run"] % 2 == 1 else "8t/pool"
            return f"{r['allocator']} {cfg}"
        return r["allocator"]

    def get_run(r):
        if r["allocator"] in WARP_ALLOCS:
            return (r["raw_run"] + 1) // 2
        return r["raw_run"]

    for r in rows:
        r["category"] = get_category(r)
        r["run"]      = get_run(r)

    # Keep only round 2 (warmed up), exclude unwanted categories
    r2 = [r for r in rows
           if r["run"] == 2
           and r["category"] not in EXCLUDE_CATS]

    # For each (category, test_type), take the row with the highest iteration
    # (that's the final cumulative avg)
    best = {}
    for r in r2:
        key = (r["category"], r["test_type"])
        if key not in best or r["iteration"] > best[key]["iteration"]:
            best[key] = r

    return best


def plot_grouped_bars(data, filename):
    cats_present = [c for c in CATEGORY_ORDER if any(
        (c, tt) in data for tt in ["suite", "fuzzy"])]

    if not cats_present:
        print("No data to plot!")
        return

    test_types = ["suite", "fuzzy"]
    x = np.arange(len(cats_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, tt in enumerate(test_types):
        vals = []
        for cat in cats_present:
            key = (cat, tt)
            vals.append(data[key]["avg_ms_per_iter"] if key in data else 0)
        bars = ax.bar(x + i * width, vals, width,
                      label=tt.capitalize(),
                      color=[CATEGORY_COLORS.get(c, "#888") for c in cats_present],
                      alpha=1.0 if i == 0 else 0.55,
                      edgecolor="white", linewidth=0.5)
        # Value labels on top of each bar
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([CATEGORY_DISPLAY.get(c, c) for c in cats_present],
                       fontsize=8)
    ax.set_ylabel("Avg ms / iteration")
    ax.set_title("Allocator Performance — Avg ms/iter (Round 2, warmed up)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved -> {out}")
    plt.close()


def plot_separate_bars(data, filename):
    """Two subplots side by side: one for suite, one for fuzzy."""
    cats_present = [c for c in CATEGORY_ORDER if any(
        (c, tt) in data for tt in ["suite", "fuzzy"])]

    if not cats_present:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, tt in zip(axes, ["suite", "fuzzy"]):
        vals   = []
        colors = []
        labels = []
        for cat in cats_present:
            key = (cat, tt)
            vals.append(data[key]["avg_ms_per_iter"] if key in data else 0)
            colors.append(CATEGORY_COLORS.get(cat, "#888"))
            labels.append(CATEGORY_DISPLAY.get(cat, cat))

        x = np.arange(len(cats_present))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Avg ms / iteration")
        ax.set_title(f"{tt.capitalize()} Tests", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

    fig.suptitle("Allocator Performance — Avg ms/iter (Round 2, warmed up)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved -> {out}")
    plt.close()


# ── Main ──
data = load_avg_ms(CSV_PATH)

print("Data loaded (round 2 final avg ms/iter):")
for (cat, tt), r in sorted(data.items()):
    print(f"  {cat:30s} / {tt:5s}: {r['avg_ms_per_iter']:.4f} ms/iter")

print("\nGenerating plots...")
plot_grouped_bars(data, "10_barchart_grouped.png")
plot_separate_bars(data, "11_barchart_separate.png")

print(f"\nDone. Saved to: {OUT_DIR}/")