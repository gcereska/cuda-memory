#!/usr/bin/env python3
"""
plot_comparison.py — bar chart: device_malloc r1 vs r2 of all other allocators

Usage:
    python3 plot_comparison.py [csv_path] [output_png]
"""

import csv as csvlib
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "comparison_bar.png"

DISPLAY_NAMES = {
    "thread_first_fit": "Thread FF",
    "thread_best_fit":  "Thread BF",
    "warp_first_fit":   "Warp FF\n1t/pool",
    "warp_best_fit":    "Warp BF\n1t/pool",
    "warp_first_fit_32": "Warp FF\n32t/pool",
    "warp_best_fit_32":  "Warp BF\n32t/pool",
    "freelist":         "Freelist",
    "device_malloc":    "Device\nmalloc",
}

ALLOC_COLORS = {
    "thread_first_fit":  "#378ADD",
    "thread_best_fit":   "#1D9E75",
    "warp_first_fit":    "#D85A30",
    "warp_best_fit":     "#D4537E",
    "warp_first_fit_32": "#E8A87C",
    "warp_best_fit_32":  "#E8A0B8",
    "freelist":          "#BA7517",
    "device_malloc":     "#E24B4A",
}

WARP_ALLOCS = {"warp_first_fit", "warp_best_fit"}

# ── load CSV with run detection ───────────────────────────────────────────────
def load_with_runs(path):
    rows      = []
    last_iter = {}
    run_num   = {}

    with open(path, newline="") as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            key = (row["allocator"], row["test_type"])
            it  = int(row["iteration"])
            if key not in last_iter:
                last_iter[key] = -1
                run_num[key]   = 1
            elif it <= last_iter[key]:
                run_num[key] += 1
            last_iter[key] = it
            rows.append({
                "allocator":       row["allocator"],
                "test_type":       row["test_type"],
                "iteration":       it,
                "cumulative_ms":   float(row["cumulative_ms"]),
                "avg_ms_per_iter": float(row["avg_ms_per_iter"]),
                "run":             run_num[key],
            })

    return pd.DataFrame(rows)


df = load_with_runs(CSV_PATH)

# ── get final avg ms for a given allocator, test_type, run ───────────────────
def get_avg(alloc, ttype, run):
    sub = df[(df["allocator"] == alloc) & (df["test_type"] == ttype) & (df["run"] == run)]
    if sub.empty:
        return None
    return sub.loc[sub["iteration"].idxmax(), "avg_ms_per_iter"]


# ── build the series we want ──────────────────────────────────────────────────
# device_malloc r1, then r2 of everything else
# For warp, r1=1t/pool r2=32t/pool (based on benchmark loop structure)
# We want r2 of all non-device allocators = their second pass

entries = []

# device_malloc r1 (the reference bar)
for ttype in ["suite", "fuzzy"]:
    v = get_avg("device_malloc", ttype, 1)
    if v is not None:
        entries.append({"key": "device_malloc", "label": "Device\nmalloc", "test_type": ttype, "avg_ms": v})

# r2 of all other allocators
for alloc in ["thread_first_fit", "thread_best_fit", "warp_first_fit", "warp_best_fit", "freelist"]:
    for ttype in ["suite", "fuzzy"]:
        v = get_avg(alloc, ttype, 2)
        if v is None:
            continue
        # Warp r2 = 32t/pool config
        if alloc in WARP_ALLOCS:
            key   = alloc + "_32"
            label = DISPLAY_NAMES[key]
        else:
            key   = alloc
            label = DISPLAY_NAMES.get(alloc, alloc)
        entries.append({"key": key, "label": label, "test_type": ttype, "avg_ms": v})

if not entries:
    print("ERROR: No matching data found. Make sure device_malloc is uncommented and re-run the benchmark.")
    sys.exit(1)

result_df = pd.DataFrame(entries)

# ── define bar order ──────────────────────────────────────────────────────────
BAR_ORDER = [
    "device_malloc",
    "thread_first_fit",
    "thread_best_fit",
    "warp_first_fit",
    "warp_best_fit",
    "warp_first_fit_32",
    "warp_best_fit_32",
    "freelist",
]
BAR_ORDER = [b for b in BAR_ORDER if b in result_df["key"].values]

# ── plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "monospace",
    "axes.titlesize":  11,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":      150,
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, ttype in zip(axes, ["suite", "fuzzy"]):
    sub = result_df[result_df["test_type"] == ttype]
    sub = sub.set_index("key").reindex(BAR_ORDER).dropna(subset=["avg_ms"]).reset_index()

    labels = [DISPLAY_NAMES.get(k, k) for k in sub["key"]]
    values = sub["avg_ms"].tolist()
    colors = [ALLOC_COLORS.get(k, "#888") for k in sub["key"]]
    x      = np.arange(len(labels))

    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.6)

    # vertical separator after device_malloc
    if "device_malloc" in sub["key"].values:
        sep_idx = list(sub["key"]).index("device_malloc")
        ax.axvline(sep_idx + 0.5, color="#999999", linewidth=1.2, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, linespacing=1.4)
    ax.set_ylabel("avg ms / iter")
    ax.set_title(f"{'Allocator Suite' if ttype == 'suite' else 'Fuzzy Test'}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.5)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.015,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=6.5,
        )

    # speedup annotations relative to device_malloc
    dm_val = sub.loc[sub["key"] == "device_malloc", "avg_ms"].values
    if len(dm_val) > 0:
        dm = dm_val[0]
        for bar, key, val in zip(bars, sub["key"], values):
            if key == "device_malloc":
                continue
            speedup = dm / val
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{speedup:.1f}x",
                ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold",
            )

fig.suptitle(
    "Device malloc (r1) vs all allocators r2\n"
    "(warp r2 = 32t/pool config; others r2 = second pass)",
    fontsize=11, y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")