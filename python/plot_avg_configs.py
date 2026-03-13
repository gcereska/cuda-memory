#!/usr/bin/env python3
"""
plot_avg_configs.py — bar chart of avg ms/iter for every allocator configuration

Usage:
    python3 plot_avg_configs.py [csv_path]

Output saved to same directory as csv_path.
"""

import csv as csvlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(CSV_PATH)), "avg_configs_bar.png")

# (label, csv_allocator, run_number, color)
CONFIGS = [
    ("Thread FF",   "thread_first_fit", 2, "#378ADD"),
    ("Thread BF",   "thread_best_fit",  2, "#1D9E75"),
    ("Warp FF 1t",  "warp_first_fit",   4, "#D85A30"),
    ("Warp BF 1t",  "warp_best_fit",    4, "#D4537E"),
    ("Warp FF 8t",  "warp_first_fit",   5, "#E8864A"),
    ("Warp BF 8t",  "warp_best_fit",    5, "#C96090"),
    ("Warp FF 32t", "warp_first_fit",   6, "#8B5CF6"),
    ("Warp BF 32t", "warp_best_fit",    6, "#A855F7"),
    ("Freelist",    "freelist",         2, "#BA7517"),
]

# ── load CSV with run detection ───────────────────────────────────────────────
def load_with_runs(path):
    rows      = []
    last_iter = {}
    run_num   = {}
    with open(path, newline="") as f:
        for row in csvlib.DictReader(f):
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
                "avg_ms_per_iter": float(row["avg_ms_per_iter"]),
                "run":             run_num[key],
            })
    return pd.DataFrame(rows)


df = load_with_runs(CSV_PATH)

def get_avg(alloc, ttype, run):
    sub = df[(df["allocator"] == alloc) & (df["test_type"] == ttype) & (df["run"] == run)]
    if sub.empty:
        return None
    return sub.loc[sub["iteration"].idxmax(), "avg_ms_per_iter"]


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
    labels, values, colors = [], [], []

    for label, alloc, run, color in CONFIGS:
        v = get_avg(alloc, ttype, run)
        if v is None:
            print(f"WARNING [{ttype}]: no data for {label}")
            continue
        labels.append(label)
        values.append(v)
        colors.append(color)

    x    = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.015,
            f"{val:.4f} ms",
            ha="center", va="bottom", fontsize=6.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("avg ms / iter")
    ax.set_title(f"{'Allocator Suite' if ttype == 'suite' else 'Fuzzy Test'}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.5)

fig.suptitle("CUDA Shared Memory Allocator — avg ms per iteration", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")