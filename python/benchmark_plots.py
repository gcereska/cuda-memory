#!/usr/bin/env python3
"""
benchmark_plots.py — generate all benchmark charts from benchmark_results.csv

Usage:
    python3 benchmark_plots.py [csv_path] [output_dir]

Outputs (in output_dir, default ./plots):
    01_bar_avg_ms.png        — avg ms/iter per allocator+run, suite vs fuzzy
    02_bar_total_ms.png      — total ms per allocator+run, suite vs fuzzy
    03_window_lines.png      — avg ms per 5-iter window over time
    04_cumulative_linear.py  — cumulative ms, linear scale
    05_cumulative_log.png    — cumulative ms, log scale
"""

import csv as csvlib
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── config ───────────────────────────────────────────────────────────────────
CSV_PATH    = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
OUT_DIR     = sys.argv[2] if len(sys.argv) > 2 else "plots"

DISPLAY_NAMES = {
    "thread_first_fit": "Thread FF",
    "thread_best_fit":  "Thread BF",
    "warp_first_fit":   "Warp FF",
    "warp_best_fit":    "Warp BF",
    "device_malloc":    "Device malloc",
    "freelist":         "Freelist",
}

WARP_ALLOCS = {"warp_first_fit", "warp_best_fit"}

ALLOC_COLORS = {
    "thread_first_fit": "#378ADD",
    "thread_best_fit":  "#1D9E75",
    "warp_first_fit":   "#D85A30",
    "warp_best_fit":    "#D4537E",
    "device_malloc":    "#E24B4A",
    "freelist":         "#BA7517",
}

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":     "monospace",
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi":      150,
})


# ── load CSV, assign run numbers ─────────────────────────────────────────────
def load_with_runs(path):
    """
    Reads the CSV in file order. Detects a new run for (allocator, test_type)
    whenever the iteration counter resets back to a value <= the previous one.
    """
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

    df = pd.DataFrame(rows)

    # Human-readable label per row
    # Warp: run1=1t/pool, run2=8t/pool, run3=1t/pool(repeat), run4=8t/pool(repeat)
    def make_label(r):
        name = DISPLAY_NAMES.get(r["allocator"], r["allocator"])
        run  = r["run"]
        if r["allocator"] in WARP_ALLOCS:
            config = "1t/pool" if run % 2 == 1 else "8t/pool"
            rep    = (run + 1) // 2
            return f"{name}\n{config} r{rep}"
        return f"{name}\nr{run}"

    df["label"] = df.apply(make_label, axis=1)
    return df


df = load_with_runs(CSV_PATH)
print(f"Loaded {len(df)} rows")
print(f"Allocators : {sorted(df['allocator'].unique())}")
print(f"Runs found : {df.groupby('allocator')['run'].max().to_dict()}")


# ── helpers ───────────────────────────────────────────────────────────────────
def savefig(name):
    p = os.path.join(OUT_DIR, name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  saved → {p}")
    plt.close()


def get_summary(df):
    """Last row per (allocator, test_type, run) = cumulative total and final avg."""
    idx = df.groupby(["allocator", "test_type", "run", "label"])["iteration"].idxmax()
    g   = df.loc[idx].copy()
    g["total_ms"] = g["cumulative_ms"]
    g["avg_ms"]   = g["avg_ms_per_iter"]
    return g


summary = get_summary(df)


def alloc_order(alloc):
    order = list(DISPLAY_NAMES.keys())
    return order.index(alloc) if alloc in order else len(order)


def legend_handles():
    handles, labels = [], []
    for a, name in DISPLAY_NAMES.items():
        if a in df["allocator"].values:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=ALLOC_COLORS.get(a, "#888")))
            labels.append(name)
    return handles, labels


# ── charts 1 & 2: bar charts ─────────────────────────────────────────────────
def bar_chart(metric, ylabel, suptitle, filename):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for ax, ttype in zip(axes, ["suite", "fuzzy"]):
        sub = summary[summary["test_type"] == ttype].copy()
        sub = sub.sort_values(
            ["allocator", "run"],
            key=lambda col: col.map(alloc_order) if col.name == "allocator" else col,
        )

        labels = sub["label"].tolist()
        values = sub[metric].tolist()
        colors = [ALLOC_COLORS.get(a, "#888") for a in sub["allocator"]]
        x      = np.arange(len(labels))

        bars = ax.bar(x, values, color=colors, width=0.6,
                      edgecolor="white", linewidth=0.5)

        # run separators — vertical lines between allocator groups
        prev_alloc = None
        for i, alloc in enumerate(sub["allocator"]):
            if prev_alloc is not None and alloc != prev_alloc:
                ax.axvline(i - 0.5, color="#cccccc", linewidth=0.8, linestyle="--")
            prev_alloc = alloc

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, linespacing=1.3)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ttype}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.5)

        # value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.012,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=5.5,
            )

    h, l = legend_handles()
    fig.legend(h, l, loc="upper right", fontsize=8, ncol=3,
               framealpha=0.8, edgecolor="none")
    fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout()
    savefig(filename)


print("Generating bar charts...")
bar_chart("avg_ms",   "avg ms / iter", "Average ms per iteration — by allocator and run",
          "01_bar_avg_ms.png")
bar_chart("total_ms", "total ms",      "Total ms — by allocator and run",
          "02_bar_total_ms.png")


# ── chart 3: per-window line chart ───────────────────────────────────────────
def compute_windows(df, alloc, ttype, run):
    sub = (
        df[(df["allocator"] == alloc) & (df["test_type"] == ttype) & (df["run"] == run)]
        .sort_values("iteration")
    )
    wins     = []
    prev_cum = 0.0
    prev_it  = 0
    for _, row in sub.iterrows():
        it  = row["iteration"]
        cum = row["cumulative_ms"]
        n   = it - prev_it
        if n > 0:
            wins.append({"label": f"{prev_it + 1}-{it}", "avg_ms": (cum - prev_cum) / n})
        prev_cum = cum
        prev_it  = it
    return wins


print("Generating window line chart...")
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

for ax, ttype in zip(axes, ["suite", "fuzzy"]):
    combos = (
        summary[summary["test_type"] == ttype][["allocator", "run", "label"]]
        .drop_duplicates()
        .sort_values(["allocator", "run"],
                     key=lambda c: c.map(alloc_order) if c.name == "allocator" else c)
    )

    all_xlabels = None
    for _, info in combos.iterrows():
        alloc = info["allocator"]
        run   = info["run"]
        label = info["label"].replace("\n", " ")
        wins  = compute_windows(df, alloc, ttype, run)
        if not wins:
            continue

        xs  = list(range(len(wins)))
        ys  = [w["avg_ms"] for w in wins]
        xl  = [w["label"]  for w in wins]
        if all_xlabels is None:
            all_xlabels = xl

        col = ALLOC_COLORS.get(alloc, "#888")
        ls  = "-" if run % 2 == 1 else "--"
        lw  = 1.4 if alloc != "device_malloc" else 2.0
        ax.plot(xs, ys, color=col, linestyle=ls, linewidth=lw,
                label=label, alpha=0.88)

    if all_xlabels:
        step = max(1, len(all_xlabels) // 12)
        ax.set_xticks(range(0, len(all_xlabels), step))
        ax.set_xticklabels(
            [all_xlabels[i] for i in range(0, len(all_xlabels), step)],
            rotation=45, ha="right",
        )

    ax.set_ylabel("avg ms / iter")
    ax.set_xlabel("iteration window")
    ax.set_title(f"{ttype} — avg ms per 5-iter window")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=6, loc="upper left", ncol=2,
              framealpha=0.75, edgecolor="none")

fig.suptitle("Per-window avg time — all allocators and runs", fontsize=11)
plt.tight_layout()
savefig("03_window_lines.png")


# ── charts 4 & 5: cumulative (linear + log) ──────────────────────────────────
print("Generating cumulative charts...")

for scale, fname, scale_label in [
    ("linear", "04_cumulative_linear.png", "linear scale"),
    ("log",    "05_cumulative_log.png",    "log scale"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for ax, ttype in zip(axes, ["suite", "fuzzy"]):
        combos = (
            df[df["test_type"] == ttype][["allocator", "run", "label"]]
            .drop_duplicates()
            .sort_values(["allocator", "run"],
                         key=lambda c: c.map(alloc_order) if c.name == "allocator" else c)
        )

        for _, info in combos.iterrows():
            alloc = info["allocator"]
            run   = info["run"]
            label = info["label"].replace("\n", " ")
            sub   = (
                df[(df["allocator"] == alloc) & (df["test_type"] == ttype) & (df["run"] == run)]
                .sort_values("iteration")
            )
            col = ALLOC_COLORS.get(alloc, "#888")
            ls  = "-" if run % 2 == 1 else "--"
            lw  = 1.4 if alloc != "device_malloc" else 2.0
            ax.plot(sub["iteration"], sub["cumulative_ms"],
                    color=col, linestyle=ls, linewidth=lw,
                    label=label, alpha=0.88)

        ax.set_yscale(scale)
        ax.set_xlabel("iteration")
        ax.set_ylabel("cumulative ms")
        ax.set_title(f"{ttype}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle="--", alpha=0.3, linewidth=0.5, which="both")
        ax.legend(fontsize=6, loc="upper left", ncol=2,
                  framealpha=0.75, edgecolor="none")

    fig.suptitle(f"Cumulative time — all allocators and runs ({scale_label})", fontsize=11)
    plt.tight_layout()
    savefig(fname)

print(f"\nAll done. Plots saved to: {OUT_DIR}/")
