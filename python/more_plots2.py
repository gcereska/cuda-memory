#!/usr/bin/env python3
"""
cumulative_no_malloc_no8t.py — cumulative ms, no device malloc, no 8t/pool warp

Usage:
    python3 cumulative_no_malloc_no8t.py [csv_path] [output_dir]

Outputs:
    plots/08_cumulative_all_runs_no8t.png  — r1 dotted, r2 solid
    plots/09_cumulative_r2_only_no8t.png   — r2 only
"""

import csv as csvlib
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "plots"
os.makedirs(OUT_DIR, exist_ok=True)

WARP_ALLOCS  = {"warp_first_fit", "warp_best_fit"}
EXCLUDE_CATS = {"warp_first_fit 8t/pool", "warp_best_fit 8t/pool"}

CATEGORY_COLORS = {
    "thread_first_fit":       "#378ADD",
    "thread_best_fit":        "#1D9E75",
    "warp_first_fit 1t/pool": "#D85A30",
    "warp_best_fit 1t/pool":  "#D4537E",
    "freelist":               "#BA7517",
}

CATEGORY_DISPLAY = {
    "thread_first_fit":       "Thread FF",
    "thread_best_fit":        "Thread BF",
    "warp_first_fit 1t/pool": "Warp FF 1t/pool",
    "warp_best_fit 1t/pool":  "Warp BF 1t/pool",
    "freelist":               "Freelist",
}

CATEGORY_ORDER = list(CATEGORY_DISPLAY.keys())

plt.rcParams.update({
    "font.family":     "monospace",
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":      150,
})


def load_and_split(path):
    rows      = []
    last_iter = {}
    run_num   = {}

    with open(path, newline="") as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            alloc = row["allocator"]
            if alloc == "device_malloc":
                continue
            key = (alloc, row["test_type"])
            it  = int(row["iteration"])

            if key not in last_iter:
                last_iter[key] = -1
                run_num[key]   = 1
            elif it <= last_iter[key]:
                run_num[key] += 1

            last_iter[key] = it
            rows.append({
                "allocator":     alloc,
                "test_type":     row["test_type"],
                "iteration":     it,
                "cumulative_ms": float(row["cumulative_ms"]),
                "raw_run":       run_num[key],
            })

    df = pd.DataFrame(rows)

    def get_category(row):
        if row["allocator"] in WARP_ALLOCS:
            cfg = "1t/pool" if row["raw_run"] % 2 == 1 else "8t/pool"
            return f"{row['allocator']} {cfg}"
        return row["allocator"]

    def get_run(row):
        if row["allocator"] in WARP_ALLOCS:
            return (row["raw_run"] + 1) // 2
        return row["raw_run"]

    df["category"] = df.apply(get_category, axis=1)
    df["run"]      = df.apply(get_run, axis=1)

    # drop 8t/pool categories
    df = df[~df["category"].isin(EXCLUDE_CATS)]

    # normalize each run to start at 0
    normalized = []
    for (cat, ttype, run), g in df.groupby(["category", "test_type", "run"]):
        g = g.sort_values("iteration").copy()
        g["cumulative_ms"] -= g["cumulative_ms"].iloc[0]
        normalized.append(g)

    return pd.concat(normalized).reset_index(drop=True)


def cat_order(cat):
    return CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 99


def plot_cumulative(df, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, ttype in zip(axes, ["suite", "fuzzy"]):
        sub = df[df["test_type"] == ttype]
        combos = (
            sub[["category", "run"]].drop_duplicates()
            .sort_values(["category", "run"],
                         key=lambda c: c.map(cat_order) if c.name == "category" else c)
        )

        for _, info in combos.iterrows():
            cat  = info["category"]
            run  = int(info["run"])
            rows = sub[(sub["category"] == cat) & (sub["run"] == run)].sort_values("iteration")

            ax.plot(
                rows["iteration"], rows["cumulative_ms"],
                color=CATEGORY_COLORS.get(cat, "#888"),
                linestyle=":" if run == 1 else "-",
                linewidth=1.6,
                label=f"{CATEGORY_DISPLAY.get(cat, cat)} r{run}",
                alpha=0.9,
            )

        ax.set_xlabel("iteration")
        ax.set_ylabel("cumulative ms")
        ax.set_title(ttype)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle="--", alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=8, loc="upper left", ncol=2,
                  framealpha=0.8, edgecolor="none")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved -> {out}")
    plt.close()


df_all = load_and_split(CSV_PATH)

print("Categories and runs detected:")
for (cat, ttype), g in df_all.groupby(["category", "test_type"]):
    print(f"  {cat:30s} / {ttype}: runs {sorted(g['run'].unique())}")

df_r2 = df_all[df_all["run"] == 2]

print("\nGenerating plots...")
plot_cumulative(
    df_all,
    "Cumulative ms — 1t/pool warp only, r1 dotted / r2 solid (no device malloc)",
    "08_cumulative_all_runs_no8t.png",
)
plot_cumulative(
    df_r2,
    "Cumulative ms — 1t/pool warp only, r2 only (no device malloc)",
    "09_cumulative_r2_only_no8t.png",
)

print(f"\nDone. Saved to: {OUT_DIR}/")