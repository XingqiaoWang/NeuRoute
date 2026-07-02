from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DATASET_ORDER = ["GLDv2", "BigANN-100M", "BigANN-1B", "Deep1B-100M", "Deep1B-1B"]
METHOD_ORDER = ["IVF-PQ", "IVF-Flat", "OPQ+IVF-PQ", "DiskANN", "HNSW", "NeuRoute"]
COLORS = {
    "IVF-PQ": "#2f77b4",
    "IVF-Flat": "#ff8c1a",
    "OPQ+IVF-PQ": "#2ab7c9",
    "DiskANN": "#8e63bd",
    "HNSW": "#d62728",
    "NeuRoute": "#2ca02c",
}


def format_hours(value: float) -> str:
    if value < 1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def read_build_times(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            method = row["method"]
            hours = row["build_time_hours"].strip()
            if hours:
                values[dataset][method] = float(hours)
    return values


def plot_build_times(input_csv: Path, output_png: Path) -> None:
    values = read_build_times(input_csv)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    x_positions = list(range(len(DATASET_ORDER)))
    width = 0.12
    offsets = {
        method: (idx - (len(METHOD_ORDER) - 1) / 2) * width
        for idx, method in enumerate(METHOD_ORDER)
    }

    for method in METHOD_ORDER:
        xs = []
        ys = []
        for idx, dataset in enumerate(DATASET_ORDER):
            if method in values.get(dataset, {}):
                xs.append(idx + offsets[method])
                ys.append(values[dataset][method])
        if not xs:
            continue
        bars = ax.bar(
            xs,
            ys,
            width=width * 0.92,
            label=method,
            color=COLORS[method],
            edgecolor="#1f1f1f",
            linewidth=0.6,
        )
        for bar, y_value in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_value * 1.08,
                format_hours(y_value),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_yscale("log")
    ax.set_ylabel("Build time (hours, log scale)", fontsize=12, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(DATASET_ORDER, fontsize=11, fontweight="bold")
    ax.set_ylim(0.25, 60)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=True)
    ax.set_title("End-to-end index build time", fontsize=13, fontweight="bold", pad=22)

    fig.tight_layout()
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NeuRoute build-time comparison.")
    parser.add_argument("--input", type=Path, default=Path("results/build_time.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/buildtime.png"))
    args = parser.parse_args()
    plot_build_times(args.input, args.output)


if __name__ == "__main__":
    main()
