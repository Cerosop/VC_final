"""Plot PSNR and downsample time for different downsampling methods.

This script reads `sample_results_downsample.csv` and generates:
- A bar chart of average PSNR per downsample_method.
- A bar chart of average time_downsample_ms per downsample_method.
"""

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "sample_results_downsample.csv"
PLOTS_DIR = ROOT / "plots"


def load_data(csv_path: Path):
    """Return dict: method -> list of (psnr, time_ms)."""
    data = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["downsample_method"]
            try:
                psnr = float(row["psnr"])
            except ValueError:
                continue
            try:
                t_down = float(row["time_downsample_ms"])
            except ValueError:
                continue
            data[method].append((psnr, t_down))
    return data


def compute_averages(data):
    methods = []
    avg_psnr = []
    avg_time = []
    for method, vals in data.items():
        if not vals:
            continue
        s_psnr = sum(v[0] for v in vals)
        s_time = sum(v[1] for v in vals)
        n = len(vals)
        methods.append(method)
        avg_psnr.append(s_psnr / n)
        avg_time.append(s_time / n)
    return methods, avg_psnr, avg_time


def plot_bar(x_labels, values, ylabel, title, filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = range(len(x_labels))
    bars = ax.bar(x_pos, values, color="#55A868")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Downsample CSV not found: {CSV_PATH}.")

    data = load_data(CSV_PATH)
    methods, avg_psnr, avg_time = compute_averages(data)

    # PSNR 比較圖
    plot_bar(
        methods,
        avg_psnr,
        ylabel="Average PSNR (dB)",
        title="Downsampling Methods: Average PSNR",
        filename="downsample_psnr.png",
    )

    # Downsample 時間比較圖
    plot_bar(
        methods,
        avg_time,
        ylabel="Average Downsample Time (ms)",
        title="Downsampling Methods: Average Downsample Time",
        filename="downsample_time.png",
    )


if __name__ == "__main__":
    main()
