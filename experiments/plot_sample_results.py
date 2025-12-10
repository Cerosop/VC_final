"""Plot PSNR and upsample time for different upsampling methods.

This script reads `sample_results_clean.csv` and generates:
- A bar chart of average PSNR per upsample_method.
- A bar chart of average time_upsample_ms per upsample_method.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
# 若有設定環境變數 CLEAN_CSV_PATH 則優先使用，否則 fallback 到預設檔名
_clean_override = os.environ.get("CLEAN_CSV_PATH")
if _clean_override:
    CLEAN_CSV = Path(_clean_override).resolve()
else:
    CLEAN_CSV = ROOT / "sample_results_clean.csv"
PLOTS_DIR = ROOT / "plots"


def load_data(csv_path: Path):
    """Return dict: method -> list of (psnr, time_ms)."""
    data = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["upsample_method"]
            try:
                psnr = float(row["psnr"])
            except ValueError:
                continue
            try:
                t_up = float(row["time_upsample_ms"])
            except ValueError:
                continue
            data[method].append((psnr, t_up))
    return data


def compute_averages(data):
    """Compute average PSNR and time per method."""
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
    bars = ax.bar(x_pos, values, color="#4C72B0")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # 在柱子上標數值
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
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}. Run make_sample_results_clean first.")

    data = load_data(CLEAN_CSV)
    methods, avg_psnr, avg_time = compute_averages(data)

    # PSNR 比較圖（越高越好）
    plot_bar(
        methods,
        avg_psnr,
        ylabel="Average PSNR (dB)",
        title="Upsampling Methods: Average PSNR",
        filename="upsample_psnr.png",
    )

    # Upsample 時間比較圖（越低越好）
    plot_bar(
        methods,
        avg_time,
        ylabel="Average Upsample Time (ms)",
        title="Upsampling Methods: Average Upsample Time",
        filename="upsample_time.png",
    )


if __name__ == "__main__":
    main()
