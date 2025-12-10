import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# 確保能匯入專案根目錄的模組（如有需要）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_entropy_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "image": r["image"],
                        "method": r["method"],
                        "compression_ratio": float(r["compression_ratio"]),
                        "bpp": float(r["bpp"]),
                        "time_encoding_ms": float(r["time_encoding_ms"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def plot_entropy_summary(rows, output_path: Path):
    # 聚合成每個 method 的平均
    present_methods = {r["method"] for r in rows}

    # 以「由簡到複」的方式定義合理的顯示順序
    preferred_order = ["raw", "rle", "huff", "huff_global", "huff_dpcm"]
    # 只保留在資料中實際出現的 method，其他略過
    methods = [m for m in preferred_order if m in present_methods]

    avg_ratio = []
    avg_bpp = []
    avg_time = []

    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        avg_ratio.append(sum(x["compression_ratio"] for x in mr) / len(mr))
        avg_bpp.append(sum(x["bpp"] for x in mr) / len(mr))
        avg_time.append(sum(x["time_encoding_ms"] for x in mr) / len(mr))

    x = np.arange(len(methods))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # 圖 1：平均壓縮率與 bpp
    ax1.bar(x - width / 2, avg_ratio, width, label="Compression Ratio (orig/comp)")
    ax1.set_ylabel("Compression Ratio (x)")
    ax1_2 = ax1.twinx()
    ax1_2.plot(
        x,
        avg_bpp,
        color="orange",
        marker="o",
        label="bpp (bits per pixel)",
    )
    ax1_2.set_ylabel("bpp")
    ax1.set_title("Average Compression vs Entropy Method")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # 圖 2：平均編碼時間
    ax2.bar(x, avg_time, color="green")
    ax2.set_ylabel("Encoding Time (ms)")
    ax2.set_title("Average Encoding Time vs Entropy Method")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot summary figures from entropy_results.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="entropy_results.csv",
        help="Path to entropy_results.csv (default: entropy_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/entropy_summary.png",
        help="Output plot path (default: plots/entropy_summary.png)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    rows = load_entropy_csv(csv_path)
    if not rows:
        print("No valid rows found in CSV.")
        return

    plot_entropy_summary(rows, Path(args.output))


if __name__ == "__main__":
    main()


