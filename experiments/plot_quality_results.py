import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt

# 確保能匯入專案根目錄的模組（如有需要）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_quality_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "image": r["image"],
                        "quality": int(r["quality"]),
                        "entropy_method": r.get("entropy_method", "huff"),
                        "psnr": float(r["psnr"]),
                        "bpp": float(r["bpp"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def plot_rd_curves(rows, output_path: Path):
    # 依 entropy_method → quality 聚合 (平均 PSNR, bpp)
    methods = sorted({r["entropy_method"] for r in rows})
    agg = {
        m: defaultdict(list)
        for m in methods
    }  # method -> quality -> list of (bpp, psnr)

    for r in rows:
        m = r["entropy_method"]
        q = r["quality"]
        agg[m][q].append((r["bpp"], r["psnr"]))

    fig, ax = plt.subplots(figsize=(8, 6))

    for m in methods:
        qualities = sorted(agg[m].keys())
        avg_bpp = []
        avg_psnr = []
        for q in qualities:
            pairs = agg[m][q]
            if not pairs:
                continue
            bpps = [p[0] for p in pairs]
            psnrs = [p[1] for p in pairs]
            avg_bpp.append(sum(bpps) / len(bpps))
            avg_psnr.append(sum(psnrs) / len(psnrs))

        if not avg_bpp:
            continue

        ax.plot(
            avg_bpp,
            avg_psnr,
            marker="o",
            label=f"{m} (Q={','.join(str(q) for q in qualities)})",
        )

    ax.set_xlabel("Bits per Pixel (bpp)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate–Distortion Curves (PSNR vs bpp)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot PSNR–bpp curves from quality_results.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="quality_results.csv",
        help="Path to quality_results.csv (default: quality_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/quality_rd.png",
        help="Output plot path (default: plots/quality_rd.png)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    rows = load_quality_csv(csv_path)
    if not rows:
        print("No valid rows found in CSV.")
        return

    plot_rd_curves(rows, Path(args.output))


if __name__ == "__main__":
    main()


