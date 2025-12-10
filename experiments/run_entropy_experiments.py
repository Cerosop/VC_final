import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# 確保可以從專案根目錄匯入 main / config / entropy 等模組
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experiment_entropy import run_entropy_comparison  # 重用單張測試邏輯

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# 與 experiment_entropy.py 中的 methods 保持一致
ENTROPY_METHODS = ["raw", "rle", "huff", "huff_global", "huff_dpcm", "huff_dcac_shared"]


def run_entropy_experiments(image_dir: Path, output_csv: Path) -> None:
    image_dir = image_dir.resolve()
    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    )

    if not images:
        print(f"No supported images found in {image_dir}")
        return

    print(f"=== Entropy Experiments on folder: {image_dir} ===")
    print(f"Found {len(images)} images\n")

    rows = []

    for img_path in images:
        print(f"--- {img_path.name} ---")
        # 重用單張測試函式，避免多圖時出現 PSNR 不一致
        img_results = run_entropy_comparison(img_path, save_plot=False)
        for r in img_results:
            w = r.get("width", 0)
            h = r.get("height", 0)
            comp_bytes = r.get("compressed_bytes", 0.0)
            bpp = (comp_bytes * 8.0) / (w * h) if w > 0 and h > 0 else 0.0
            rows.append(
                {
                    "image": img_path.name,
                    "width": w,
                    "height": h,
                    "method": r.get("method"),
                    "psnr": r.get("psnr"),
                    "original_bytes": r.get("original_bytes", w * h * 3),
                    "compressed_bytes": comp_bytes,
                    "compression_ratio": r.get("ratio"),
                    "bpp": bpp,
                    "time_encoding_ms": r.get("time"),
                    "total_time_ms": r.get("time_total_ms", 0.0),
                }
            )
        print("-" * 60)

    # Write CSV
    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "width",
        "height",
        "method",
        "psnr",
        "original_bytes",
        "compressed_bytes",
        "compression_ratio",
        "bpp",
        "time_encoding_ms",
        "total_time_ms",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[Done] Results written to: {output_csv}")

    # Simple per-method summary
    print("\n=== Per-method summary (averages over all images) ===")
    for method in ENTROPY_METHODS:
        m_rows = [r for r in rows if r["method"] == method]
        if not m_rows:
            continue
        avg_psnr = sum(r["psnr"] for r in m_rows) / len(m_rows)
        avg_ratio = sum(r["compression_ratio"] for r in m_rows) / len(m_rows)
        avg_bpp = sum(r["bpp"] for r in m_rows) / len(m_rows)
        avg_enc = sum(r["time_encoding_ms"] for r in m_rows) / len(m_rows)

        print(
            f"  {method:>4} | "
            f"PSNR: {avg_psnr:6.2f} dB | "
            f"Ratio: {avg_ratio:6.2f}x | "
            f"bpp: {avg_bpp:5.3f} | "
            f"Enc time: {avg_enc:7.1f} ms"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run JPEG entropy experiments on a folder of images."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="Input image directory (default: images)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="entropy_results.csv",
        help="Path to output CSV file (default: entropy_results.csv in project root)",
    )
    args = parser.parse_args()

    run_entropy_experiments(Path(args.image_dir), Path(args.output_csv))


if __name__ == "__main__":
    main()


