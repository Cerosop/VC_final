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

import config
import main as jpeg_engine


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_and_crop_image(img_path: Path) -> np.ndarray:
    """Load image as RGB and crop to multiples of 16 (same邏輯 as main.process_image)."""
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)
    h, w, _ = img_arr.shape
    h_new, w_new = (h // 16) * 16, (w // 16) * 16
    if h_new != h or w_new != w:
        img_arr = img_arr[:h_new, :w_new, :]
    return img_arr


def _scale_qtable(q: np.ndarray, quality: int) -> np.ndarray:
    """
    JPEG style quality scaling.
    quality: 1 (最差, 壓得最兇) ~ 100 (最佳, 接近無失真)
    """
    quality = max(1, min(quality, 100))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    q_scaled = (q.astype(np.float64) * scale + 50) / 100.0
    q_scaled = np.floor(q_scaled)
    q_scaled[q_scaled < 1] = 1
    q_scaled[q_scaled > 255] = 255
    return q_scaled.astype(np.int32)


def run_quality_experiments(
    image_dir: Path, qualities: list[int], entropy_method: str, output_csv: Path
) -> None:
    image_dir = image_dir.resolve()
    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    )

    if not images:
        print(f"No supported images found in {image_dir}")
        return

    print(f"=== Quality Experiments on folder: {image_dir} ===")
    print(f"Qualities: {qualities}")
    print(f"Entropy method: {entropy_method}")
    print(f"Found {len(images)} images\n")

    # 保存原始量化表，實驗完再還原
    orig_QY = config.Q_Y.copy()
    orig_QC = config.Q_C.copy()

    rows = []

    try:
        for img_path in images:
            print(f"--- {img_path.name} ---")
            img_arr = _load_and_crop_image(img_path)
            h, w, _ = img_arr.shape

            for q in qualities:
                # 依 quality 產生新的量化表，並暫時覆寫到 config
                config.Q_Y = _scale_qtable(orig_QY, q)
                config.Q_C = _scale_qtable(orig_QC, q)

                recon_img, stats = jpeg_engine.jpeg_pipeline(
                    img_arr,
                    dct_method="baseline",
                    idct_method="baseline",
                    downsample_method="baseline",
                    upsample_method="baseline",
                    entropy_method=entropy_method,
                )

                psnr = jpeg_engine.utils.calculate_psnr(img_arr, recon_img)

                orig_bytes = stats.get("original_bytes", h * w * 3)
                comp_bytes = stats.get("compressed_bytes", 0.0)
                ratio = stats.get("compression_ratio", 0.0)
                enc_time = stats.get("time_encoding", 0.0)
                total_time = stats.get("total_time", 0.0)

                bpp = (comp_bytes * 8.0) / (h * w) if h > 0 and w > 0 else 0.0

                rows.append(
                    {
                        "image": img_path.name,
                        "width": w,
                        "height": h,
                        "quality": q,
                        "entropy_method": entropy_method,
                        "psnr": psnr,
                        "original_bytes": orig_bytes,
                        "compressed_bytes": comp_bytes,
                        "compression_ratio": ratio,
                        "bpp": bpp,
                        "time_encoding_ms": enc_time,
                        "total_time_ms": total_time,
                    }
                )

                print(
                    f"  [Q={q}] "
                    f"PSNR: {psnr:.2f} dB | "
                    f"Ratio: {ratio:.2f}x | "
                    f"bpp: {bpp:.3f} | "
                    f"Enc: {enc_time:.1f} ms | "
                    f"Total: {total_time:.1f} ms"
                )

            print("-" * 60)
    finally:
        # 還原原本的量化表，避免影響其他程式
        config.Q_Y = orig_QY
        config.Q_C = orig_QC

    # 寫入 CSV
    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "width",
        "height",
        "quality",
        "entropy_method",
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

    # 簡單 summary：各 quality 的平均
    print("\n=== Per-quality summary (averages over all images) ===")
    for q in qualities:
        q_rows = [r for r in rows if r["quality"] == q]
        if not q_rows:
            continue
        avg_psnr = sum(r["psnr"] for r in q_rows) / len(q_rows)
        avg_ratio = sum(r["compression_ratio"] for r in q_rows) / len(q_rows)
        avg_bpp = sum(r["bpp"] for r in q_rows) / len(q_rows)

        print(
            f"  Q={q:3d} | "
            f"PSNR: {avg_psnr:6.2f} dB | "
            f"Ratio: {avg_ratio:6.2f}x | "
            f"bpp: {avg_bpp:5.3f}"
        )


def _parse_qualities(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    qs: list[int] = []
    for p in parts:
        try:
            qs.append(int(p))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid quality value: {p!r}")
    if not qs:
        raise argparse.ArgumentTypeError("Quality list cannot be empty")
    return qs


def main():
    parser = argparse.ArgumentParser(
        description="Run JPEG quality (quantization) experiments on a folder of images."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="Input image directory (default: images)",
    )
    parser.add_argument(
        "--qualities",
        type=_parse_qualities,
        default=[30, 50, 70, 90],
        help="Comma-separated quality values, e.g. '30,50,70,90' (default: 30,50,70,90)",
    )
    parser.add_argument(
        "--entropy_method",
        type=str,
        default="huff",
        help="Entropy coding method to use (default: huff)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="quality_results.csv",
        help="Path to output CSV file (default: quality_results.csv in project root)",
    )
    args = parser.parse_args()

    run_quality_experiments(
        Path(args.image_dir),
        list(args.qualities) if isinstance(args.qualities, (list, tuple)) else args.qualities,
        args.entropy_method,
        Path(args.output_csv),
    )


if __name__ == "__main__":
    main()


