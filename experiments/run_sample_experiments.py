import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# 加入專案根目錄到 sys.path，方便匯入 main / config / utils / sampler 等模組
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as jpeg_engine
import config

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_and_crop_image(img_path: Path) -> np.ndarray:
    """與 main.process_image 一致：讀圖 + 裁成 16 的倍數。"""
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)
    h, w, _ = img_arr.shape
    h_new, w_new = (h // 16) * 16, (w // 16) * 16
    if h_new != h or w_new != w:
        img_arr = img_arr[:h_new, :w_new, :]
    return img_arr


def run_sample_experiments(
    image_dir: Path,
    downsample_methods: list[str],
    upsample_methods: list[str],
    entropy_method: str,
    output_csv: Path,
) -> None:
    import csv

    image_dir = image_dir.resolve()
    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    )

    if not images:
        print(f"No supported images found in {image_dir}")
        return

    print(f"=== Sampling Experiments on folder: {image_dir} ===")
    print(f"Downsample methods: {downsample_methods}")
    print(f"Upsample methods:   {upsample_methods}")
    print(f"Entropy method:     {entropy_method}")
    print(f"Found {len(images)} images\n")

    rows = []

    for img_path in images:
        print(f"--- {img_path.name} ---")
        img_arr = _load_and_crop_image(img_path)
        h, w, _ = img_arr.shape

        for ds in downsample_methods:
            for us in upsample_methods:
                recon_img, stats = jpeg_engine.jpeg_pipeline(
                    img_arr,
                    dct_method="baseline",
                    idct_method="baseline",
                    downsample_method=ds,
                    upsample_method=us,
                    entropy_method=entropy_method,
                )

                psnr = jpeg_engine.utils.calculate_psnr(img_arr, recon_img)

                orig_bytes = stats.get("original_bytes", h * w * 3)
                comp_bytes = stats.get("compressed_bytes", 0.0)
                ratio = stats.get("compression_ratio", 0.0)
                enc_time = stats.get("time_encoding", 0.0)
                total_time = stats.get("total_time", 0.0)
                down_time = stats.get("time_downsample", 0.0)
                up_time = stats.get("time_upsample", 0.0)

                bpp = (comp_bytes * 8.0) / (h * w) if h > 0 and w > 0 else 0.0

                rows.append(
                    {
                        "image": img_path.name,
                        "width": w,
                        "height": h,
                        "downsample_method": ds,
                        "upsample_method": us,
                        "entropy_method": entropy_method,
                        "psnr": psnr,
                        "original_bytes": orig_bytes,
                        "compressed_bytes": comp_bytes,
                        "compression_ratio": ratio,
                        "bpp": bpp,
                        "time_downsample_ms": down_time,
                        "time_upsample_ms": up_time,
                        "time_encoding_ms": enc_time,
                        "total_time_ms": total_time,
                    }
                )

                print(
                    f"  [DS={ds:10s} | US={us:10s}] "
                    f"PSNR: {psnr:6.2f} dB | "
                    f"Ratio: {ratio:6.2f}x | "
                    f"bpp: {bpp:5.3f} | "
                    f"Down: {down_time:6.1f} ms | "
                    f"Up: {up_time:6.1f} ms | "
                    f"Total: {total_time:6.1f} ms"
                )

        print("-" * 80)

    # 寫入 CSV
    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "width",
        "height",
        "downsample_method",
        "upsample_method",
        "entropy_method",
        "psnr",
        "original_bytes",
        "compressed_bytes",
        "compression_ratio",
        "bpp",
        "time_downsample_ms",
        "time_upsample_ms",
        "time_encoding_ms",
        "total_time_ms",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[Done] Sampling results written to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments on different down/upsampling methods."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="Input image directory (default: images)",
    )
    parser.add_argument(
        "--downsample_methods",
        type=str,
        default="baseline",
        help=(
            "Comma-separated downsample method suffixes, "
            "e.g. 'baseline,bilinear'. Each will map to sampler.downsample_<name> if not 'baseline'."
        ),
    )
    parser.add_argument(
        "--upsample_methods",
        type=str,
        default="baseline,bilinear",
        help=(
            "Comma-separated upsample method suffixes, "
            "e.g. 'baseline,bilinear'. Each will map to sampler.upsample_<name> if not 'baseline'."
        ),
    )
    parser.add_argument(
        "--entropy_method",
        type=str,
        default="huff_dcac_shared",
        help="Entropy coding method (default: huff_dcac_shared)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="sample_results.csv",
        help="Path to output CSV file (default: sample_results.csv)",
    )

    args = parser.parse_args()

    def _parse_list(s: str) -> list[str]:
        return [p.strip() for p in s.split(",") if p.strip()]

    ds_list = _parse_list(args.downsample_methods)
    us_list = _parse_list(args.upsample_methods)

    run_sample_experiments(
        Path(args.image_dir),
        ds_list,
        us_list,
        args.entropy_method,
        Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
