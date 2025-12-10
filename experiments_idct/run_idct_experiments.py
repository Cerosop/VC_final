"""
IDCT 方法效能實驗
測試 7 種 IDCT 實作方法在 Kodak 資料集上的表現
"""

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

# 確保可以從專案根目錄匯入模組
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as jpeg_engine

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 7 種 IDCT 方法
IDCT_METHODS = [
    "baseline",
    "vectorized",
    "direct",
    "separable",
    "lut",
    "fft",
    "aan",
]


def run_idct_experiments(image_dir: Path, output_csv: Path) -> None:
    """
    批量測試所有 IDCT 方法
    """
    image_dir = image_dir.resolve()
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS])

    if not images:
        print(f"在 {image_dir} 找不到支援的圖片")
        return

    print(f"=== IDCT 效能實驗 ===")
    print(f"資料夾: {image_dir}")
    print(f"找到 {len(images)} 張圖片")
    print(f"測試方法: {', '.join(IDCT_METHODS)}\n")

    rows = []

    for img_path in images:
        print(f"--- {img_path.name} ---")
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_arr = np.array(img)
            
            # 裁切成 16 的倍數
            h, w, _ = img_arr.shape
            h_new, w_new = (h // 16) * 16, (w // 16) * 16
            if h_new != h or w_new != w:
                img_arr = img_arr[:h_new, :w_new, :]
                print(f"  裁切: {w}x{h} -> {w_new}x{h_new}")

        except Exception as e:
            print(f"  [跳過] 無法讀取: {e}")
            continue

        # 測試每種 IDCT 方法
        for method in IDCT_METHODS:
            try:
                # 執行 JPEG pipeline
                recon_img, stats = jpeg_engine.jpeg_pipeline(
                    img_arr,
                    dct_method="baseline",
                    idct_method=method,
                    downsample_method="baseline",
                    upsample_method="baseline",
                    entropy_method="baseline",
                )

                # 計算 PSNR
                psnr = jpeg_engine.utils.calculate_psnr(img_arr, recon_img)

                # 記錄數據
                rows.append({
                    "image": img_path.name,
                    "width": w_new,
                    "height": h_new,
                    "method": method,
                    "psnr": float(psnr),
                    "time_idct_ms": float(stats.get("time_decoding_dct", 0)),
                    "time_total_ms": float(stats.get("total_time", 0)),
                    "time_dct_ms": float(stats.get("time_encoding", 0)),
                    "time_upsample_ms": float(stats.get("time_upsample", 0)),
                })

                print(f"  {method:12s} | "
                      f"IDCT: {stats.get('time_decoding_dct', 0):6.1f}ms | "
                      f"Total: {stats.get('total_time', 0):6.1f}ms | "
                      f"PSNR: {psnr:5.2f}dB")

            except Exception as e:
                print(f"  {method:12s} | [失敗] {e}")
                continue

        print("-" * 70)

    # 寫入 CSV
    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "width",
        "height",
        "method",
        "psnr",
        "time_idct_ms",
        "time_total_ms",
        "time_dct_ms",
        "time_upsample_ms",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[完成] 結果已寫入: {output_csv}")

    # 顯示統計摘要
    print("\n" + "=" * 70)
    print("各方法平均效能 (所有圖片)")
    print("=" * 70)

    for method in IDCT_METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue

        avg_idct = sum(r["time_idct_ms"] for r in method_rows) / len(method_rows)
        avg_total = sum(r["time_total_ms"] for r in method_rows) / len(method_rows)
        avg_psnr = sum(r["psnr"] for r in method_rows) / len(method_rows)

        print(f"{method:12s} | "
              f"IDCT: {avg_idct:6.1f}ms | "
              f"Total: {avg_total:6.1f}ms | "
              f"PSNR: {avg_psnr:5.2f}dB")


def main():
    parser = argparse.ArgumentParser(
        description="批量測試 IDCT 方法效能"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images\PhotoCD_PCD0992",
        help="圖片資料夾 (預設: images)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="idct_results.csv",
        help="輸出 CSV 檔案 (預設: idct_results.csv)",
    )
    args = parser.parse_args()

    run_idct_experiments(Path(args.image_dir), Path(args.output_csv))


if __name__ == "__main__":
    main()
