import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

# 確保可以從專案根目錄匯入 main / config / entropy 等模組
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as jpeg_engine  # 引入我們的主程式


def run_entropy_comparison(image_path, save_plot=True):
    print(f"=== 正在測試圖片: {image_path} ===")

    try:
        img = Image.open(image_path).convert('RGB')
        img_arr = np.array(img)
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return

    # 為了公平比較，先裁切成 16 倍數
    h, w, _ = img_arr.shape
    h_new, w_new = (h // 16) * 16, (w // 16) * 16
    img_arr = img_arr[:h_new, :w_new, :]

    methods = ['raw', 'rle', 'huff', 'huff_global', 'huff_dpcm', 'huff_dcac_shared']
    results = []

    print(f"{'Method':<10} | {'Size (KB)':<10} | {'Ratio':<8} | {'PSNR (dB)':<10} | {'Enc Time (ms)':<12}")
    print("-" * 65)

    for method in methods:
        # 呼叫主程式的 pipeline
        # 注意：我們只改變 entropy_method，其他保持 baseline
        recon_img, stats = jpeg_engine.jpeg_pipeline(
            img_arr,
            dct_method='baseline',
            idct_method='baseline',
            downsample_method='baseline',
            upsample_method='baseline',
            entropy_method=method
        )

        # 計算 PSNR
        psnr = jpeg_engine.utils.calculate_psnr(img_arr, recon_img)

        # 收集數據
        comp_size_kb = stats['compressed_bytes'] / 1024
        ratio = stats['compression_ratio']
        enc_time = stats['time_encoding']

        results.append({
            'method': method,
            'size': comp_size_kb,
            'ratio': ratio,
            'psnr': psnr,
            'time': enc_time,
            'width': w,
            'height': h,
            'original_bytes': h * w * 3,
            'compressed_bytes': stats['compressed_bytes'],
            'time_total_ms': stats.get('total_time', 0.0),
        })

        print(f"{method:<10} | {comp_size_kb:<10.2f} | {ratio:<8.2f} | {psnr:<10.2f} | {enc_time:<12.1f}")

    print("-" * 65)

    if save_plot:
        # 畫圖比較 (長條圖)
        methods_list = [r['method'] for r in results]
        ratios = [r['ratio'] for r in results]
        times = [r['time'] for r in results]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        x = np.arange(len(methods_list))
        width = 0.35

        ax1.bar(x - width/2, ratios, width, label='Compression Ratio', color='skyblue')
        ax1.set_ylabel('Compression Ratio (Higher is better)')
        ax1.set_title(f'Entropy Methods Comparison: {Path(image_path).name}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods_list)

        ax2 = ax1.twinx()
        ax2.plot(x, times, color='red', marker='o', label='Encoding Time (ms)')
        ax2.set_ylabel('Time (ms)')

        # 合併圖例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        output_plot = f"comparison_{Path(image_path).stem}.png"
        plt.savefig(output_plot)
        print(f"\n[圖表已儲存] {output_plot}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to the input image')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print("檔案不存在，請確認路徑。")
    else:
        run_entropy_comparison(args.image_path)


