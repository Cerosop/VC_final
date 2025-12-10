"""
繪製 IDCT 方法比較圖表
包含：執行時間、加速比、準確度分析
"""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# 確保能匯入專案根目錄的模組
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_idct_csv(csv_path: Path):
    """讀取 IDCT 實驗結果"""
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "image": r["image"],
                    "method": r["method"],
                    "psnr": float(r["psnr"]),
                    "time_idct_ms": float(r["time_idct_ms"]),
                    "time_total_ms": float(r["time_total_ms"]),
                    "time_dct_ms": float(r["time_dct_ms"]),
                    "time_upsample_ms": float(r["time_upsample_ms"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def plot_idct_time_speedup(rows, output_path: Path):
    """
    繪製 IDCT 執行時間 + 加速比圖
    """
    # 定義方法顯示順序
    method_order = ["baseline", "vectorized", "direct", "separable", "lut", "fft", "aan"]
    
    # 過濾出實際存在的方法
    present_methods = {r["method"] for r in rows}
    methods = [m for m in method_order if m in present_methods]

    # 計算每個方法的平均值
    avg_idct = []

    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        avg_idct.append(np.mean([r["time_idct_ms"] for r in m_rows]))

    # 計算加速比 (相對於 baseline)
    baseline_idct = avg_idct[0] if methods[0] == "baseline" else avg_idct[0]
    speedup = [baseline_idct / t for t in avg_idct]

    x = np.arange(len(methods))
    width = 0.6

    # 建立圖表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # IDCT 時間長條圖 (coolwarm 藍色)
    bars = ax1.bar(x, avg_idct, width, color='#3B4CC0', edgecolor='#1F2E7A', linewidth=1.5)
    ax1.set_ylabel('IDCT Time (ms)', fontsize=12)
    ax1.set_title('IDCT Method Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 在長條上標註時間數值
    for i, (bar, time_val) in enumerate(zip(bars, avg_idct)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}ms',
                ha='center', va='bottom', fontsize=10)

    # 加速比折線圖（第二個 Y 軸，coolwarm 紅色）
    ax1_2 = ax1.twinx()
    line = ax1_2.plot(x, speedup, color='#B40426', marker='o', linewidth=2.5, 
                      markersize=10, label='Speedup', markerfacecolor='#B40426', 
                      markeredgecolor='white', markeredgewidth=2)
    ax1_2.set_ylabel('Speedup vs Baseline', color='#B40426', fontsize=12)
    ax1_2.tick_params(axis='y', labelcolor='#B40426')
    ax1_2.axhline(y=1.0, color='#B40426', linestyle='--', alpha=0.5, linewidth=2)

    # 在折線上標註加速比
    for i, speed in enumerate(speedup):
        ax1_2.text(i, speed + 0.015, f'{speed:.2f}x',
                  ha='center', va='bottom', fontsize=9, color='#B40426', fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[已儲存] {output_path}")


def plot_total_time(rows, output_path: Path):
    """
    繪製總 Pipeline 時間比較圖
    """
    # 定義方法顯示順序
    method_order = ["baseline", "vectorized", "direct", "separable", "lut", "fft", "aan"]
    
    # 過濾出實際存在的方法
    present_methods = {r["method"] for r in rows}
    methods = [m for m in method_order if m in present_methods]

    # 計算每個方法的平均值
    avg_total = []

    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        avg_total.append(np.mean([r["time_total_ms"] for r in m_rows]))

    x = np.arange(len(methods))
    width = 0.6

    # 建立圖表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 總時間長條圖 (coolwarm 中間色調，偏藍綠)
    bars = ax.bar(x, avg_total, width, color='#7FA7D1', edgecolor='#4A6FA5', linewidth=1.5)
    ax.set_ylabel('Total Pipeline Time (ms)', fontsize=12)
    ax.set_title('Total JPEG Pipeline Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 標註數值
    for bar, time_val in zip(bars, avg_total):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[已儲存] {output_path}")


def plot_time_breakdown(rows, output_path: Path):
    """
    繪製時間分解堆疊圖
    顯示 DCT、IDCT、Upsample、其他的時間佔比
    """
    method_order = ["baseline", "vectorized", "direct"]
    present_methods = {r["method"] for r in rows}
    methods = [m for m in method_order if m in present_methods]

    # 計算平均時間
    avg_dct = []
    avg_idct = []
    avg_upsample = []
    avg_other = []

    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        dct = np.mean([r["time_dct_ms"] for r in m_rows])
        idct = np.mean([r["time_idct_ms"] for r in m_rows])
        upsample = np.mean([r["time_upsample_ms"] for r in m_rows])
        total = np.mean([r["time_total_ms"] for r in m_rows])
        other = total - dct - idct - upsample

        avg_dct.append(dct)
        avg_idct.append(idct)
        avg_upsample.append(upsample)
        avg_other.append(max(0, other))  # 避免負數

    # 繪製堆疊圖
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(methods))
    width = 0.5

    # 堆疊長條 (使用 coolwarm 色系)
    p1 = ax.bar(x, avg_dct, width, label='DCT/Quant', color='#B40426')  # 深紅
    p2 = ax.bar(x, avg_idct, width, bottom=avg_dct, label='IDCT', color='#3B4CC0')  # 深藍
    p3 = ax.bar(x, avg_upsample, width, 
                bottom=np.array(avg_dct) + np.array(avg_idct),
                label='Upsample', color='#7FA7D1')  # 淺藍
    p4 = ax.bar(x, avg_other, width,
                bottom=np.array(avg_dct) + np.array(avg_idct) + np.array(avg_upsample),
                label='Other', color='#E88E89')  # 淺紅

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('JPEG Pipeline Time Breakdown by Stage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 標註總時間
    for i, m in enumerate(methods):
        total = avg_dct[i] + avg_idct[i] + avg_upsample[i] + avg_other[i]
        ax.text(i, total, f'{total:.0f}ms', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[已儲存] {output_path}")


def plot_accuracy_comparison(rows, output_path: Path):
    """
    繪製準確度比較圖
    顯示各方法的 PSNR 一致性
    """
    method_order = ["baseline", "vectorized", "direct", "separable", "lut", "fft", "aan"]
    present_methods = {r["method"] for r in rows}
    methods = [m for m in method_order if m in present_methods]

    # 計算 PSNR 統計
    avg_psnr = []
    std_psnr = []

    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        psnr_values = [r["psnr"] for r in m_rows]
        avg_psnr.append(np.mean(psnr_values))
        std_psnr.append(np.std(psnr_values))

    # 繪製圖表
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    
    # 長條圖 + 誤差線 (使用 coolwarm 藍色)
    bars = ax.bar(x, avg_psnr, color='#5E80B8', edgecolor='#3B4CC0', linewidth=1.5,
                  yerr=std_psnr, capsize=5, error_kw={'linewidth': 2, 'ecolor': '#2D3436'})
    
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR Comparison Across IDCT Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=avg_psnr[0], color='#B40426', linestyle='--', alpha=0.6, linewidth=2,
               label=f'Baseline: {avg_psnr[0]:.2f} dB')

    # 標註數值
    for bar, psnr_val in zip(bars, avg_psnr):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{psnr_val:.2f}',
                ha='center', va='bottom', fontsize=9)

    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[已儲存] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="繪製 IDCT 實驗結果圖表"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="idct_results.csv",
        help="IDCT 結果 CSV 檔案",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots_idct",
        help="輸出圖表資料夾",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"找不到 CSV: {csv_path}")
        return

    rows = load_idct_csv(csv_path)
    if not rows:
        print("CSV 中沒有有效資料")
        return

    output_dir = Path(args.output_dir)

    # 繪製四張圖
    print("\n繪製圖表中...")
    plot_idct_time_speedup(rows, output_dir / "idct_time_speedup.png")
    plot_total_time(rows, output_dir / "total_time.png")
    plot_time_breakdown(rows, output_dir / "time_breakdown.png")
    plot_accuracy_comparison(rows, output_dir / "accuracy_comparison.png")
    
    print("\n✅ 所有圖表已完成！")


if __name__ == "__main__":
    main()