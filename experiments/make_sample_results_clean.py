"""Generate clean CSVs for sampling comparison (up/down).

Input:  a full metrics CSV from run_sample_experiments (e.g. sample_results_upsample.csv
    or sample_results_downsample.csv).
Output: a clean CSV with only a subset of columns, and an avg-by-method CSV.

By default this script assumes we want to analyze **upsampling** and therefore keeps
`time_upsample_ms` and computes averages over it. For downsampling experiments, you
can either change the flags in `main()` or import `make_sample_results_clean` from
another script and set `analyze="down"`.
"""

import csv
from pathlib import Path
from collections import defaultdict


def make_sample_results_clean(
    input_csv: Path,
    output_csv: Path,
    analyze: str = "up",  # "up" -> use time_upsample_ms, "down" -> use time_downsample_ms
) -> None:
    input_csv = input_csv.resolve()
    output_csv = output_csv.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base = {
                "image": row.get("image", ""),
                "width": row.get("width", ""),
                "height": row.get("height", ""),
                "downsample_method": row.get("downsample_method", ""),
                "upsample_method": row.get("upsample_method", ""),
                "psnr": row.get("psnr", ""),
            }

            # 根據 analyze 決定要保留哪種時間欄位
            if analyze == "down":
                base["time_downsample_ms"] = row.get("time_downsample_ms", "")
            else:
                base["time_upsample_ms"] = row.get("time_upsample_ms", "")

            rows.append(base)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if analyze == "down":
        fieldnames = [
            "image",
            "width",
            "height",
            "downsample_method",
            "upsample_method",
            "psnr",
            "time_downsample_ms",
        ]
        time_key = "time_downsample_ms"
    else:
        fieldnames = [
            "image",
            "width",
            "height",
            "downsample_method",
            "upsample_method",
            "psnr",
            "time_upsample_ms",
        ]
        time_key = "time_upsample_ms"

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[Done] Clean CSV written to: {output_csv}")

    # 另外產生一份 per-method 平均結果的 CSV，方便直接看 avg
    avg_output_csv = output_csv.with_name("avg_" + output_csv.name)
    method_stats: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for r in rows:
        # 聚合依據：
        # - upsampling 實驗：照 upsample_method 分組
        # - downsampling 實驗：照 downsample_method 分組
        if analyze == "down":
            method = str(r["downsample_method"])
        else:
            method = str(r["upsample_method"])

        try:
            psnr_val = float(r["psnr"])
            t_val = float(r[time_key])
        except (TypeError, ValueError):
            continue
        method_stats[method].append((psnr_val, t_val))

    avg_rows = []
    for method, vals in method_stats.items():
        if not vals:
            continue
        n = len(vals)
        sum_psnr = sum(v[0] for v in vals)
        sum_time = sum(v[1] for v in vals)
        avg_rows.append(
            {
                "method": method,
                "num_samples": n,
                "avg_psnr": sum_psnr / n,
                "avg_time_ms": sum_time / n,
            }
        )

    avg_fieldnames = [
        "method",
        "num_samples",
        "avg_psnr",
        "avg_time_ms",
    ]

    with avg_output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=avg_fieldnames)
        writer.writeheader()
        for r in avg_rows:
            writer.writerow(r)

    print(f"[Done] Avg-by-method CSV written to: {avg_output_csv}")


def main():
    """Default: treat sample_results.csv as upsampling experiment.

    如果你想用 CLI 直接針對 up/down 分別輸出，推薦建立額外腳本來呼叫
    `make_sample_results_clean(input, output, analyze="up" or "down")`，
    或者手動修改這裡的 input/output 與 analyze 參數。
    """

    root = Path(__file__).resolve().parent.parent
    input_csv = root / "sample_results.csv"
    output_csv = root / "sample_results_clean.csv"
    make_sample_results_clean(input_csv, output_csv, analyze="up")


if __name__ == "__main__":
    main()
