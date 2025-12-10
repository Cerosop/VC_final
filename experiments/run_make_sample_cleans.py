"""Convenience CLI to generate clean + avg CSVs for up/down sampling experiments.

Usage (from project root):

  python -m experiments.run_make_sample_cleans

Assumes the following input files already exist:
  - sample_results_upsample.csv
  - sample_results_downsample.csv

Outputs:
  - sample_results_clean_upsample.csv
  - avg_sample_results_clean_upsample.csv
  - sample_results_clean_downsample.csv
  - avg_sample_results_clean_downsample.csv
"""

from pathlib import Path

from .make_sample_results_clean import make_sample_results_clean


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    up_in = root / "sample_results_upsample.csv"
    up_out = root / "sample_results_clean_upsample.csv"

    down_in = root / "sample_results_downsample.csv"
    down_out = root / "sample_results_clean_downsample.csv"

    if not up_in.exists():
        print(f"[Warn] Upsample input CSV not found: {up_in} (skip upsample clean)")
    else:
        print(f"[Info] Generating upsample clean CSVs from: {up_in}")
        make_sample_results_clean(up_in, up_out, analyze="up")

    if not down_in.exists():
        print(f"[Warn] Downsample input CSV not found: {down_in} (skip downsample clean)")
    else:
        print(f"[Info] Generating downsample clean CSVs from: {down_in}")
        make_sample_results_clean(down_in, down_out, analyze="down")


if __name__ == "__main__":
    main()
