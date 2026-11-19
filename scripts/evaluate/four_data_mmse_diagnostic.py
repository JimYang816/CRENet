"""Explain the two MMSE baselines using paired CSI controls on saved 4D frames."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from data_generation.src.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"outputs/experiment/four_data_12hz_q24_l176_20261119")
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frame-limit", type=int)
    args = parser.parse_args()
    if args.threads < 1 or args.frame_limit is not None and args.frame_limit < 1:
        parser.error("Thread and frame limits must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from threadpoolctl import threadpool_limits
    from src.common.numerics import default_device
    from src.evaluation.four_data_mmse_diagnostic import run_audit

    with threadpool_limits(args.threads):
        run_audit(args.run, args.output or args.run/"diagnostic", args.archive,
                  device=args.device or default_device(), frame_limit=args.frame_limit)


if __name__ == "__main__":
    main()
