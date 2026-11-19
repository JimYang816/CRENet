"""Audit pilot rank and channel representation for fixed 12-Hz, no-P 4D frames."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from data_generation.src.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-ranks", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--temporal-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from threadpoolctl import threadpool_limits
    from src.evaluation.four_data_design import run_design, temporal_audit

    output = args.output or ROOT/("outputs/smoke" if args.smoke else "outputs/experiment")/"four_data_12hz_design_20261119"
    with threadpool_limits(args.threads):
        if args.temporal_only:
            temporal_audit(output, device=args.device, ranks=not args.skip_ranks)
        else:
            run_design(output, args.archive, smoke=args.smoke, ranks=not args.skip_ranks, device=args.device)


if __name__ == "__main__":
    main()
