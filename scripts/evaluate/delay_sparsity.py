"""Analyze delay-group energy in the accepted NOF1 and NCS1 reference data."""
from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.evaluation.delay_sparsity import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nof1", type=Path, default=ROOT / "data/processed/nof1_compact_20261026_final")
    parser.add_argument("--ncs1", type=Path, default=ROOT / "data/processed/ncs1_compact_20261026_nmse")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-windows", type=int, default=1,
                        help="Number of physical windows per split to verify on CPU when using CUDA")
    args = parser.parse_args()
    if args.threads < 1 or args.verify_windows < 0:
        parser.error("Threads must be positive and verify-windows nonnegative")
    parent = (ROOT / "outputs/experiment").resolve()
    output = (args.output or parent / f"delay_sparsity_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if parent not in output.parents:
        parser.error(f"Output must be a new directory under {parent}")
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with threadpool_limits(args.threads):
        if device == "cuda":
            import torch
            torch.set_num_threads(args.threads)
        run({"NOF1": args.nof1.resolve(), "NCS1": args.ncs1.resolve()},
            output, device, verify_windows=args.verify_windows)
    logging.info("Completed report in %s", output)


if __name__ == "__main__":
    main()
