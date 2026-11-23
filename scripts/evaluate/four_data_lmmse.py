"""Run the small four-data-symbol raw/ideal-CSI LMMSE experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_lmmse import load_run_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/evaluation/four_data_lmmse.json")
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/smoke/four_data_lmmse")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stage", choices=("all", "prepare", "solve"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT/"outputs/smoke").resolve() not in output.parents:
        parser.error("This small experiment must write inside outputs/smoke")
    if args.threads < 1:
        parser.error("Threads must be positive")
    with threadpool_limits(args.threads):
        if args.device == "cuda":
            import torch
            torch.set_num_threads(args.threads)
        run(output, args.archive.resolve(), load_run_config(args.config), DenseBackend(args.device), args.stage)


if __name__ == "__main__":
    main()
