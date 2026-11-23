"""Search Q=16..32 on the existing paired physical waveforms using raw CSI."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits
from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_raw_q_sweep import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    p.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.threads < 1:
        p.error("Threads must be positive")
    category = "smoke" if args.smoke else "experiment"
    default_source = "four_data_raw_source_20261123" if args.smoke else "four_data_256p_q16_32_20261123"
    source = (args.source or ROOT/"outputs"/category/default_source).resolve()
    output = (args.output or ROOT/"outputs"/category/"four_data_256p_raw_q16_32_20261123").resolve()
    if (ROOT/"outputs"/category).resolve() not in output.parents:
        p.error("Output must stay inside its smoke or experiment directory")
    with threadpool_limits(args.threads):
        if args.device == "cuda":
            import torch
            torch.set_num_threads(args.threads)
        run(source, output, args.archive.resolve(), args.smoke, DenseBackend(args.device))


if __name__ == "__main__":
    main()
