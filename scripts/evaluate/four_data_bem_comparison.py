"""Compare the current DPSS-BEM with CE-BEM at ±16 subcarriers."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_bem_comparison import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="One test frame per channel, 0/15/30 dB")
    args = parser.parse_args()
    root = ROOT/"outputs"/("smoke" if args.smoke else "experiment")
    output = (args.output or root/"four_data_bem_comparison").resolve()
    if root.resolve() not in output.parents or args.threads < 1:
        parser.error(f"Use a run directory inside {root} and a positive thread count")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        run(output, args.archive.resolve(), backend, smoke=args.smoke)


if __name__ == "__main__":
    main()
