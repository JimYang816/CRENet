"""Compare NCS1 dense L176, first64, best-window64 and independent-prior-selected64."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.ncs1_delay_support import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/smoke/ncs1_delay64")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Two independent prior windows, one test frame, three SNRs")
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT/"outputs/smoke").resolve() not in output.parents or args.threads < 1:
        parser.error("Use an output inside outputs/smoke and positive threads")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        run(output, args.archive.resolve(), backend, args.smoke)


if __name__ == "__main__":
    main()
