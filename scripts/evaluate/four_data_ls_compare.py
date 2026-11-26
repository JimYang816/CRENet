"""Compare single-symbol minimum-norm LS and LMMSE on paired CP80 frames."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_ls_compare import ACCEPTED, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--data", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--accepted", type=Path, default=ROOT/ACCEPTED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="Check both estimators on small physical splits on CPU")
    args = parser.parse_args(argv)
    if min(args.threads, args.workers) < 1 or (args.smoke and args.device != "cpu"):
        parser.error("Use positive thread and worker counts and CPU for smoke runs")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        run(args.output.resolve(), args.data.resolve(), args.archive.resolve(), args.accepted.resolve(),
            backend, workers=1 if args.smoke else args.workers, smoke=args.smoke)


if __name__ == "__main__":
    main()
