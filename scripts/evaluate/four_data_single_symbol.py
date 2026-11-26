"""Search single-symbol CSI models and test them on paired CP80 observations."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_single_symbol import CONFIG, load_options, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/CONFIG)
    parser.add_argument("--data", type=Path, default=ROOT/"data/processed",
                        help="Parent directory of the accepted CP80 channel datasets")
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Use the CPU smoke dataset and a reduced search grid")
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.smoke and args.device != "cpu"):
        parser.error("Use positive threads and CPU for smoke runs")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        run(args.output.resolve(), args.data.resolve(), args.archive.resolve(), backend,
            load_options(args.config, smoke=args.smoke), smoke=args.smoke)


if __name__ == "__main__":
    main()
