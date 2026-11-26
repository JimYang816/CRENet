"""Evaluate single-symbol CSI on the default CP80 frame or an explicit comb study."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation import four_data_protected_pilots as protected
from src.evaluation import four_data_single_symbol as comb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help=f"Reproduce the original comb study with a config such as {comb.CONFIG}")
    parser.add_argument("--data", type=Path, default=ROOT/"data/processed",
                        help="Parent directory of the accepted CP80 channel datasets")
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8, help="Physical replay workers for the default frame")
    parser.add_argument("--smoke", action="store_true", help="Use small physical splits on CPU")
    args = parser.parse_args(argv)
    if min(args.threads, args.workers) < 1 or (args.smoke and args.device != "cpu"):
        parser.error("Use positive thread and worker counts and CPU for smoke runs")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        output, data, archive = (path.resolve() for path in (args.output, args.data, args.archive))
        if args.config is None:
            protected.run(output, data, archive, protected.load_options(smoke=args.smoke), backend,
                          workers=1 if args.smoke else args.workers, smoke=args.smoke)
        else:
            comb.run(output, data, archive, backend, comb.load_options(args.config, smoke=args.smoke), smoke=args.smoke)


if __name__ == "__main__":
    main()
