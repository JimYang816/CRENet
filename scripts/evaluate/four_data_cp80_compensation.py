"""Evaluate compensated and perfect CSI on the accepted CP=80-ms frames."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_cp80_compensation import run
from src.evaluation.four_data_delay192_compensation import load_target_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"outputs/smoke/four_data_cp80_20261125")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/evaluation/four_data_delay192_compensation.json")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/smoke/four_data_cp80_assisted_20261125")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT/"outputs/smoke").resolve() not in output.parents or args.threads < 1:
        parser.error("Use an output inside outputs/smoke and positive threads")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        run(args.source, output, backend, options=load_target_config(args.config))


if __name__ == "__main__":
    main()
