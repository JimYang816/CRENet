"""Run NOF1 and NCS1 with 192 discrete physical delay taps and raw CSI."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_delay192 import load_study_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/evaluation/four_data_delay192.json")
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/smoke/four_data_delay192")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Two prior windows, one test frame, 0/15/30 dB; NOF1 selection is provisional")
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT/"outputs/smoke").resolve() not in output.parents or args.threads < 1:
        parser.error("Use an output inside outputs/smoke and positive threads")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        run(output, args.archive.resolve(), backend, args.smoke, options=load_study_config(args.config))


if __name__ == "__main__":
    main()
