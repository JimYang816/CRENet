"""Compare NCS1 continuous L176 and the configured fixed discrete support."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.ncs1_delay_support import load_study_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/evaluation/ncs1_delay_support.json")
    parser.add_argument("--tap-count", type=int, help="Override the configured count; 64 repeats the earlier four-candidate study")
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/smoke/ncs1_delay176")
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
        study = load_study_config(args.config, tap_count=args.tap_count)
        run(output, args.archive.resolve(), backend, args.smoke, study=study)


if __name__ == "__main__":
    main()
