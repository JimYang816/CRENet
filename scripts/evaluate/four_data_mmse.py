"""Run the isolated no-P 4D experiment at 12 Hz, Q=24 and L=176."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from data_generation.src.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--channels", nargs="+", choices=("NOF1", "NCS1"), default=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Two prior windows and one test frame per channel")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from threadpoolctl import threadpool_limits
    from src.common.numerics import default_device
    from src.evaluation.four_data_mmse import run_experiment

    output = args.output or ROOT/("outputs/smoke" if args.smoke else "outputs/experiment")/"four_data_12hz_q24_l176_20261119"
    if args.smoke and ROOT/"outputs/smoke" not in output.resolve().parents:
        parser.error("Smoke output must be under outputs/smoke")
    with threadpool_limits(args.threads):
        report = run_experiment(args.archive, output, channels=args.channels,
                                device=args.device or default_device(), smoke=args.smoke)
    for channel, result in report["channels"].items():
        print(channel, "NMSE dB:", result["mean_nmse_db"], flush=True)
        for name, values in result["algorithms"].items():
            print(channel, name, "BER:", values["ber"], flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
