"""Compare the full-band rotated 4D receiver with the original Q24/L176 run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from data_generation.src.config import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    p.add_argument("--design", type=Path, default=ROOT/"outputs/experiment/four_data_12hz_design_20261119")
    p.add_argument("--baseline", type=Path, default=ROOT/"outputs/experiment/four_data_12hz_q24_l176_20261119")
    p.add_argument("--output", type=Path)
    p.add_argument("--channels", nargs="+", choices=("NOF1", "NCS1"), default=("NOF1", "NCS1"))
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    if not args.report_only:
        p.error("The changed full-band layout is archived. Use scripts.evaluate.four_data_covariance for the fixed original layout; this entry only supports --report-only.")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from threadpoolctl import threadpool_limits
    from src.common.numerics import default_device
    from src.evaluation.four_data_rotated import run_experiment
    out = args.output or ROOT/("outputs/smoke" if args.smoke else "outputs/experiment")/"four_data_rotated_full_band_20261119"
    if args.smoke and ROOT/"outputs/smoke" not in out.resolve().parents:
        p.error("Smoke output must be under outputs/smoke")
    with threadpool_limits(args.threads):
        report = run_experiment(args.archive, args.design, args.baseline, out, channels=args.channels,
            device=args.device or default_device(), smoke=args.smoke, report_only=args.report_only)
    for channel, row in report["channels"].items():
        for scheme in ("old", "new"):
            values = row[scheme]
            print(channel, scheme, "NMSE", values["mean_nmse_db"], flush=True)
            print(channel, scheme, "BER", {name:x["ber"] for name,x in values["algorithms"].items()}, flush=True)


if __name__ == "__main__":
    main()
