"""Audit the saved high-SNR CSI regression without changing original results."""
import argparse
import os
from pathlib import Path
from data_generation.src.config import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT/"outputs/experiment/four_data_rotated_full_band_20261119")
    p.add_argument("--output", type=Path)
    p.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    p.add_argument("--device", choices=("cpu","cuda"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--frame-limit", type=int)
    p.add_argument("--channels", nargs="+", choices=("NOF1", "NCS1"), default=("NOF1", "NCS1"))
    a = p.parse_args()
    if a.threads < 1 or a.frame_limit is not None and a.frame_limit < 1:
        p.error("Limits must be positive")
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(a.threads)
    from threadpoolctl import threadpool_limits
    from src.common.numerics import default_device
    from src.evaluation.four_data_rotated_audit import run_audit
    with threadpool_limits(a.threads):
        run_audit(a.run,a.output or a.run/"covariance_audit",a.archive,
                  device=a.device or default_device(),channels=a.channels,frame_limit=a.frame_limit)


if __name__ == "__main__":
    main()
