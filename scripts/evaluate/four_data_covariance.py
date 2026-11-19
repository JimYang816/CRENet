"""Run the 4D CSI study with the original carrier layout fixed and checked."""
import argparse
import os
from pathlib import Path
from data_generation.src.config import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    p.add_argument("--basis-source", "--previous", dest="basis_source", type=Path,
                   default=ROOT/"outputs/experiment/four_data_12hz_design_20261119",
                   help="Independent temporal-basis NPZ and physical prior-projection cache; contains no transmitted carrier layout")
    p.add_argument("--baseline", type=Path, default=ROOT/"outputs/experiment/four_data_12hz_q24_l176_20261119")
    p.add_argument("--output", type=Path, default=ROOT/"outputs/experiment/four_data_original_layout_joint_20261119")
    p.add_argument("--channels", nargs="+", choices=("NOF1", "NCS1"), default=("NOF1", "NCS1"))
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if a.threads < 1:
        p.error("Threads must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(a.threads)
    from threadpoolctl import threadpool_limits
    from src.common.numerics import default_device
    from src.evaluation.four_data_original import run_experiment
    with threadpool_limits(a.threads):
        run_experiment(a.archive, a.basis_source, a.baseline, a.output, channels=a.channels,
                       device=a.device or default_device(), smoke=a.smoke, report_only=a.report_only)


if __name__ == "__main__":
    main()
