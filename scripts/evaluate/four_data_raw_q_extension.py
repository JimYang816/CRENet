"""Extend the NCS1 raw-CSI 4D experiment from Q32 through Q40."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits
from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_raw_q_extension import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    p.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.threads < 1:
        p.error("Threads must be positive")
    category = "smoke" if a.smoke else "experiment"
    original = "four_data_extension_source_20261123" if a.smoke else "four_data_256p_q16_32_20261123"
    source = (a.source or ROOT/"outputs"/category/original).resolve()
    baseline = (a.baseline or ROOT/"outputs/experiment/four_data_256p_raw_q16_32_20261123").resolve()
    output = (a.output or ROOT/"outputs"/category/"ncs1_4d_raw_q33_40_20261123").resolve()
    if (ROOT/"outputs"/category).resolve() not in output.parents:
        p.error("Output must stay inside its smoke or experiment directory")
    with threadpool_limits(a.threads):
        if a.device == "cuda":
            import torch
            torch.set_num_threads(a.threads)
        run(source, baseline, output, a.archive.resolve(), a.smoke, DenseBackend(a.device))


if __name__ == "__main__":
    main()
