"""Run the isolated 12-Hz four-data 256-pilot Q sweep."""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits
from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_q_sweep import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stage", choices=("all", "prepare", "solve"), default="all")
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    output = (a.output or ROOT/("outputs/smoke" if a.smoke else "outputs/experiment")/"four_data_256p_q16_32_20261123").resolve()
    parent = (ROOT/("outputs/smoke" if a.smoke else "outputs/experiment")).resolve()
    if parent not in output.parents:
        p.error(f"Output must remain under {parent}")
    with threadpool_limits(a.threads):
        if a.device == "cuda":
            import torch
            torch.set_num_threads(a.threads)
        run(output, a.archive.resolve(), a.smoke, DenseBackend(a.device), stage=a.stage)


if __name__ == "__main__":
    main()
