"""Run the opt-in 5 dB reference-assisted CSI sensitivity experiment."""
import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.four_data_csi_compensation import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/smoke/four_data_cp256_q24_20261123")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/smoke/four_data_cp256_q24_assisted_5db")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / "outputs/smoke").resolve() not in output.parents:
        parser.error("This small experiment must write inside outputs/smoke")
    if args.threads < 1:
        parser.error("Threads must be positive")
    with threadpool_limits(args.threads):
        if args.device == "cuda":
            import torch
            torch.set_num_threads(args.threads)
        run(args.source, output, DenseBackend(args.device))


if __name__ == "__main__":
    main()
