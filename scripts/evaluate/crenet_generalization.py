"""Diagnose the frozen NOF1-trained CRENet without fitting on target data."""
import argparse
import json
from pathlib import Path

import torch

from data_generation.src.config import ROOT
from src.common.numerics import default_device
from src.evaluation.crenet_generalization import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/evaluation/crenet_generalization.json")
    parser.add_argument("--device",choices=("cpu","cuda"),default=default_device())
    parser.add_argument("--frames",type=int)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.frames is not None:
        if args.frames < 1:
            parser.error("frames must be positive")
        config["frames"] = args.frames
    run(config,args.output or ROOT/config["output"],device=args.device)


if __name__ == "__main__":
    main()
