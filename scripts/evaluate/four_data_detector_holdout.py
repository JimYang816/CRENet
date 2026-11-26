"""Score a frozen, validation-selected CRENet detector on the paired test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_generation.src.config import ROOT
from src.common.numerics import default_device
from src.evaluation.four_data_detector_holdout import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=("NOF1", "NCS1"), required=True)
    parser.add_argument("--diagnosis", type=Path, required=True,
                        help="diagnosis root containing the chosen channel directory")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ch = args.channel.lower()
    config = json.loads((ROOT/"configs/training/four_data_crenet.json").read_text(encoding="utf-8"))
    source = (args.previous or ROOT/config["output"])/ch
    provenance = json.loads((source/"provenance.json").read_text(encoding="utf-8"))
    report = run(ROOT/provenance["config"]["datasets"][args.channel], source, args.diagnosis/ch,
                 args.output/ch, variant=args.variant, device=args.device)
    print(json.dumps(dict(channel=args.channel, error_change_total=report["error_change_total"],
                          error_change_by_snr=report["error_change_by_snr"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
