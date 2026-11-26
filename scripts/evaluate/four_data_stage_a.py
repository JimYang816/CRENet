"""Compare detector updates on the original training/validation splits."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from data_generation.src.config import ROOT
from src.common.numerics import default_device
from src.evaluation.four_data_stage_a import VARIANTS, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", choices=("NOF1", "NCS1", "all"), default="all")
    parser.add_argument("--start", choices=("A", "B"), default="A",
                        help="start from stage-A epoch 0, or the saved stage-B correction model")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--intervals", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.threads, args.intervals, args.steps, args.batch_size) < 1:
        parser.error("Thread, interval, step and batch counts must be positive")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Variants must be unique")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = args.config or ROOT/"configs/training/four_data_crenet.json"
    base = json.loads(config_path.read_text(encoding="utf-8"))
    previous = args.previous or ROOT/base["output"]
    channels = list(base["datasets"]) if args.channel == "all" else [args.channel]
    for channel in channels:
        source = previous/channel.lower()
        if args.config is None:
            provenance = json.loads((source/"provenance.json").read_text(encoding="utf-8"))
            config = provenance["config"]
        else:
            config = copy.deepcopy(base)
            for section, values in base.get("channel_overrides", {}).get(channel, {}).items():
                config[section].update(values)
        run(config, ROOT/config["datasets"][channel], source,
            args.output/channel.lower(), device=args.device, variants=args.variants,
            intervals=args.intervals, steps=args.steps, batch_size=args.batch_size, start=args.start)


if __name__ == "__main__":
    main()
