"""Independent complete UDNet/two-MMSE comparison."""
from __future__ import annotations
from src.common.numerics import default_device

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.common.outputs import is_smoke_path, resolve_output_root
from src.evaluation.udnet import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/udnet.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        for channel, directory in config["datasets"].items():
            if args.channel and channel != args.channel:
                continue
            smoke = is_smoke_path(directory)
            run = resolve_output_root(config["output"], args.run_root, smoke=smoke)/channel.lower()
            options = dict(config["evaluation"])
            if args.cache_dir is not None:
                options["cache_dir"] = args.cache_dir
            elif smoke:
                options.setdefault("cache_dir", Path("outputs/smoke/_cache"))
            evaluate(Path(directory)/"test.h5", run/"best.pt", run/"evaluation", DenseBackend(args.device),
                     **options)


if __name__ == "__main__":
    main()
