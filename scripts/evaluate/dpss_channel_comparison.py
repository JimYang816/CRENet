"""Run an isolated receiver-only DPSS group OMP/LMMSE channel comparison."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend, default_device
from src.evaluation.dpss_channel_comparison import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/evaluation/dpss_channel_comparison.json")
    parser.add_argument("--nof1", type=Path, default=ROOT/"data/processed/nof1_compact_20261026_final")
    parser.add_argument("--ncs1", type=Path, default=ROOT/"data/processed/ncs1_compact_20261026_nmse")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    parent = (ROOT/"outputs/experiment").resolve()
    output = (args.output or parent/f"dpss_bomp_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if parent not in output.parents:
        parser.error(f"Output must stay under {parent}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    counts, ridge = config["group_counts"], config["ridge_multipliers"]
    if counts != sorted(set(counts)) or not counts or counts[0] < 1 or not ridge or any(x <= 0 for x in ridge):
        parser.error("Invalid validation candidate grid")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        run({"NOF1": args.nof1.resolve(), "NCS1": args.ncs1.resolve()}, output, config, backend)
    logging.info("Completed comparison in %s", output)


if __name__ == "__main__":
    main()
