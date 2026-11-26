"""Evaluate banded LMMSE on the explicitly selected test CSI view."""
from __future__ import annotations
from src.common.numerics import default_device

import argparse
import json
import logging
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.common.outputs import is_smoke_path, resolve_output_root
from src.evaluation.comparison import evaluate, replot
from src.evaluation.registry import IDENTIFIERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/ber_comparison.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--baselines-only", action="store_true", help="Ignore learned checkpoints in a historical configuration")
    parser.add_argument("--input-view", choices=("raw", "reference_assisted"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--force", nargs="+", choices=(*IDENTIFIERS, "all"), default=[],
                       help="Recompute the selected LMMSE baselines")
    modes.add_argument("--plot-only", action="store_true", help="Redraw saved complete reports without data or GPU")
    parser.add_argument("--from-run", type=Path, help="With --plot-only, read saved reports from this root and write to the configured output")
    args = parser.parse_args()
    if args.from_run is not None and not args.plot_only:
        parser.error("--from-run requires --plot-only")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    datasets = {channel: path for channel, path in config["datasets"].items()
                if not args.channel or channel == args.channel}
    if args.plot_only:
        output = Path(config["output"])
        if args.from_run is not None and args.from_run.resolve() == output.resolve():
            parser.error("--from-run must differ from the configured output; omit it to redraw in place")
        for channel in datasets:
            if args.from_run is None:
                replot(output/channel.lower())
            else:
                replot(args.from_run/channel.lower(), output=output/channel.lower(), expected_channel=channel,
                       expected_view=args.input_view or config.get("input_view", "raw"),
                       required_algorithms=config.get("algorithms", {}))
        return
    if not args.baselines_only and any(config.get("checkpoints", {}).get(channel) for channel in datasets):
        parser.error("Use --baselines-only to evaluate a historical configuration with learned checkpoints")
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        for channel, dataset in datasets.items():
            smoke = is_smoke_path(dataset)
            output = resolve_output_root(config["output"], smoke=smoke)
            cache_dir = config.get("cache_dir") or (Path("outputs/smoke/_cache") if smoke else None)
            evaluate(dataset, output/channel.lower(), backend,
                     batch_size=args.batch_size if args.batch_size is not None else config["batch_size"],
                     algorithm_parameters=config.get("algorithms"), force=args.force,
                     cache_dir=cache_dir, input_view=args.input_view or config.get("input_view", "raw"))


if __name__ == "__main__":
    main()
