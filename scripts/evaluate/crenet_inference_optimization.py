"""Compare the two exact frozen CRENet inference optimizations on both channels."""
from src.common.numerics import default_device

import argparse
import json
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.experiment import write_json
from src.evaluation.crenet_inference_optimization import run_channel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/crenet.json"))
    parser.add_argument("--trained-root", type=Path, default=Path("outputs/experiment/icr_full_20261031_095549"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", choices=("NOF1", "NCS1"), default=["NOF1", "NCS1"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--validation-frames", type=int)
    parser.add_argument("--test-frames", type=int)
    args = parser.parse_args()
    if min(args.batch_size, args.threads, args.repeats) < 1:
        parser.error("batch size, threads and repeats must be positive")
    for limit in (args.validation_frames, args.test_frames):
        if limit is not None and limit < 1:
            parser.error("frame limits must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    with threadpool_limits(args.threads):
        try:
            for channel in args.channels:
                write_json(args.output/"status.json", dict(phase="running", channel=channel))
                reports[channel] = run_channel(config["datasets"][channel],
                    args.trained_root/"joint"/channel.lower()/"icr_mmse.pt", args.output/channel.lower(),
                    device=args.device, batch_size=args.batch_size, repeats=args.repeats,
                    validation_frames=args.validation_frames, test_frames=args.test_frames)
            passed = all(report["default_numerical_checks_passed"] for report in reports.values())
            write_json(args.output/"summary.json", dict(complete=True, default_numerical_checks_passed=passed, channels=reports))
            write_json(args.output/"status.json", dict(phase="completed", channels=args.channels,
                                                       default_numerical_checks_passed=passed))
        except Exception as error:
            write_json(args.output/"status.json", dict(phase="failed", error=repr(error)))
            raise


if __name__ == "__main__":
    main()
