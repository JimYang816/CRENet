"""Recalibrate accepted NOF1/NCS1 CSI targets, then test B16 on complete test splits."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import time

from threadpoolctl import threadpool_limits
from data_generation.src.config import ROOT
from data_generation.src.dataset.four_data_protocol import (
    DEFAULT_OUTPUT_ROOT,
    load_options,
)
from data_generation.src.dataset.four_data_recalibration import recalibrate_datasets
from data_generation.src.dataset.four_data_targets import CHANNELS, load_target_config
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend, default_device
from src.evaluation.assisted_banded import test_datasets

CODE = (
    "data_generation/scripts/recalibrate_four_data.py",
    "data_generation/src/dataset/four_data_recalibration.py",
    "data_generation/src/dataset/four_data_targets.py",
    "data_generation/src/dataset/four_data_export.py",
    "data_generation/src/dataset/four_data_acceptance.py",
    "data_generation/src/dataset/four_data_math.py",
    "src/data/four_data.py",
    "src/evaluation/assisted_banded.py",
    "src/evaluation/comparison.py",
    "src/evaluation/inputs.py",
    "src/evaluation/nmse.py",
    "src/evaluation/four_data_lmmse.py",
    "src/algorithms/banded_lmmse/equalizer.py",
    "src/algorithms/likelihood.py",
    "src/algorithms/interface.py",
    "src/evaluation/scoring.py",
    "src/evaluation/fingerprints.py",
    "src/evaluation/result_cache.py",
    "src/common/numerics.py",
    "src/common/experiment.py",
    "configs/data/four_data_dataset.json",
    "configs/evaluation/four_data_delay192_compensation.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("all", "calibrate", "test"), nargs="?", default="all"
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/experiment/dddd_assisted_b16_20261127_v2",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "configs/evaluation/four_data_delay192_compensation.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", nargs="+", choices=CHANNELS, default=CHANNELS)
    args = parser.parse_args()
    if min(args.threads, args.batch_size) < 1:
        parser.error("Use positive thread and batch counts")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError(
            "Recalibration and formal tests require clean committed source"
        )
    run = args.output.resolve()
    if (ROOT / "outputs/experiment").resolve() not in run.parents:
        parser.error("Formal calibration records must be inside outputs/experiment")
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run / "run.log", encoding="utf-8"),
        ],
    )
    targets = load_target_config(args.targets)
    options = load_options()
    channels = [channel for channel in CHANNELS if channel in args.channels]
    options = dict(options, channels=channels)
    identity = dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dataset_root=str(args.dataset_root.resolve()),
        targets=targets,
        channels=channels,
        code_sha256={name: file_sha256(ROOT / name) for name in CODE},
        reference_assisted=True,
        half_bandwidth=16,
    )
    identity_path = run / "run_identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if (
            previous["targets"] != targets
            or previous["dataset_root"] != identity["dataset_root"]
            or previous.get("channels", list(CHANNELS)) != channels
        ):
            raise ValueError("Output belongs to different targets or source data")
    else:
        write_json(identity_path, identity)
    runtime_path = run / "runtime.json"
    runtime = (
        json.loads(runtime_path.read_text())
        if runtime_path.exists()
        else dict(attempts=[])
    )
    attempt = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        stage=args.stage,
        channels=channels,
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        commit=identity["commit"],
        complete=False,
    )
    runtime["attempts"].append(attempt)
    write_json(runtime_path, runtime)
    began = time.perf_counter()
    try:
        with threadpool_limits(args.threads):
            backend = DenseBackend(args.device)
            if backend.torch:
                backend.torch.set_num_threads(args.threads)
            if args.stage in ("all", "calibrate"):
                recalibrate_datasets(args.dataset_root, run, options, targets, backend)
            if args.stage in ("all", "test"):
                if not (run / "dataset_manifest.json").exists():
                    raise ValueError(
                        "Complete calibration and acceptance before full testing"
                    )
                test_datasets(
                    args.dataset_root, run, targets, backend, args.batch_size,
                    channels=channels,
                )
        attempt["complete"] = True
    except BaseException as exc:
        attempt["error"] = repr(exc)
        raise
    finally:
        attempt["seconds"] = time.perf_counter() - began
        write_json(runtime_path, runtime)


if __name__ == "__main__":
    main()
