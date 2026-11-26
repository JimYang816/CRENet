"""Generate and accept formal NOF1/NCS1 DDDD datasets with fixed CSI targets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import shutil
import subprocess
import time

from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT, digest
from data_generation.src.dataset.four_data_protocol import (
    DEFAULT_OUTPUT_ROOT, DEFAULT_RUN_ID, SPLITS, channel_config, counts, estimated_max_bytes, load_options, windows,
)
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend, default_device
from data_generation.src.dataset.four_data_source import CODE as SOURCE_CODE, verify_source
from data_generation.src.dataset.four_data_targets import load_target_config

CODE = tuple(dict.fromkeys((*SOURCE_CODE, "data_generation/scripts/four_data.py",
    "data_generation/src/dataset/four_data_protocol.py", "data_generation/src/dataset/four_data_math.py",
    "data_generation/src/dataset/four_data_physical.py", "data_generation/src/dataset/four_data_export.py",
    "data_generation/src/dataset/four_data_acceptance.py", "src/data/four_data.py", "src/data/__init__.py",
    "configs/data/four_data_dataset.json", "configs/data/nof1/dddd.json", "configs/data/ncs1/dddd.json")))
LOG = logging.getLogger(__name__)


def code_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in CODE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "audit", "generate", "accept"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/data/four_data_dataset.json")
    parser.add_argument("--archive", type=Path, default=ROOT/"data/raw/watermark/WatermarkV1.zip")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--channels", nargs="+", choices=("NOF1", "NCS1"), default=["NOF1", "NCS1"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="CPU verification: one window per split, test repeats=2, 0/15/30 dB")
    args = parser.parse_args(argv)
    if (not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", "..")
            or any(value in args.run_id for value in ("/", "\\", ":")) or not 1 <= args.workers <= 8
            or min(args.threads, args.worker_threads) < 1 or len(set(args.channels)) != len(args.channels)):
        parser.error("Use a single safe run ID, distinct channels, positive threads and 1..8 workers")
    if args.smoke and args.device != "cpu":
        parser.error("Local smoke generation must use CPU")
    output_root = args.output_root.resolve()
    if not args.smoke and output_root != (ROOT/"data/processed").resolve() and (ROOT/"data/processed").resolve() not in output_root.parents:
        parser.error("Formal datasets must live inside data/processed")
    options = load_options(args.config)
    target_path = ROOT/options["target_config"]
    targets = load_target_config(target_path)
    source = (ROOT/options["source_study"]).resolve()
    configurations = {channel: channel_config(channel, args.smoke) for channel in args.channels}
    run_id = args.run_id+("_smoke" if args.smoke and not args.run_id.endswith("_smoke") else "")
    run = ROOT/"outputs"/("smoke" if args.smoke else "experiment")/run_id
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run/"run.log", encoding="utf-8")])
    directories = {channel: (run/"data" if args.smoke else output_root)/(channel.lower()+"_dddd") for channel in args.channels}
    hashes = code_hashes()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True).strip()
    if dirty and not args.smoke:
        raise RuntimeError("Formal generation requires clean committed source; datasets/results stay ignored by Git")
    source_identity = dict(archive_sha256=file_sha256(args.archive.resolve()),
        study_metrics_sha256=file_sha256(source/"metrics.json"),
        frozen_support_sha256={channel: file_sha256(source/channel.lower()/"frozen_support192.json") for channel in args.channels})
    verified_source = None
    if not args.smoke:
        verified_source = verify_source(source)
        source_identity["physical_sha256"] = {channel: verified_source[2][channel]["sha256"] for channel in args.channels}
    identities = {channel: digest(dict(config=c, options=options, targets=targets, generation_code=hashes, source=source_identity))
                  for channel, c in configurations.items()}
    state_path = run/"run_identity.json"
    state = dict(schema=options["schema"], initial_commit=commit, code_sha256=hashes,
        identities=identities, options=options, targets=targets, smoke=args.smoke, source=source_identity,
        directories={ch: str(path) for ch, path in directories.items()})
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding="utf-8"))
        if old["identities"] != identities or old["directories"] != state["directories"]:
            raise ValueError("Run ID belongs to different source/settings; use a new run ID")
        state = old
    else:
        write_json(state_path, state)
    runtime_path = run/"runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else dict(attempts=[])
    attempt = dict(started_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage, device=args.device,
        workers=args.workers, threads=args.threads, worker_threads=args.worker_threads, complete=False)
    runtime["attempts"].append(attempt)
    write_json(runtime_path, runtime)
    began = time.perf_counter()
    reports = {}
    try:
        from data_generation.src.channel.watermark.replay import WatermarkArchive
        from data_generation.src.dataset.four_data_acceptance import accept
        from data_generation.src.dataset.four_data_export import generate, load_prior
        from data_generation.src.receiver.blocked import BlockedReceiver

        needed = sum(max(0, estimated_max_bytes(c)-sum(path.stat().st_size for path in directories[ch].glob("*.h5")))
                     for ch, c in configurations.items())
        if args.stage in ("all", "generate") and shutil.disk_usage(output_root if output_root.exists() else ROOT).free < needed+2*1024**3:
            raise RuntimeError(f"Insufficient disk for conservative {needed/1024**3:.1f} GiB output bound plus 2 GiB margin")
        with threadpool_limits(args.threads):
            backend = DenseBackend(args.device)
            if backend.torch:
                backend.torch.set_num_threads(args.threads)
            for channel, c in configurations.items():
                channel_run = run/channel.lower()
                channel_run.mkdir(exist_ok=True)
                write_json(channel_run/"configuration.json", dict(config=c, counts=counts(c), options=options, targets=targets,
                    estimated_max_bytes=estimated_max_bytes(c), identity=identities[channel]))
                if args.stage != "accept":
                    archive = WatermarkArchive(args.archive.resolve(), channel)
                    records = sorted({record for split in SPLITS for record, _ in windows(c, split)} |
                                     ({51} if args.smoke else set(c["prior_records"])))
                    source_manifest = archive.audit(records)
                    if source_manifest["archive_sha256"] != source_identity["archive_sha256"]:
                        raise ValueError("Archive changed during generation")
                    if not args.smoke and source_manifest["archive_sha256"] != verified_source[0]["channel_manifests"][channel]["archive_sha256"]:
                        raise ValueError("Archive differs from the frozen independent-prior study")
                    duration = (4*(c["nfft"]+c["cp"])+2*c["guard_samples"])/c["fs_baseband"]
                    for split in SPLITS:
                        for record, start in windows(c, split):
                            if start-c["guard_samples"]/c["fs_baseband"]+duration >= source_manifest["records"][str(record)]["time_axis_seconds"][1]:
                                raise ValueError("A configured physical window crosses its recording boundary")
                    source_manifest.update(counts=counts(c), reserved_csi_validation_records=c["validation_records"],
                        reserved_records_used_for_training=False)
                    write_json(channel_run/"source_manifest.json", source_manifest)
                    if args.stage == "audit":
                        continue
                    receiver = BlockedReceiver(c)
                    prior, prior_info = load_prior(c, source, archive, receiver, args.smoke)
                    provenance = dict(commit=state["initial_commit"], code_sha256=hashes,
                        archive_sha256=source_manifest["archive_sha256"], source_records=source_manifest["records"],
                        prior=prior_info, options=options, targets=targets,
                        python=platform.python_version(), numpy=__import__("numpy").__version__,
                        default_view=options["default_input_view"], reference_assisted_csi_adjustment=True,
                        random_stream="SeedSequence([master,split,record,start_ms,0=payload/1=noise,repeat]); public=[master,11]")
                    generate(directories[channel], channel_run, c, options, targets, archive, receiver, prior,
                        backend, identities[channel], provenance, workers=args.workers, worker_threads=args.worker_threads)
                    del prior, receiver
                if args.stage in ("all", "accept"):
                    reports[channel] = accept(directories[channel], channel_run, c, options, backend)
                    print(json.dumps(dict(channel=channel, complete=True, directory=str(directories[channel]),
                        counts=counts(c), assisted_mean_nmse_db={split: reports[channel]["splits"][split]["mean_assisted_nmse_db"] for split in SPLITS}), ensure_ascii=False), flush=True)
        if reports:
            complete = all(channel in reports for channel in options["channels"])
            write_json(run/"dataset_manifest.json", dict(complete=complete, schema=options["schema"], run_identity=state,
                reports=reports, reference_assisted=True, counts={ch: counts(c) for ch, c in configurations.items()}))
        attempt["complete"] = True
    except BaseException as exc:
        attempt["failure"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        attempt["seconds"] = time.perf_counter()-began
        write_json(runtime_path, runtime)
    return reports


if __name__ == "__main__":
    main()
