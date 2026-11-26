"""Train one CRENet on NOF1 and test its frozen weights on NOF1 and NCS1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from data_generation.src.config import ROOT, digest
from data_generation.src.dataset.four_data_protocol import channel_config
from src.common.experiment import file_sha256, write_json
from src.common.numerics import default_device
from src.training.four_data_crenet import train
from src.evaluation.four_data_crenet import evaluate, plot_reports


SOURCE_FILES = (
    "scripts/run/four_data_experiment.py", "src/training/four_data_crenet.py",
    "src/evaluation/four_data_crenet.py", "src/data/four_data_training.py",
    "src/data/four_data.py", "src/algorithms/crenet/model.py",
    "src/algorithms/crenet/operators.py", "src/algorithms/crenet/metrics.py",
    "src/algorithms/crenet/lmmse.py", "src/algorithms/crenet/dpss.py",
    "src/common/numerics.py", "configs/training/four_data_crenet.json",
)


def preflight(directory, channel):
    from src.data.four_data import FourDataDataset
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if (not manifest["complete"] or manifest["smoke"] or manifest["channel"] != channel
            or manifest.get("cp") != 480 or manifest.get("cp_ms") != 80):
        raise ValueError("Expected an accepted formal CP80 four-data dataset; regenerate superseded CP256 data")
    if manifest.get("config_hash") != digest(channel_config(channel)):
        raise ValueError("Training requires an accepted dataset with the current 224-pilot frame configuration")
    for name, expected in manifest["files"].items():
        path = directory/name
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Accepted dataset file changed: {path}")
    print(f"{channel}: accepted HDF5 file hashes verified",flush=True)
    splits = {}
    for split,expected in (("train",525),("validation",75),("test",600)):
        with FourDataDataset(directory/f"{split}.h5") as ds:
            if ds.config["cp"] != 480 or ds.config["fs_baseband"] != 6000:
                raise ValueError("Training requires CP=80 ms (480 samples at 6 kHz)")
            if ds.frame_count != expected or len(ds.snr_db) != 7:
                raise ValueError("Formal DDDD split counts changed")
            splits[split] = set(ds._h["windows/record"][:].tolist())
    if any(splits[a]&splits[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Formal DDDD record leakage")
    return dict(manifest_sha256=file_sha256(directory/"manifest.json"),files=manifest["files"],
                calibration_identities=manifest.get("calibration_identities", {}),
                basis_sha256=manifest["basis_sha256"],
                records={key:sorted(value) for key,value in splits.items()})


def write_provenance(path, value):
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["identity"] != value["identity"]:
            raise ValueError(f"Existing run identity changed: {path}")
    write_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/training/four_data_crenet.json")
    parser.add_argument("--test-channels",choices=("NOF1","NCS1"),nargs="+")
    parser.add_argument("--phase",choices=("all","train","test","preflight"),default="all")
    parser.add_argument("--device",choices=("cpu","cuda"),default=default_device())
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--frame-limit",type=int)
    parser.add_argument("--stage-epochs",type=int,nargs=3)
    args = parser.parse_args()
    if args.frame_limit is not None and args.frame_limit < 1:
        parser.error("frame-limit must be positive")
    if args.stage_epochs is not None and any(e < 1 for e in args.stage_epochs):
        parser.error("all three stages need at least one epoch")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = json.loads(args.config.read_text(encoding="utf-8"))
    if base["train_channel"] != "NOF1":
        parser.error("This experiment uses NOF1 training and validation only")
    if args.stage_epochs:
        base["training"]["stage_epochs"] = args.stage_epochs
    output = args.output or ROOT/base["output"]
    if args.frame_limit is not None and "outputs/smoke/" not in output.as_posix().lower()+"/":
        parser.error("bounded tests must use outputs/smoke")
    output.mkdir(parents=True,exist_ok=True)
    channels = args.test_channels or base["test_channels"]
    if len(set(channels)) != len(channels):
        parser.error("test channels must be distinct")
    sources = {name:file_sha256(ROOT/name) for name in SOURCE_FILES}
    train_channel = base["train_channel"]
    train_directory = ROOT/base["datasets"][train_channel]
    data = {train_channel:preflight(train_directory,train_channel)}
    if args.phase != "train":
        for channel in channels:
            if channel not in data:
                data[channel] = preflight(ROOT/base["datasets"][channel],channel)
    if len({entry["basis_sha256"] for entry in data.values()}) != 1:
        raise ValueError("Train/test channels must share the accepted actual DPSS basis")
    identity = digest(dict(config=base,channel=train_channel,data=data[train_channel],
                           sources=sources,frame_limit=args.frame_limit))
    train_run = output/"training"/train_channel.lower()
    train_run.mkdir(parents=True,exist_ok=True)
    runtime = dict(device=args.device,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    provenance = dict(identity=identity,config=base,data=data[train_channel],sources=sources,
                      train_channel=train_channel,test_used_in_training=False,**runtime)
    write_provenance(train_run/"provenance.json",provenance)
    if args.phase == "preflight":
        write_json(output/"preflight.json",dict(complete=True,data=data,training=provenance))
        print("All selected accepted datasets passed preflight",flush=True)
        return
    if args.phase in ("all","train"):
        write_json(output/"progress.json",dict(stage="training",channel=train_channel))
        train(base,train_directory,train_run,identity,device=args.device,frame_limit=args.frame_limit)
    reports = []
    if args.phase not in ("all","test"):
        return
    checkpoint_path = train_run/"model.pt"
    checkpoint_hash = file_sha256(checkpoint_path)
    for channel in channels:
        directory = ROOT/base["datasets"][channel]
        test_identity = digest(dict(training_identity=identity,checkpoint_sha256=checkpoint_hash,
                                    channel=channel,data=data[channel],frame_limit=args.frame_limit))
        run = output/channel.lower()
        run.mkdir(parents=True,exist_ok=True)
        write_provenance(run/"provenance.json",dict(identity=test_identity,config=base,data=data[channel],
            sources=sources,train_channel=train_channel,training_identity=identity,
            checkpoint_sha256=checkpoint_hash,checkpoint="../training/nof1/model.pt",**runtime))
        write_json(output/"progress.json",dict(stage="testing",channel=channel))
        report = evaluate(base,directory,run,test_identity,device=args.device,frame_limit=args.frame_limit,
                          checkpoint_path=checkpoint_path,training_identity=identity)
        reports.append(report)
        plot_reports([report],run)
        if file_sha256(checkpoint_path) != checkpoint_hash:
            raise ValueError("Frozen NOF1 checkpoint changed during testing")
    if reports:
        plot_reports(reports,output)
        write_json(output/"experiment.json",dict(complete=True,formal=args.frame_limit is None,
            input_view=base["input_view"],channels=[r["channel"] for r in reports],
            train_channel=train_channel,validation_channel=train_channel,
            training_identity=identity,checkpoint_sha256=checkpoint_hash,
            test_used_in_training=False,
            reports={r["channel"]:r["channel"].lower()+"/metrics.json" for r in reports}))
        write_json(output/"progress.json",dict(stage="complete",channels=channels))


if __name__ == "__main__":
    main()
