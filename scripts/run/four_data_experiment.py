"""Run fresh per-channel DDDD training and exhaustive paired final testing."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess

import torch

from data_generation.src.config import ROOT, digest
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
)


def preflight(directory, channel):
    from src.data.four_data import FourDataDataset
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"] or manifest["smoke"] or manifest["channel"] != channel:
        raise ValueError("Expected an accepted formal four-data dataset")
    for name, expected in manifest["files"].items():
        path = directory/name
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Accepted dataset file changed: {path}")
    print(f"{channel}: accepted HDF5 file hashes verified",flush=True)
    splits = {}
    for split,expected in (("train",525),("validation",75),("test",600)):
        with FourDataDataset(directory/f"{split}.h5") as ds:
            if ds.frame_count != expected or len(ds.snr_db) != 7:
                raise ValueError("Formal DDDD split counts changed")
            splits[split] = set(ds._h["windows/record"][:].tolist())
    if any(splits[a]&splits[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Formal DDDD record leakage")
    return dict(manifest_sha256=file_sha256(directory/"manifest.json"),files=manifest["files"],
                records={key:sorted(value) for key,value in splits.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/training/four_data_crenet.json")
    parser.add_argument("--channel",choices=("NOF1","NCS1","all"),default="all")
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
    if args.stage_epochs:
        base["training"]["stage_epochs"] = args.stage_epochs
    output = args.output or ROOT/base["output"]
    if args.frame_limit is not None and "outputs/smoke/" not in output.as_posix().lower()+"/":
        parser.error("bounded tests must use outputs/smoke")
    output.mkdir(parents=True,exist_ok=True)
    channels = list(base["datasets"]) if args.channel == "all" else [args.channel]
    sources = {name:file_sha256(ROOT/name) for name in SOURCE_FILES}
    reports = []
    for channel in channels:
        config = copy.deepcopy(base)
        for section,values in base.get("channel_overrides",{}).get(channel,{}).items():
            config[section].update(values)
        directory = ROOT/config["datasets"][channel]
        data = preflight(directory,channel)
        identity = digest(dict(config=config,channel=channel,data=data,sources=sources,frame_limit=args.frame_limit))
        run = output/channel.lower()
        run.mkdir(parents=True,exist_ok=True)
        provenance = dict(identity=identity,config=config,data=data,sources=sources,device=args.device,
            torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
            git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
        write_json(run/"provenance.json",provenance)
        if args.phase == "preflight":
            print(json.dumps(provenance,ensure_ascii=False,indent=2))
            continue
        if args.phase in ("all","train"):
            train(config,directory,run,identity,device=args.device,frame_limit=args.frame_limit)
        if args.phase in ("all","test"):
            report = evaluate(config,directory,run,identity,device=args.device,frame_limit=args.frame_limit)
            reports.append(report)
            plot_reports([report],run)
    if reports:
        plot_reports(reports,output)
        write_json(output/"experiment.json",dict(complete=True,formal=args.frame_limit is None,
            input_view=base["input_view"],channels=[r["channel"] for r in reports],
            reports={r["channel"]:str((output/r["channel"].lower()/"metrics.json").relative_to(ROOT)) for r in reports}))


if __name__ == "__main__":
    main()
