"""Paired, validation-only diagnosis of four-data CRENet detector training."""
from __future__ import annotations

import copy
from itertools import islice
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from data_generation.src.config import ROOT
from src.algorithms.crenet.model import CRENet, layerwise_loss
from src.common.experiment import file_sha256, write_json
from src.data.four_data_training import FourDataTrainingSet
from src.training.four_data_crenet import configure_stage, run_epoch


# Rates are absolute Adam rates. All candidates use the same initial model and
# the same randomly ordered training batches for each interval. The optional
# third value selects the smoothed-BER objective.
VARIANTS = {
    "all_1e-3": (1e-3, 1e-3),
    "all_1e-4": (1e-4, 1e-4),
    "scalars_1e-3": (1e-3, 0.),
    "neural_1e-3": (0., 1e-3),
    "scaled_neural_1e-2": (1e-3, 1e-5),
    "softber_all_1e-3": (1e-3, 1e-3, "softber"),
    "softber_all_1e-4": (1e-4, 1e-4, "softber"),
    "softber_scaled": (1e-3, 1e-4, "softber"),
    "softber_neural_1e-4": (0., 1e-4, "softber"),
}

SOURCE_FILES = (
    "scripts/evaluate/four_data_stage_a.py", "src/evaluation/four_data_stage_a.py",
    "src/training/four_data_crenet.py", "src/data/four_data_training.py", "src/data/four_data.py",
    "src/algorithms/crenet/model.py", "src/algorithms/crenet/operators.py",
    "src/algorithms/crenet/lmmse.py", "src/algorithms/crenet/dpss.py", "src/algorithms/crenet/metrics.py",
)


def detector_group(name):
    if not name.startswith("equalizers."):
        return None
    return "neural" if ".hidden." in name or ".classifier." in name else "scalars"


def detector_optimizer(model, rates):
    """Freeze CSI repair and select physical scalars and/or local neural weights."""
    configure_stage(model, "A")
    grouped = {"scalars": [], "neural": []}
    for name, parameter in model.named_parameters():
        group = detector_group(name)
        if group is not None:
            parameter.requires_grad_(rates[group] > 0)
            if parameter.requires_grad:
                grouped[group].append(parameter)
    groups = [dict(params=parameters, lr=rates[name], name=name)
              for name, parameters in grouped.items() if parameters]
    if not groups:
        raise ValueError("At least one detector group must be trained")
    return torch.optim.Adam(groups)


def detector_changes(model, initial):
    """Report L2 changes and number of changed tensors for each detector group."""
    result = {}
    for group in ("scalars", "neural"):
        squared, changed, count = 0., 0, 0
        for name, parameter in model.named_parameters():
            if detector_group(name) != group:
                continue
            difference = parameter.detach().cpu().double() - initial[name].double()
            squared += float(difference.square().sum())
            changed += int(torch.any(difference != 0))
            count += 1
        result[group] = dict(l2=squared**.5, changed_tensors=changed, tensors=count)
    return result


def soft_ber_loss(output, classes, data_mask, *, slope=4.):
    """Smooth the two hard bit decisions, retaining the usual layer weights."""
    if slope <= 0 or output.log_probabilities is None:
        raise ValueError("A positive slope and layer log probabilities are required")
    p = output.log_probabilities[:, :, data_mask].exp()
    if classes.shape != (len(p), p.shape[-2]):
        raise ValueError("Classes must cover data carriers in physical order")
    margin0 = p[..., 0]+p[..., 1]-p[..., 2]-p[..., 3]
    margin1 = p[..., 0]+p[..., 2]-p[..., 1]-p[..., 3]
    signed0 = (1-2*(classes//2))[:, None, :] * margin0
    signed1 = (1-2*(classes%2))[:, None, :] * margin1
    terms = .5*(torch.sigmoid(-slope*signed0)+torch.sigmoid(-slope*signed1))
    terms = terms.mean((0, 2))
    weights = terms.new_full((len(terms),), .1/max(len(terms)-1, 1))
    weights[-1] = .9 if len(terms) > 1 else 1.
    return (terms*weights).sum()


def train_interval(model, dataset, optimizer, *, batch_size, steps, seed, objective="bit_ce"):
    """Train on a seeded subset of batches; never read validation or test labels."""
    model.train()
    loss_sum, examples, used = 0., 0, 0
    gradient = {}
    begin = time.monotonic()
    for inputs, targets, _ in islice(dataset.batches(batch_size, seed=seed), steps):
        result = model(**inputs, correction_enabled=False, return_diagnostics=False)
        ce = layerwise_loss(result, targets["classes"], None, inputs["data_mask"],
                            lambda_h=0., objective="bit_ce", final_weight=.9)
        if objective == "bit_ce":
            loss = ce
        elif objective == "softber":
            # CE still supplies gradients for confidently wrong decisions.
            loss = soft_ber_loss(result, targets["classes"], inputs["data_mask"]) + .1*ce
        else:
            raise ValueError("Unknown diagnostic training objective")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite detector diagnosis loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not gradient:
            for name in ("scalars", "neural"):
                squared = sum(float(p.grad.detach().double().square().sum())
                              for n, p in model.named_parameters()
                              if detector_group(n) == name and p.grad is not None)
                gradient[name] = squared**.5
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       10., error_if_nonfinite=True)
        optimizer.step()
        count = len(targets["classes"])
        loss_sum += float(loss.detach()) * count
        examples += count
        used += 1
    if used != steps:
        raise ValueError(f"Requested {steps} batches, but the dataset supplied {used}")
    return dict(loss=loss_sum/examples, objective=objective, examples=examples, steps=used,
                first_gradient_l2=gradient, seconds=time.monotonic()-begin)


def _verified_source(directory, previous):
    provenance = json.loads((previous/"provenance.json").read_text(encoding="utf-8"))
    manifest = directory/"manifest.json"
    if file_sha256(manifest) != provenance["data"]["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the original experiment")
    # Deliberately do not open or hash test.h5 during model development.
    for name in ("model.h5", "train.h5", "validation.h5"):
        expected = provenance["data"]["files"][name]
        path = directory/name
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Dataset file differs from the original experiment: {name}")
    checkpoint = torch.load(previous/"best_A.pt", map_location="cpu", weights_only=False)
    if (checkpoint["identity"] != provenance["identity"] or checkpoint["stage"] != "A"
            or checkpoint["epoch"] != 0):
        raise ValueError("Expected the original stage-A epoch-0 checkpoint")
    return provenance, checkpoint


def run(config, directory, previous, output, *, device, variants, intervals, steps, batch_size):
    """Compare paired stage-A updates against the frozen original initialization."""
    directory, previous, output = map(Path, (directory, previous, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty diagnosis directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance, checkpoint = _verified_source(directory, previous)
    if config["model"] != provenance["config"]["model"] or config["input_view"] != provenance["config"]["input_view"]:
        raise ValueError("Diagnostic model or input view differs from the original experiment")
    train = FourDataTrainingSet(directory/"train.h5", device=device, input_view=config["input_view"])
    validation = FourDataTrainingSet(directory/"validation.h5", device=device, input_view=config["input_view"])
    if train.channel != validation.channel or set(train.records.tolist()) & set(validation.records.tolist()):
        raise ValueError("Training and validation must have the same channel and disjoint records")
    model = CRENet(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    baseline = run_epoch(model, validation, stage="A", batch_size=batch_size, lambda_h=0.)
    recorded = checkpoint["validation"]
    summary = dict(schema="four-data-stage-a-diagnosis-v1", complete=False,
        channel=train.channel, input_view=config["input_view"], test_used=False,
        original_identity=provenance["identity"], original_checkpoint_sha256=file_sha256(previous/"best_A.pt"),
        manifest_sha256=provenance["data"]["manifest_sha256"],
        source_checkpoint_epoch=checkpoint["epoch"], recorded_validation=recorded, baseline=baseline,
        validation_error_count_difference=sum(x["errors"] for x in baseline["per_snr"])
                                          -sum(x["errors"] for x in recorded["per_snr"]),
        device=str(device), torch=torch.__version__, batch_size=batch_size, steps_per_interval=steps,
        intervals=intervals, variants={}, config=copy.deepcopy(config),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={name:file_sha256(ROOT/name) for name in SOURCE_FILES})
    write_json(output/"summary.json", summary)
    print(f"{train.channel} baseline: BER={baseline['ber']:.8f}, loss={baseline['loss']:.6f}", flush=True)
    for variant in variants:
        specification = VARIANTS[variant]
        scalar_rate, neural_rate = specification[:2]
        objective = specification[2] if len(specification) > 2 else "bit_ce"
        rates = dict(scalars=scalar_rate, neural=neural_rate)
        model.load_state_dict(checkpoint["model"])
        optimizer = detector_optimizer(model, rates)
        record = dict(rates=rates, objective=objective, intervals=[], best_interval=None, best_validation=None)
        summary["variants"][variant] = record
        for index in range(1, intervals+1):
            train_metrics = train_interval(model, train, optimizer, batch_size=batch_size, steps=steps,
                seed=config["training"]["seed"]+index, objective=objective)
            metrics = run_epoch(model, validation, stage="A", batch_size=batch_size, lambda_h=0.)
            changes = detector_changes(model, initial)
            row = dict(interval=index, updates=index*steps, training=train_metrics,
                       validation=metrics, changes=changes)
            record["intervals"].append(row)
            if record["best_validation"] is None or (metrics["ber"], metrics["loss"]) < (
                    record["best_validation"]["ber"], record["best_validation"]["loss"]):
                record["best_interval"], record["best_validation"] = index, metrics
                torch.save(dict(variant=variant, interval=index, model={k:v.detach().cpu().clone()
                           for k,v in model.state_dict().items()}, validation=metrics,
                           source_identity=provenance["identity"]), output/f"{variant}.pt")
            write_json(output/"summary.json", summary)
            print(f"{train.channel} {variant} update {index*steps}: BER={metrics['ber']:.8f}, "
                  f"loss={metrics['loss']:.6f}, scalar_change={changes['scalars']['l2']:.4g}, "
                  f"neural_change={changes['neural']['l2']:.4g}", flush=True)
    summary["complete"] = True
    write_json(output/"summary.json", summary)
    plot(summary, output)
    return summary


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("initial", summary["baseline"])] + [
        (name, record["best_validation"]) for name, record in summary["variants"].items()]
    for kind in ("ber", "nmse"):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for label, metrics in rows:
            snr = [r["snr_db"] for r in metrics["per_snr"]]
            values = ([max(r["ber"], .5/r["bits"]) for r in metrics["per_snr"]] if kind == "ber" else
                      [10*np.log10(max(r["corrected_nmse"], 1e-30)) for r in metrics["per_snr"]])
            line, = axis.plot(snr, values, marker="o", markersize=3, label=label)
            if kind == "ber":
                zeros = [i for i, row in enumerate(metrics["per_snr"]) if row["ber"] == 0]
                axis.scatter([snr[i] for i in zeros], [values[i] for i in zeros],
                             facecolor="white", edgecolor=line.get_color(), s=16, zorder=4)
        if kind == "ber":
            axis.set_yscale("log")
        axis.set(xlabel="SNR (dB)", ylabel="BER" if kind == "ber" else "Mean full-matrix NMSE (dB)",
                 title=f"{summary['channel']} validation: stage-A detector diagnosis")
        axis.grid(True, which="both", alpha=.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(Path(output)/f"{kind}_snr.{suffix}", dpi=180)
        plt.close(fig)
