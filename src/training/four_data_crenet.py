"""CRENet fitting using only the selected channel's train/validation splits."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.algorithms.crenet.model import CRENet, layerwise_loss
from src.algorithms.crenet.metrics import full_matrix_nmse
from src.common.experiment import write_json
from src.data.four_data_training import FourDataTrainingSet


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def configure_stage(model, stage):
    for name, parameter in model.named_parameters():
        correction = name.startswith("corrector.") or name == "eta_logits"
        parameter.requires_grad_(stage == "C" or (correction if stage == "B" else not correction))


def selection_key(stage, metrics):
    if stage == "A":
        return (metrics["ber"], metrics["loss"])
    if stage == "B":
        return (metrics["corrected_nmse"], metrics["ber"])
    eligible = metrics["corrected_nmse"] < metrics["initial_nmse"] and all(
        row["corrected_nmse"] <= row["initial_nmse"] for row in metrics["per_snr"])
    return (0 if eligible else 1, metrics["ber"], metrics["corrected_nmse"])


def run_epoch(model, dataset, *, stage, batch_size, lambda_h, optimizer=None, seed=None):
    training = optimizer is not None
    model.train(training)
    device = dataset.device
    # errors, bits, sum initial NMSE, sum corrected NMSE, symbol-block count
    totals = torch.zeros(len(dataset.snr_db), 5, device=device, dtype=torch.float64)
    loss_sum, count = 0., 0
    begin = time.monotonic()
    with torch.set_grad_enabled(training):
        for batch_index, (inputs, targets, scoring) in enumerate(dataset.batches(batch_size, seed=seed)):
            result = model(**inputs, correction_enabled=stage != "A", detach_feedback=stage == "B",
                           return_diagnostics=False)
            initial, nmse = full_matrix_nmse(result, targets["reference"])
            if stage == "B":
                loss = nmse.mean()
            else:
                loss = layerwise_loss(result, targets["classes"], None, inputs["data_mask"],
                                      lambda_h=0., objective="bit_ce", final_weight=.9)
                if stage == "C":
                    weights = nmse.new_full((nmse.shape[1],), .1/max(nmse.shape[1]-1, 1))
                    weights[-1] = .9 if len(weights) > 1 else 1.
                    loss = loss+lambda_h*(nmse*weights).sum(1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite four-data CRENet loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               10., error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                symbols = result.symbols[:,-1,inputs["data_mask"]]
                bits = targets["bits"]
                errors = ((symbols.real < 0) != bits[...,0].bool()).sum(-1)
                errors += ((symbols.imag < 0) != bits[...,1].bool()).sum(-1)
                batch_count = len(bits)
                values = torch.stack((errors.double(), errors.new_full(errors.shape, 2*bits.shape[1]).double(),
                                      initial.detach(), nmse[:,-1].detach(),
                                      torch.ones_like(initial)), 1)
                totals.index_add_(0, targets["snr_indices"], values)
                loss_sum += float(loss.detach())*batch_count
                count += batch_count
            del result, loss, inputs, targets, scoring, initial, nmse
            if training and (batch_index+1) % 100 == 0:
                print(f"{dataset.channel} stage {stage}: {count} symbol blocks, "
                      f"{time.monotonic()-begin:.1f}s", flush=True)
    values = totals.cpu().numpy()
    rows = [dict(snr_db=float(snr), errors=int(v[0]), bits=int(v[1]), ber=float(v[0]/v[1]),
                 initial_nmse=float(v[2]/v[4]), corrected_nmse=float(v[3]/v[4]), blocks=int(v[4]))
            for snr,v in zip(dataset.snr_db, values)]
    return dict(loss=loss_sum/count, ber=float(values[:,0].sum()/values[:,1].sum()),
                initial_nmse=float(values[:,2].sum()/values[:,4].sum()),
                corrected_nmse=float(values[:,3].sum()/values[:,4].sum()), per_snr=rows,
                symbol_blocks=count, seconds=time.monotonic()-begin)


def train(config, directory, output, identity, *, device="cpu", frame_limit=None):
    """Never open test.h5; selection and early stopping use validation only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output/"training.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["identity"] != identity:
            raise ValueError("Training output identity changed; use a new output directory")
        if report["complete"]:
            return report
    options = config["training"]
    torch.manual_seed(options["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options["seed"])
    model = CRENet(**config["model"]).to(device)
    state_path = output/"resume.pt"
    state = dict(identity=identity, stage_index=0, next_epoch=1, history=[],
                 stale=0, best_key=None, optimizer=None)
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        torch.set_rng_state(state["rng"])
        if state.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    train_set = FourDataTrainingSet(Path(directory)/"train.h5", device=device,
        input_view=config["input_view"], frame_limit=frame_limit)
    validation = FourDataTrainingSet(Path(directory)/"validation.h5", device=device,
        input_view=config["input_view"], frame_limit=frame_limit)
    if set(train_set.records.tolist()) & set(validation.records.tolist()):
        raise ValueError("Training/validation record leakage")
    batch_size, lambda_h = options["batch_size"], options["lambda_h"]
    begin = time.monotonic()
    for stage_index in range(state["stage_index"], 3):
        stage = "ABC"[stage_index]
        configure_stage(model, stage)
        rate = options[{"A":"learning_rate", "B":"stage_b_learning_rate", "C":"stage_c_learning_rate"}[stage]]
        groups = []
        for correction in (False, True):
            parameters = [p for n,p in model.named_parameters() if p.requires_grad
                          and (n.startswith("corrector.") or n == "eta_logits") == correction]
            if parameters:
                groups.append(dict(params=parameters, lr=options["stage_c_correction_learning_rate"]
                                   if stage == "C" and correction else rate))
        optimizer = torch.optim.Adam(groups)
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        best_path = output/f"best_{stage}.pt"
        if state["best_key"] is None:
            initial_metrics = run_epoch(model, validation, stage=stage, batch_size=batch_size, lambda_h=lambda_h)
            state["best_key"] = selection_key(stage, initial_metrics)
            save_checkpoint(best_path, dict(identity=identity, model=cpu_state(model),
                                           stage=stage, epoch=0, validation=initial_metrics))
        for epoch in range(state["next_epoch"], options["stage_epochs"][stage_index]+1):
            training_metrics = run_epoch(model, train_set, stage=stage, batch_size=batch_size,
                lambda_h=lambda_h, optimizer=optimizer, seed=options["seed"]+10000*stage_index+epoch)
            metrics = run_epoch(model, validation, stage=stage, batch_size=batch_size, lambda_h=lambda_h)
            key = selection_key(stage, metrics)
            improved = key < tuple(state["best_key"])
            if improved:
                state["best_key"], state["stale"] = key, 0
                save_checkpoint(best_path, dict(identity=identity, model=cpu_state(model),
                                               stage=stage, epoch=epoch, validation=metrics))
            else:
                state["stale"] += 1
            state["history"].append(dict(stage=stage, epoch=epoch, train=training_metrics,
                                         validation=metrics, selected=improved))
            print(f"{train_set.channel} {stage} epoch {epoch}: validation BER={metrics['ber']:.6g}, "
                  f"NMSE={10*np.log10(metrics['corrected_nmse']):.3f} dB, selected={improved}", flush=True)
            state.update(stage_index=stage_index, next_epoch=epoch+1, model=cpu_state(model),
                         optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            save_checkpoint(state_path, state)
            write_json(output/"history.json", state["history"])
            if state["stale"] >= options["patience"]:
                break
        chosen = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(chosen["model"])
        state.update(stage_index=stage_index+1, next_epoch=1, best_key=None, stale=0,
                     model=cpu_state(model), optimizer=None, rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        save_checkpoint(state_path, state)
    chosen = torch.load(output/"best_C.pt", map_location="cpu", weights_only=False)
    save_checkpoint(output/"model.pt", dict(chosen, config=copy.deepcopy(config["model"]),
                                           input_view=config["input_view"], protocol="DDDD"))
    report = dict(identity=identity, channel=train_set.channel, complete=True,
                  formal=frame_limit is None, test_used_in_training=False,
                  selection="validation only; A BER, B full-matrix NMSE, C BER with per-SNR CSI improvement",
                  selected_stage=chosen["stage"], selected_epoch=chosen["epoch"],
                  validation=chosen["validation"], config=config,
                  training_data=train_set.identity, validation_data=validation.identity,
                  history=state["history"], seconds=time.monotonic()-begin)
    write_json(report_path, report)
    return report
