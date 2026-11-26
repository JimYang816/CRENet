"""Frozen-checkpoint attribution and source-only stress diagnostics for CRENet.

Target test labels are used only to explain the already observed failure.
No optimizer, checkpoint selection, threshold selection, or fitted target statistic
is used here. Counterfactual inference controls are diagnostic ablations.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from data_generation.src.config import ROOT
from src.algorithms.crenet.model import CRENet
from src.algorithms.crenet.metrics import full_matrix_nmse
from src.algorithms.crenet.operators import band_indices
from src.common.experiment import file_sha256, write_json
from src.data.four_data_training import FourDataTrainingSet
from src.evaluation.four_data_crenet import banded_lmmse, bit_errors


def correction_decomposition(output, reference):
    """Exact error change = correction energy + twice error/correction inner product."""
    error = output.initial_channel-reference
    denominator = reference.abs().double().square().sum((-2,-1)).clamp_min(1e-30)
    cols, valid = band_indices(reference.shape[-1],output.channels[0].bandwidth,reference.device)
    rows = torch.arange(reference.shape[-1],device=reference.device)[:,None]
    local_error = error[:,rows,cols]*valid
    local_energy = local_error.abs().double().square().sum((-2,-1))
    energies, crosses, alignments = [], [], []
    for channel in output.channels:
        delta = channel.bands*valid
        energy = delta.abs().double().square().sum((-2,-1))
        inner = (local_error.conj()*delta).real.double().sum((-2,-1))
        energies.append(energy/denominator)
        crosses.append(2*inner/denominator)
        alignments.append(-inner/(local_energy*energy).sqrt().clamp_min(1e-30))
    initial = error.abs().double().square().sum((-2,-1))/denominator
    return dict(initial_nmse=initial,correction_energy=torch.stack(energies,1),
        error_cross_term=torch.stack(crosses,1),alignment=torch.stack(alignments,1),
        correctable_error_fraction=local_energy/(initial*denominator).clamp_min(1e-30))


def perturb_source(inputs, reference, *, error_multiplier=1., phase=0.):
    """Only NOF1 supervision defines synthetic CSI-error stress; y and labels stay paired."""
    value = dict(inputs)
    channel = reference+error_multiplier*(inputs["initial_channel"]-reference)
    rotation = torch.exp(channel.new_tensor(1j*phase))
    value["initial_channel"] = (channel*rotation).to(inputs["initial_channel"].dtype)
    value["observations"] = inputs["observations"]*rotation.to(inputs["observations"].dtype)
    return value,reference*rotation


def channel_statistics(channel, reference, observations, known, variance, covariance):
    n = channel.shape[-1]
    energy = reference.abs().double().square().sum((-2,-1)).clamp_min(1e-30)
    diagonal = reference.diagonal(dim1=-2,dim2=-1)
    data_scale = channel.abs().double().square().sum((-2,-1))/n
    normalized = channel/data_scale.sqrt()[:,None,None]
    diagonal_energy = diagonal.abs().double().square().sum(-1)
    nd = normalized.diagonal(dim1=-2,dim2=-1)
    # Adjacent frequency roughness measures profile/correlation shift after global gain normalization.
    roughness = (nd[:,1:]-nd[:,:-1]).abs().double().square().mean(-1)
    roughness /= nd.abs().double().square().mean(-1).clamp_min(1e-30)
    effective_noise = variance.double()*covariance.diagonal().real.double().mean()/data_scale
    return dict(channel_scale=data_scale.sqrt(),ici_fraction=1-diagonal_energy/energy,
                adjacent_frequency_roughness=roughness,normalized_noise=effective_noise)


def _record(rows, mode, dataset, inputs, targets, scoring, result, reference, *, analytic=None):
    initial,nmse = full_matrix_nmse(result,reference)
    terms = correction_decomposition(result,reference)
    reconstructed = initial[:,None]+terms["correction_energy"]+terms["error_cross_term"]
    torch.testing.assert_close(reconstructed,nmse,rtol=3e-5,atol=2e-7)
    bits = targets["bits"]
    truth = (1-2*bits[...,0])+1j*(1-2*bits[...,1])
    truth = truth/np.sqrt(2)
    mask = inputs["data_mask"]
    stats = channel_statistics(inputs["initial_channel"],reference,inputs["observations"],
                               inputs["known"],inputs["rf_variance"],inputs["unit_covariance"])
    b = len(bits)
    # Only export scalar diagnostics, never target tensors or a modified checkpoint.
    arrays = {k:v.detach().cpu().numpy() for k,v in stats.items()}
    array_initial = initial.cpu().numpy()
    fraction = terms["correctable_error_fraction"].cpu().numpy()
    for layer in range(result.symbols.shape[1]):
        symbols = result.symbols[:,layer,mask]
        errors = bit_errors(symbols,bits).cpu().numpy()
        mse = (symbols-truth).abs().double().square().mean(-1).cpu().numpy()
        predicted_variance = result.variances[:,layer,mask].double().mean(-1).cpu().numpy()
        confidence = result.probabilities[:,layer,mask].amax(-1).double().mean(-1).cpu().numpy()
        channel_nmse = nmse[:,layer].cpu().numpy()
        correction = terms["correction_energy"][:,layer].cpu().numpy()
        cross = terms["error_cross_term"][:,layer].cpu().numpy()
        alignment = terms["alignment"][:,layer].cpu().numpy()
        for j,(si,fi,di) in enumerate(scoring["keys"]):
            frame = dataset.frame_identities[fi]
            rows.append(dict(mode=mode,snr_db=float(dataset.snr_db[si]),frame_index=int(fi),
                symbol_index=int(di),record=frame["record"],start_seconds=frame["start_seconds"],
                repeat=frame["repeat"],layer=layer+1,errors=int(errors[j]),bits=2*bits.shape[1],
                initial_nmse=float(array_initial[j]),corrected_nmse=float(channel_nmse[j]),
                correction_energy=float(correction[j]),error_cross_term=float(cross[j]),
                alignment=float(alignment[j]),correctable_error_fraction=float(fraction[j]),
                symbol_mse=float(mse[j]),predicted_variance=float(predicted_variance[j]),
                confidence=float(confidence[j]),**{k:float(v[j]) for k,v in arrays.items()}))
    if analytic is not None:
        for mode_name,symbols in analytic.items():
            counts = bit_errors(symbols,bits).cpu().numpy()
            for j,(si,fi,di) in enumerate(scoring["keys"]):
                rows.append(dict(mode=mode_name,snr_db=float(dataset.snr_db[si]),frame_index=int(fi),
                    symbol_index=int(di),record=dataset.frame_identities[fi]["record"],
                    start_seconds=dataset.frame_identities[fi]["start_seconds"],
                    repeat=dataset.frame_identities[fi]["repeat"],layer=0,errors=int(counts[j]),
                    bits=2*bits.shape[1],initial_nmse=float(array_initial[j]),
                    corrected_nmse=float(channel_nmse[j]) if "corrected" in mode_name else float(array_initial[j])))


def summarize(rows):
    output = []
    groups = sorted({(r["mode"],r["snr_db"],r["layer"]) for r in rows})
    for mode,snr,layer in groups:
        group = [r for r in rows if (r["mode"],r["snr_db"],r["layer"]) == (mode,snr,layer)]
        item = dict(mode=mode,snr_db=snr,layer=layer,symbol_blocks=len(group),
                    errors=sum(r["errors"] for r in group),bits=sum(r["bits"] for r in group))
        item["ber"] = item["errors"]/item["bits"]
        for key in ("initial_nmse","corrected_nmse","correction_energy","error_cross_term","alignment",
                    "correctable_error_fraction","symbol_mse","predicted_variance","confidence",
                    "channel_scale","ici_fraction","adjacent_frequency_roughness","normalized_noise"):
            if key in group[0]:
                values = np.array([r[key] for r in group])
                item[key] = float(values.mean())
                if key in ("channel_scale","ici_fraction","adjacent_frequency_roughness","normalized_noise"):
                    item[key+"_p10_p50_p90"] = np.quantile(values,[.1,.5,.9]).tolist()
        item["initial_nmse_db"] = float(10*np.log10(item["initial_nmse"]))
        item["corrected_nmse_db"] = float(10*np.log10(item["corrected_nmse"]))
        output.append(item)
    return output


def run(config, output, *, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    checkpoint_path = ROOT/config["checkpoint"]
    checkpoint_hash = file_sha256(checkpoint_path)
    write_json(output/"provenance.json",dict(config=config,checkpoint_sha256=checkpoint_hash,
        sources={name:file_sha256(ROOT/name) for name in (
            "src/evaluation/crenet_generalization.py","src/data/four_data_training.py",
            "src/algorithms/crenet/model.py","src/algorithms/crenet/operators.py",
            "src/algorithms/crenet/metrics.py","src/evaluation/four_data_crenet.py")},
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        device=device,torch=torch.__version__,cuda=torch.version.cuda,
        target_role="post-hoc failure attribution only; no fitting or selection"))
    checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    model = CRENet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    no_correction_full = CRENet(**dict(checkpoint["config"],noise_mode="full")).to(device)
    no_correction_full.load_state_dict(checkpoint["model"])
    no_correction_full.eval().requires_grad_(False)
    once = CRENet(**dict(checkpoint["config"],correction_mode="once")).to(device)
    once.load_state_dict(checkpoint["model"])
    once.eval().requires_grad_(False)
    loading = copy.deepcopy(model)
    with torch.no_grad():
        for layer in loading.equalizers:
            layer.csi_loading_log.copy_(torch.log(torch.expm1(torch.nn.functional.softplus(layer.csi_loading_log)*10)))
    all_reports = []
    begin = time.monotonic()
    for spec in config["datasets"]:
        key = spec["channel"].lower()+"_"+spec["split"]
        folder = output/key
        folder.mkdir(exist_ok=True)
        dataset = FourDataTrainingSet(ROOT/spec["path"],device=device,reference_precision="complex128")
        if (dataset.channel,dataset.split) != (spec["channel"],spec["split"]):
            raise ValueError("Diagnostic split/channel identity mismatch")
        if dataset.channel != "NOF1" and dataset.split != "test":
            raise ValueError("Target data may only be opened as diagnostic test data")
        frames = np.unique(np.linspace(0,dataset.frame_count-1,min(config["frames"],dataset.frame_count),dtype=int))
        factor = torch.linalg.cholesky(dataset.covariance)
        rows = []
        with torch.inference_mode():
            for bi,(inputs,targets,scoring) in enumerate(dataset.batches(config["batch_size"],frame_indices=frames)):
                reference = scoring["reference"]
                original = model(**inputs,return_diagnostics=False)
                h0 = scoring["channel"]
                repaired = original.channels[-1].dense().to(torch.complex128)
                arguments = (scoring["observations"],scoring["known"],scoring["rf_variance"],inputs["data_mask"],factor)
                analytic = dict(b16_input=banded_lmmse(h0,*arguments),
                    b16_corrected=banded_lmmse(repaired,*arguments),
                    full_input=banded_lmmse(h0,*arguments,half_bandwidth=dataset.n))
                _record(rows,"original",dataset,inputs,targets,scoring,original,reference,analytic=analytic)
                del repaired,analytic
                for name,network,enabled in (("no_correction",model,False),("no_correction_full_noise",no_correction_full,False),
                        ("full_noise",no_correction_full,True),("one_correction",once,True),
                        ("no_correction_loading_x10",loading,False)):
                    value = network(**inputs,correction_enabled=enabled,return_diagnostics=False)
                    _record(rows,name,dataset,inputs,targets,scoring,value,reference)
                    del value
                if spec["channel"] == "NOF1":
                    for name,beta,phase in (("source_error_x2",2.,0.),("source_error_x3",3.,0.),
                                          ("source_phase_quarter_turn",1.,np.pi/2)):
                        altered,truth = perturb_source(inputs,reference,error_multiplier=beta,phase=phase)
                        value = model(**altered,return_diagnostics=False)
                        _record(rows,name,dataset,altered,targets,scoring,value,truth)
                        del value,altered,truth
                del original,inputs,targets,scoring,reference,h0
                if bi == 0 or (bi+1)%20 == 0:
                    print(f"{key}: {min((bi+1)*config['batch_size'],len(frames)*28)}/{len(frames)*28} blocks",flush=True)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (folder/"per_symbol.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report = dict(channel=dataset.channel,split=dataset.split,frames=frames.tolist(),
            frame_identities=[dataset.frame_identities[i] for i in frames],snr_db=dataset.snr_db.tolist(),
            summary=summarize(rows),checkpoint_sha256=checkpoint_hash,diagnostic_only=True,
            target_used_for_training_or_selection=False)
        write_json(folder/"diagnosis.json",report)
        all_reports.append(report)
        plot(report,folder)
        del dataset,rows
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError("Frozen checkpoint changed during diagnosis")
    result = dict(complete=True,diagnostic_only=True,source_training_unchanged=True,
        target_used_for_training_or_selection=False,config=config,checkpoint_sha256=checkpoint_hash,
        seconds=time.monotonic()-begin,reports=all_reports)
    write_json(output/"diagnosis.json",result)
    return result


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["summary"]
    styles = (("original",8,"CRENet"),("no_correction",8,"CRENet, no CSI correction"),
              ("b16_input",0,"B16, input CSI"),("b16_corrected",0,"B16, corrected CSI"),
              ("one_correction",8,"CRENet, one correction"))
    for metric,name,label in (("ber","ber_snr","BER"),("corrected_nmse_db","nmse_snr","Mean full-matrix NMSE (dB)")):
        fig,ax = plt.subplots(figsize=(7,4.5))
        for mode,layer,legend in styles:
            group = sorted((r for r in rows if r["mode"]==mode and r["layer"]==layer),key=lambda r:r["snr_db"])
            if not group:
                continue
            ax.plot([r["snr_db"] for r in group],[r[metric] for r in group],"o-",label=legend,markersize=4)
        if metric == "ber":
            ax.set_yscale("log")
        ax.set(xlabel="SNR (dB)",ylabel=label,title=f"{report['channel']} {report['split']} | frozen checkpoint diagnosis")
        ax.grid(True,which="both",alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png","pdf"):
            fig.savefig(Path(output)/f"{name}.{suffix}",dpi=180)
        plt.close(fig)
