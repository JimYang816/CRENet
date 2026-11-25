"""Exhaustive paired four-data test scoring and BER/NMSE figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.algorithms.crenet.model import CRENet
from src.algorithms.crenet.metrics import full_matrix_nmse
from src.common.experiment import file_sha256, write_json
from src.data.four_data_training import FourDataTrainingSet


ALGORITHMS = ("single_tap_lmmse", "full_matrix_lmmse", "crenet", "perfect_csi_full_matrix_lmmse")
LABELS = ("Single-tap LMMSE", "Full-matrix LMMSE", "CRENet", "Full-matrix LMMSE (perfect CSI)")


def full_matrix_lmmse(channel, observations, known, variance, data_mask, factor):
    """All receive rows, all data columns, pilot cancellation, full colored noise."""
    h = torch.linalg.solve_triangular(factor, channel[...,data_mask], upper=False)
    y = observations-(channel@known[...,None])[...,0]
    y = torch.linalg.solve_triangular(factor, y[...,None], upper=False)
    gram, matched = h.mH@h, h.mH@y
    gram.diagonal(dim1=-2, dim2=-1).add_(variance[:,None])
    return torch.linalg.solve(gram, matched)[...,0]


def bit_errors(symbols, bits):
    return (((symbols.real < 0) != bits[...,0].bool()).sum(-1)
            + ((symbols.imag < 0) != bits[...,1].bool()).sum(-1))


def evaluate(config, directory, output, identity, *, device="cpu", frame_limit=None):
    output = Path(output)
    report_path = output/"metrics.json"
    checkpoint_hash = file_sha256(output/"model.pt")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["identity"] != identity or report["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Existing evaluation identity changed")
        if report["complete"]:
            return report
    checkpoint = torch.load(output/"model.pt", map_location="cpu", weights_only=False)
    if checkpoint["identity"] != identity or checkpoint["input_view"] != config["input_view"]:
        raise ValueError("Four-data checkpoint/input identity mismatch")
    model = CRENet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = FourDataTrainingSet(Path(directory)/"test.h5", device=device,
        input_view=config["input_view"], frame_limit=frame_limit, reference_precision="complex128")
    n_snr, n_frame = len(dataset.snr_db), dataset.frame_count
    errors = {key:np.zeros((n_snr,n_frame,4), np.int64) for key in ALGORITHMS}
    nmse = np.zeros((n_snr,n_frame,4,2), np.float64)
    complete = np.zeros((n_snr,n_frame,4), bool)
    factor = torch.linalg.cholesky(dataset.covariance)
    begin = time.monotonic()
    with torch.no_grad():
        for batch_index, (inputs, targets, scoring) in enumerate(dataset.batches(config["evaluation"]["batch_size"])):
            h, y, known, var = (scoring[k] for k in ("channel", "observations", "known", "rf_variance"))
            mask = inputs["data_mask"]
            diagonal = h.diagonal(dim1=-2, dim2=-1)[:,mask]
            single = diagonal.conj()*y[:,mask]/(diagonal.abs().square()
                         + var[:,None]*dataset.covariance.diagonal().real[mask])
            full = full_matrix_lmmse(h, y, known, var, mask, factor)
            perfect = full_matrix_lmmse(scoring["reference"], y, known, var, mask, factor)
            # References and labels do not enter this call.
            result = model(**inputs, return_diagnostics=False)
            learned = result.symbols[:,-1,mask]
            initial, corrected = full_matrix_nmse(result, scoring["reference"])
            si,fi,di = scoring["keys"].T
            if complete[si,fi,di].any():
                raise ValueError("Duplicate four-data test symbol")
            for key, symbols in zip(ALGORITHMS, (single,full,learned,perfect)):
                errors[key][si,fi,di] = bit_errors(symbols, targets["bits"]).cpu().numpy()
            nmse[si,fi,di] = torch.stack((initial,corrected[:,-1]),1).cpu().numpy()
            complete[si,fi,di] = True
            if batch_index == 0 or (batch_index+1) % 100 == 0:
                print(f"{dataset.channel} test: {int(complete.sum())}/{complete.size} symbol blocks, "
                      f"{time.monotonic()-begin:.1f}s", flush=True)
            del result, inputs, targets, scoring, h, y, known, var, single, full, perfect, learned
    if not complete.all():
        raise ValueError("Missing four-data test symbols")
    mean_nmse = nmse.mean(axis=(1,2))
    deviation = float(np.max(np.abs(10*np.log10(mean_nmse[:,0]/dataset.expected_nmse))))
    if frame_limit is None and deviation > 1e-4:
        raise ValueError(f"Input CSI NMSE disagrees with accepted dataset: {deviation} dB")
    bits_per_snr = n_frame*4*dataset.config["data_count"]*2
    curves = {key:dict(errors=value.sum(axis=(1,2)).tolist(),
                      ber=(value.sum(axis=(1,2))/bits_per_snr).tolist()) for key,value in errors.items()}
    report = dict(identity=identity, complete=True, formal=frame_limit is None,
        channel=dataset.channel, input_view=dataset.input_view, protocol="DDDD",
        cp=dataset.config["cp"], fs_baseband=dataset.config["fs_baseband"],
        reference_assisted_input_uses_full_reference=config["input_view"] == "reference_assisted",
        checkpoint_sha256=checkpoint_hash, selected_epoch=checkpoint["epoch"],
        snr_db=dataset.snr_db.tolist(), frames_per_snr=n_frame,
        symbol_blocks_per_snr=4*n_frame, bits_per_snr=bits_per_snr, curves=curves,
        initial_nmse_linear=mean_nmse[:,0].tolist(), corrected_nmse_linear=mean_nmse[:,1].tolist(),
        initial_nmse_db=(10*np.log10(mean_nmse[:,0])).tolist(),
        corrected_nmse_db=(10*np.log10(mean_nmse[:,1])).tolist(),
        nmse_gain_db=(10*np.log10(mean_nmse[:,0]/mean_nmse[:,1])).tolist(),
        perfect_csi_nmse_linear=0., accepted_input_nmse_max_deviation_db=deviation,
        nmse_definition="mean per-symbol full-matrix linear error ratio, then 10*log10",
        baseline_precision="complex128; full noise covariance; thermal-noise LMMSE regularization",
        crenet_precision="complex64; diagonal noise weighting; final layer CSI correction",
        frames=dataset.frame_identities, seconds=time.monotonic()-begin)
    np.savez_compressed(output/"frame_metrics.npz", **errors, nmse_linear=nmse, complete=complete)
    with (output/"per_frame.csv").open("w",newline="",encoding="utf-8") as stream:
        fields = ["snr_db","frame_index","record","start_seconds","repeat","bits",*ALGORITHMS,
                  "initial_nmse_linear","corrected_nmse_linear"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for si,snr in enumerate(dataset.snr_db):
            for fi,frame in enumerate(dataset.frame_identities):
                writer.writerow(dict(snr_db=snr, frame_index=fi, record=frame["record"],
                    start_seconds=frame["start_seconds"], repeat=frame["repeat"], bits=bits_per_snr//n_frame,
                    **{key:int(value[si,fi].sum()) for key,value in errors.items()},
                    initial_nmse_linear=float(nmse[si,fi,:,0].mean()),
                    corrected_nmse_linear=float(nmse[si,fi,:,1].mean())))
    write_json(report_path,report)
    return report


def plot_reports(reports, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "pdf.fonttype":42})
    for kind in ("ber", "nmse"):
        fig,axes = plt.subplots(1,len(reports),figsize=(6*len(reports),4.5),squeeze=False)
        for axis,report in zip(axes[0],reports):
            snr = report["snr_db"]
            if kind == "ber":
                for key,label,marker in zip(ALGORITHMS,LABELS,("o","s","^","D")):
                    ber = np.asarray(report["curves"][key]["ber"])
                    line, = axis.semilogy(snr,np.maximum(ber,.5/report["bits_per_snr"]),
                                          marker=marker,label=label,linewidth=1.5,markersize=4)
                    zeros = ber == 0
                    axis.scatter(np.asarray(snr)[zeros],np.full(zeros.sum(),.5/report["bits_per_snr"]),
                                 facecolor="white",edgecolor=line.get_color(),marker=marker,zorder=4)
                axis.set_ylabel("BER")
                axis.text(.02,.02,"Hollow marker: zero errors, shown at 0.5 / bits",transform=axis.transAxes,fontsize=7)
            else:
                axis.plot(snr,report["initial_nmse_db"],"o--",label="Input CSI (shared by LMMSE and CRENet)")
                axis.plot(snr,report["corrected_nmse_db"],"s-",label="CRENet corrected CSI")
                axis.set_ylabel("Mean full-matrix NMSE (dB)")
                axis.text(.02,.02,"Perfect CSI: NMSE = 0 (linear)",transform=axis.transAxes,fontsize=8)
            axis.set(title=report["channel"],xlabel="SNR (dB)",xticks=snr)
            axis.grid(True,which="both",alpha=.25)
            axis.legend(fontsize=8)
        # Reports without an explicit CP are the retained historical CP256 runs.
        prefixes = {(r.get("cp", 256), r.get("fs_baseband", 6000)) for r in reports}
        if len(prefixes) != 1:
            raise ValueError("Reports use different cyclic-prefix configurations")
        cp, fs = prefixes.pop()
        fig.suptitle(f"DDDD | CP={cp} ({1000*cp/fs:g} ms) | Q=24 | 192 raw delays | reference-assisted input")
        fig.tight_layout()
        for suffix in ("png","pdf"):
            fig.savefig(output/f"{kind}_snr.{suffix}",dpi=220,bbox_inches="tight")
        plt.close(fig)
    with (output/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel","snr_db","bits",*[key+"_ber" for key in ALGORITHMS],
                         "initial_nmse_db","corrected_nmse_db","nmse_gain_db"])
        for report in reports:
            for si,snr in enumerate(report["snr_db"]):
                writer.writerow([report["channel"],snr,report["bits_per_snr"],
                    *[report["curves"][key]["ber"][si] for key in ALGORITHMS],
                    report["initial_nmse_db"][si],report["corrected_nmse_db"][si],report["nmse_gain_db"][si]])
