"""Frozen physical joint-moment CSI, compared with both saved 4D baselines."""
from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import subprocess
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT, digest
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.four_data import BlockedFourDataReceiver
from data_generation.src.receiver.four_data_covariance import (
    JointFourDataEstimator, JointMoments, collect_joint_moments,
)
from data_generation.src.receiver.four_data_rotated import make_full_band_frame
from data_generation.src.transmitter.ofdm import receive_symbols
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.evaluation.four_data_mmse import ALGORITHMS, _summary, experiment_config, window_ids
from src.evaluation.four_data_rotated import PARAMETERS, baseline_report, score
from src.evaluation.four_data_rotated_audit import coefficient_power, thermal_mode_energy


# Declared before final evaluation; neither grid nor objective reads test CSI.
ALPHAS = (.1, .4, .7, 1.)
RHOS = (.05, .2, .5, .8)
CODE = ("data_generation/src/receiver/four_data.py", "data_generation/src/receiver/four_data_covariance.py",
        "src/evaluation/four_data_covariance.py", "scripts/evaluate/four_data_covariance.py",
        "src/evaluation/four_data_rotated.py", "src/evaluation/four_data_rotated_audit.py",
        "data_generation/src/receiver/chain.py", "data_generation/src/receiver/estimation.py",
        "src/algorithms/single_tap_mmse/equalizer.py", "src/algorithms/full_matrix_lmmse/equalizer.py",
        "src/algorithms/likelihood.py", "src/common/numerics.py")


def channel_report(root, channel):
    path = root/channel.lower()/"metrics.json"
    if not path.exists():
        path = root/f"{channel.lower()}_metrics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report["complete"]:
        raise ValueError("Need a completed source experiment")
    return path, report


class RecordSubset:
    """Enforce the prior/validation boundary at the source-loading interface."""

    def __init__(self, archive, records):
        self.archive, self.records, self.channel = archive, set(records), archive.channel

    def load(self, record):
        if record not in self.records:
            raise ValueError("Record is outside the independent calibration subset")
        return self.archive.load(record)


def validation_data(prior, known, masks, receiver, archive, identities, *, frame_factory):
    config = prior.model.c
    q, n, length = prior.model.count, config["nfft"], prior.model.length
    basis = prior.model.sampled_basis.reshape(4, n, q)
    clean, variances, cross, energy = [], [], [], []
    for record, start in identities:
        sounding = archive.load(record)
        frame = frame_factory(config, record, start, known, masks)
        origin = start-config["guard_samples"]/config["fs_baseband"]
        physical = receiver.replay_frame(sounding, origin, frame)
        y = receive_symbols(receiver.chain.receive(physical), config)
        # In coordinates y/g, exact power matching cancels from both clean
        # observations and RF variance. No random validation noise is required.
        clean.append(y)
        variances.append(float(np.mean(abs(y)**2))/receiver.chain.noise_gain
                         /10**(np.asarray(config["snr_db"])/10))
        truth = np.stack([receiver.reference_taps(sounding, start, s)[0] for s in range(4)])
        cross.append(np.einsum("dnq,dnl->dql", basis.conj(), truth[..., :length], optimize=True))
        energy.append(np.sum(abs(truth)**2, axis=(1, 2)))
        print(f"{archive.channel} independent validation {record:03d}@{start}s", flush=True)
    return {name:np.stack(value) for name,value in
            (("clean", clean), ("variances", variances), ("cross", cross), ("energy", energy))}


def validation_score(estimator, data):
    model = estimator.prior.model
    basis = model.sampled_basis.reshape(4, model.c["nfft"], model.count)
    variance = data["variances"]
    y = np.broadcast_to(data["clean"][:, None], (*variance.shape, 4, model.c["nfft"]))
    coefficients = estimator.estimate_coefficients(y, variance)
    power = coefficient_power(basis, coefficients)
    correlation = np.einsum("bsql,bdql->bsd", coefficients.conj(), data["cross"], optimize=True).real
    clean_error = np.maximum((power-2*correlation+data["energy"][:, None])/data["energy"][:, None], 0)
    modes = thermal_mode_energy(estimator, basis)
    values = estimator.backend.numpy(estimator.values)
    denominator = values[None, None]+variance[..., None]+estimator.ridge
    thermal = variance[..., None]*(modes[None, None]/denominator[..., None, :]**2).sum(axis=-1)/data["energy"][:, None]
    curve = (clean_error+thermal).mean(axis=(0, 2))
    return float(curve.mean()), curve


def save_estimator(path, estimator, identity, selection):
    with h5py.File(path, "w") as saved:
        saved.attrs.update(complete=False, identity=identity, diagnostics_json=json.dumps(estimator.diagnostics),
                           selection_json=json.dumps(selection))
        for name in ("values", "projected", "gain"):
            saved.create_dataset(name, data=estimator.backend.numpy(getattr(estimator, name)))
        saved.attrs["complete"] = True


def load_estimator(path, prior, known, masks, backend, identity):
    with h5py.File(path, "r") as saved:
        if not saved.attrs.get("complete") or saved.attrs["identity"] != identity:
            raise ValueError("Frozen estimator source identity mismatch")
        estimator = object.__new__(JointFourDataEstimator)
        estimator.prior, estimator.backend, estimator.known = prior, backend, known
        estimator.rows = [np.flatnonzero(m) for m in masks["tracking"]]
        estimator.diagnostics = json.loads(saved.attrs["diagnostics_json"])
        estimator.ridge = estimator.diagnostics["whitened_ridge"]
        for name in ("values", "projected", "gain"):
            setattr(estimator, name, backend.array(saved[name][:]))
        return estimator, json.loads(saved.attrs["selection_json"])


def calibrate(run, identity, prior, known, masks, noise, receiver, archive, prior_ids, validation_ids, backend, *, smoke, frame_factory):
    frozen = run/"frozen_estimator.h5"
    if frozen.exists():
        return load_estimator(frozen, prior, known, masks, backend, identity)
    raw_path = run/"frozen_moments.h5"
    if raw_path.exists():
        with h5py.File(raw_path, "r") as saved:
            if not saved.attrs.get("complete") or saved.attrs["identity"] != identity:
                raise ValueError("Frozen joint moments source mismatch")
            metadata = json.loads(saved.attrs["metadata_json"])
            raw = {name:saved[f"moments/{name}"][:] for name in saved["moments"]}
    else:
        raw, metadata = collect_joint_moments(prior, known, masks, noise, receiver,
            RecordSubset(archive, {r for r,_ in prior_ids}), prior_ids, backend)
        with h5py.File(raw_path, "w") as saved:
            saved.attrs.update(complete=False, identity=identity, metadata_json=json.dumps(metadata))
            saved.create_dataset("basis", data=prior.model.basis)
            saved.create_dataset("known", data=known)
            saved.create_dataset("unit_noise_covariance", data=noise)
            for name, value in masks.items():
                saved.create_dataset(f"masks/{name}", data=value)
            for name, value in raw.items():
                saved.create_dataset(f"moments/{name}", data=value)
            saved.attrs["complete"] = True
    write_json(run/"calibration.json", dict(complete=True, identity=identity, **metadata))
    whitened = JointMoments(prior, known, masks, raw, metadata, backend)
    del raw
    val_path = run/"validation.h5"
    if val_path.exists():
        with h5py.File(val_path, "r") as saved:
            if not saved.attrs.get("complete") or saved.attrs["identity"] != identity:
                raise ValueError("Independent validation cache source mismatch")
            validation = {name:saved[name][:] for name in ("clean", "variances", "cross", "energy")}
    else:
        validation = validation_data(prior, known, masks, receiver,
            RecordSubset(archive, {r for r,_ in validation_ids}), validation_ids, frame_factory=frame_factory)
        with h5py.File(val_path, "w") as saved:
            saved.attrs.update(complete=False, identity=identity, windows_json=json.dumps(validation_ids))
            for name, value in validation.items():
                saved.create_dataset(name, data=value)
            saved.attrs["complete"] = True
    selection = dict(complete=False, identity=identity, prior_windows=prior_ids, validation_windows=validation_ids,
        objective="mean full-reference linear NMSE over independent validation frames, four D and all seven SNRs; exact thermal expectation",
        grid_alpha=[.4] if smoke else list(ALPHAS), grid_rho=[.5] if smoke else list(RHOS), candidates=[])
    best_score, best = np.inf, None
    for alpha in selection["grid_alpha"]:
        for rho in selection["grid_rho"]:
            candidate = JointFourDataEstimator(whitened, alpha, rho)
            objective, curve = validation_score(candidate, validation)
            entry = dict(alpha=alpha, rho=rho, objective=objective, mean_nmse_linear=curve.tolist(),
                         mean_nmse_db=(10*np.log10(curve)).tolist(), diagnostics=candidate.diagnostics)
            selection["candidates"].append(entry)
            if objective < best_score:
                best_score, best = objective, candidate
                selection["selected"] = entry
            write_json(run/"selection.json", selection)
            print(f"{archive.channel} validation alpha={alpha:g}, rho={rho:g}: objective={objective:.8g}, "
                  f"30dB NMSE={10*np.log10(curve[-1]):.3f} dB", flush=True)
    selection["complete"] = True
    write_json(run/"selection.json", selection)
    save_estimator(frozen, best, identity, selection)
    return best, selection


def run_channel(channel, archive_path, previous, baseline, output, backend, source, *, smoke):
    config = experiment_config(channel)
    source_path, before = channel_report(previous, channel)
    _, original = baseline_report(baseline, channel)
    prior_ids, test_ids = window_ids(config, smoke)
    validation_ids = [(record, start) for record in range(56, 61) for start in (2., 14.)]
    if smoke:
        validation_ids = validation_ids[:1]
    if (before["smoke"] != smoke or original["smoke"] != smoke
            or [(f["record"], f["start_seconds"]) for f in before["frames"]] != test_ids
            or [(f["record"], f["start_seconds"]) for f in original["frames"]] != test_ids):
        raise ValueError("All three experiments must use the same test windows and scope")
    if ({r for r,_ in prior_ids} & {r for r,_ in validation_ids}
            or {r for r,_ in prior_ids+validation_ids} & {r for r,_ in test_ids}):
        raise ValueError("Calibration/validation/test records must be disjoint")
    q, length = PARAMETERS[channel]
    if (before["parameters"]["effective_q"], before["parameters"]["length"]) != (q, length):
        raise ValueError("Frozen Qeff/L must remain unchanged")
    identity = digest(dict(channel=channel, source=source, smoke=smoke, prior=prior_ids,
                           validation=validation_ids, test=test_ids, alphas=ALPHAS, rhos=RHOS))
    run = output/channel.lower()
    run.mkdir(parents=True, exist_ok=True)
    path = run/"metrics.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError("Refusing to overwrite results with a different source identity")
        if result["complete"]:
            return result, before, original
    else:
        result = dict(schema="four-data-joint-physical-mmse-v1", complete=False, channel=channel,
            identity=identity, smoke=smoke, source=source, prior_windows=prior_ids,
            validation_windows=validation_ids, test_windows=test_ids, frames=[],
            parameters=dict(before["parameters"], csi_view="direct receiver CSI with independent-prior physical joint moments",
                covariance="pilot residual + coefficient/residual cross moments + full physical data covariance including cross-D and conjugate paths",
                hyperparameters="frozen by independent validation NMSE; no test CSI", detector="unchanged single/full LMMSE with actual RF thermal variance"),
            power_calibration=before["power_calibration"], source_receiver_identity=before["identity"])
    with h5py.File(previous/channel.lower()/"frozen_prior.h5", "r") as saved:
        model = DPSSModel(saved["basis"][:], 12., 8.22586010451598e-6, length, config)
        coefficients = saved["coefficients"][:]
        if json.loads(saved.attrs["independent_records_json"]) != [list(x) for x in prior_ids]:
            raise ValueError("Prior source windows differ from the original frozen experiment")
    prior = FrozenPrior.fit_coefficients(model, coefficients, 0, 1, config["prior_records"],
                                        regularizer=1e-6, delay_block_size=32)
    prior = replace(prior, fit_domain="four_fft_windows_with_full_frame_temporal_basis")
    archive, receiver = WatermarkArchive(archive_path, channel), BlockedFourDataReceiver(config)
    source_cache = previous/channel.lower()/"receiver_estimates.h5"
    with h5py.File(source_cache, "r") as saved:
        if not saved.attrs.get("complete") or saved.attrs["identity"] != before["identity"]:
            raise ValueError("Source receiver observations are incomplete or from another run")
        known, noise = saved["known"][:], saved["unit_noise_covariance"][:]
        masks = {name:saved[f"masks/{name}"][:] for name in ("data", "tracking", "pilot", "protection")}
    estimator, selection = calibrate(run, identity, prior, known, masks, noise, receiver, archive,
                                     prior_ids, validation_ids, backend, smoke=smoke, frame_factory=make_full_band_frame)
    result["selected_parameters"] = selection["selected"]
    result["estimator"] = estimator.diagnostics
    result["frozen_estimator_sha256"] = file_sha256(run/"frozen_estimator.h5")
    write_json(path, result)
    # Test observations and test reference CSI are accessed only after the
    # complete prior/validation estimator has been serialized and frozen.
    whitener = make_whitener(noise, backend)
    done = {(f["record"], f["start_seconds"]) for f in result["frames"]}
    with h5py.File(source_cache, "r") as received, h5py.File(run/"receiver_estimates.h5", "a") as saved:
        if "identity" in saved.attrs and saved.attrs["identity"] != identity:
            raise ValueError("Corrected receiver cache identity mismatch")
        saved.attrs.update(identity=identity, complete=False, source_receiver_identity=before["identity"],
                           estimator_sha256=result["frozen_estimator_sha256"])
        for index, (record, start) in enumerate(test_ids):
            if (record, start) in done:
                continue
            begin = time.perf_counter()
            group = received[f"frames/{index:04d}"]
            gain = float(group.attrs["tx_gain"])
            variances, bits = group["rf_variances"][:], group["bits"][:]
            y = group["clean_observations"][:][None]+np.sqrt(variances[:, None, None])*group["standard_noise"][:][None]
            fitted = estimator.estimate_coefficients(y, variances/gain**2)
            sounding = archive.load(record)
            references = [receiver.reference_taps(sounding, start, s) for s in range(4)]
            truth = gain*np.stack([t for t,_ in references])
            nmse, errors = score(model, fitted, truth, y, known, bits, variances, masks,
                noise, backend, whitener, verify=index == 0)
            key = f"frames/{index:04d}"
            if key in saved:
                del saved[key]
            target = saved.create_group(key)
            target.attrs.update(record=record, start_seconds=start, tx_gain=gain)
            for name, value in dict(coefficients=fitted, rf_variances=variances, nmse_linear=nmse,
                    **{f"errors_{name}":v for name,v in errors.items()}).items():
                target.create_dataset(name, data=value)
            saved.flush()
            old_item = before["frames"][index]
            item = dict(record=record, start_seconds=start, tx_gain=gain, payload_seed=old_item["payload_seed"],
                noise_seed=old_item["noise_seed"], old_rf_tx_energy=old_item["old_rf_tx_energy"],
                new_rf_tx_energy=old_item["new_rf_tx_energy"], rf_power_ratio=old_item["rf_power_ratio"],
                nmse_linear_per_d=nmse.tolist(), bit_errors={name:v.tolist() for name,v in errors.items()},
                intersymbol_leakage=[v for _,v in references], seconds=time.perf_counter()-begin)
            result["frames"].append(item)
            result["summary"] = _summary(result["frames"], config["snr_db"], bits.size)
            write_json(path, result)
            print(f"{channel} corrected {index+1}/{len(test_ids)} {record:03d}@{start}s: "
                  f"30dB NMSE={10*np.log10(nmse[-1].mean()):.3f} dB; "
                  f"BER single/full={errors['single_tap_mmse'][-1].sum()/bits.size:.5f}/"
                  f"{errors['full_matrix_mmse'][-1].sum()/bits.size:.5f}", flush=True)
        saved.attrs["complete"] = True
    if file_sha256(run/"frozen_estimator.h5") != result["frozen_estimator_sha256"]:
        raise AssertionError("Frozen estimator changed during test evaluation")
    result["complete"] = True
    write_json(path, result)
    return result, before, original


def comparison(output, results, source, *, original_layout=False):
    report = dict(schema="four-data-original-layout-three-way-v1" if original_layout else "four-data-three-way-covariance-v1",
                  complete=True, source=source, channels={})
    for channel, (new, before, original) in results.items():
        report["channels"][channel] = dict(corrected=new["summary"], before=before["summary"],
            q24_l176=original["summary"], parameters=new["parameters"], selected_parameters=new["selected_parameters"],
            rf_power_ratio_min=min(f["rf_power_ratio"] for f in new["frames"]),
            rf_power_ratio_max=max(f["rf_power_ratio"] for f in new["frames"]))
    write_json(output/"metrics.json", report)
    with (output/"metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("channel", "scheme", "snr_db", "frames", "bits", "single_tap_errors", "single_tap_ber",
                         "full_matrix_errors", "full_matrix_ber", "mean_nmse_db"))
        for channel, data in report["channels"].items():
            for scheme in ("q24_l176", "before", "corrected"):
                s = data[scheme]
                for index, snr in enumerate(s["snr_db"]):
                    single, full = (s["algorithms"][name] for name in ALGORITHMS)
                    writer.writerow((channel, scheme, snr, s["frame_count"], s["bits_per_snr"],
                        single["bit_errors"][index], single["ber"][index], full["bit_errors"][index], full["ber"][index], s["mean_nmse_db"][index]))
    def draw(axis, channel, metric):
        data = report["channels"][channel]
        for scheme, label, color, marker in (("q24_l176", "Original Q24/L176", "#8a919a", "v"),
                ("before", "Qeff/L, structured prior" if original_layout else "Before correction", "#cf7455", "o"),
                ("corrected", "Qeff/L, joint covariance" if original_layout else "Joint covariance", "#245b89", "s")):
            s = data[scheme]
            if metric == "ber":
                for algorithm, style, suffix in (("single_tap_mmse", "--", "single-tap"), ("full_matrix_mmse", "-", "full-matrix")):
                    values = np.asarray(s["algorithms"][algorithm]["ber"])
                    axis.semilogy(s["snr_db"], np.maximum(values, .5/s["bits_per_snr"]),
                        color=color, linestyle=style, marker=marker, markersize=4, label=f"{label}: {suffix}")
            else:
                axis.plot(s["snr_db"], s["mean_nmse_db"], color=color, marker=marker, markersize=4, label=label)
        p = data["parameters"]
        axis.set(title=f"{channel}: Qeff={p['effective_q']}, L={p['length']}", xlabel="SNR (dB)",
                 ylabel="Bit error rate" if metric == "ber" else "Mean channel NMSE (dB)")
        axis.set_xticks(data["corrected"]["snr_db"])
        axis.grid(True, which="both", alpha=.22)
        axis.legend(fontsize=7.5)
    channels = list(report["channels"])
    title = "12 Hz | 4D, no P | frozen independent-prior joint covariance"
    caption = "Same observations and matched RF power before/after correction; no test CSI in calibration\nNew layout: 512 pilots / 256 protection / 256 data per D; original Q24/L176 has twice the payload"
    if original_layout:
        title = "12 Hz | 4D, no P | original carrier layout retained"
        caption = "All three schemes: identical original waveform, data bits and RF noise; no test CSI in calibration\n128 pilots / 256 protection / 512 data per D; 64 empty bins at each edge; identical masks in all four D"
    fig, axes = plt.subplots(2, len(channels), figsize=(6.4*len(channels), 8.7), squeeze=False, layout="constrained")
    for col, channel in enumerate(channels):
        draw(axes[0, col], channel, "ber")
        draw(axes[1, col], channel, "nmse")
    fig.suptitle(title, fontsize=12)
    fig.supxlabel(caption, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"comparison.{ext}", dpi=180)
    plt.close(fig)
    for metric in ("ber", "nmse"):
        fig, axes = plt.subplots(1, len(channels), figsize=(6.4*len(channels), 4.8), squeeze=False, layout="constrained")
        for col, channel in enumerate(channels):
            draw(axes[0, col], channel, metric)
        fig.suptitle(title, fontsize=11)
        fig.supxlabel(caption, fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{metric}_snr.{ext}", dpi=180)
        plt.close(fig)
    return report


def run_experiment(archive, previous, baseline, output, *, channels, device, smoke=False, report_only=False):
    previous, baseline, output = Path(previous), Path(baseline), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if report_only:
        results = {channel:(channel_report(output, channel)[1], channel_report(previous, channel)[1],
                            baseline_report(baseline, channel)[1]) for channel in channels}
        return comparison(output, results, next(iter(results.values()))[0]["source"])
    source = dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={name:file_sha256(ROOT/name) for name in CODE}, archive_sha256=file_sha256(archive),
        previous={channel:{name:file_sha256(previous/channel.lower()/name) for name in
            ("metrics.json", "frozen_prior.h5", "receiver_estimates.h5")} for channel in channels},
        baseline={channel:file_sha256(baseline_report(baseline, channel)[0]) for channel in channels},
        device=device, smoke=smoke, precision="complex128", validation_records=list(range(56, 61)),
        test_cache_fields=("clean_observations", "standard_noise", "rf_variances", "bits", "tx_gain"))
    write_json(output/"source_manifest.json", source)
    backend = DenseBackend(device)
    results = {channel:run_channel(channel, archive, previous, baseline, output, backend, source, smoke=smoke) for channel in channels}
    return comparison(output, results, source)
