"""Receiver-estimated, power-matched full-band 4D experiment versus Q24/L176."""
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
from data_generation.src.receiver.chain import relative_energy, taps_to_matrix
from data_generation.src.receiver.estimation import input_windows
from data_generation.src.receiver.four_data import BlockedFourDataReceiver, FourDataEstimator, make_four_data_frame
from data_generation.src.receiver.four_data_rotated import (
    TransmitPower, full_band_layout, load_rotated_prior, make_full_band_frame,
)
from data_generation.src.transmitter.ofdm import decisions, fft, receive_symbols
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.evaluation.four_data_mmse import ALGORITHMS, _summary, experiment_config, window_ids


PARAMETERS = {"NOF1": (5, 512), "NCS1": (12, 224)}
CODE = ("data_generation/src/receiver/four_data.py", "data_generation/src/receiver/four_data_rotated.py",
        "src/evaluation/four_data_rotated.py", "scripts/evaluate/four_data_rotated.py",
        "data_generation/src/receiver/chain.py", "data_generation/src/receiver/estimation.py",
        "src/algorithms/single_tap_mmse/equalizer.py", "src/algorithms/full_matrix_lmmse/equalizer.py",
        "src/algorithms/likelihood.py", "src/common/numerics.py")


def baseline_report(root, channel):
    path = root/channel.lower()/"metrics.json"
    if not path.exists():
        path = root/f"{channel.lower()}_metrics.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    p = result["parameters"]
    if not result["complete"] or (p["bandwidth_hz"], p["q"], p["delay_length"], p["ideal_compensation_db"]) != (12, 24, 176, 0):
        raise ValueError("Comparison needs completed direct-CSI Q24/L176 results")
    return path, result


def score(model, fitted, reference, y, known, bits, variances, masks, noise, backend, whitener, *, verify=False):
    nmse = np.zeros((len(variances), 4))
    errors = {name: np.zeros((len(variances), 4), dtype=np.int64) for name in ALGORITHMS}
    for symbol in range(4):
        taps = model.reconstruct(fitted, symbol)
        truth = reference[symbol]
        energy = float(np.sum(abs(truth)**2))
        omitted = float(np.sum(abs(truth[:, model.length:])**2))
        nmse[:, symbol] = (np.sum(abs(taps-truth[:, :model.length])**2, axis=(1, 2))+omitted)/energy
        matrices = np.stack([taps_to_matrix(tap) for tap in taps])
        if verify:
            full = taps_to_matrix(truth)
            for si in (0, len(variances)-1):
                direct = np.sum(abs(matrices[si]-full)**2)/np.sum(abs(full)**2)
                np.testing.assert_allclose(nmse[si, symbol], direct, rtol=2e-12, atol=1e-14)
        raw = y[:, symbol]
        cancelled = raw-(matrices @ known[symbol, :, None])[..., 0]
        indices = np.flatnonzero(masks["data"][symbol])
        batch = EqualizerBatch(backend.array(matrices), backend.array(raw), backend.array(cancelled),
                              variances, indices, noise)
        for name, equalizer in ALGORITHMS.items():
            parameters = {"whitener": whitener} if name == "full_matrix_mmse" else {}
            result = equalizer(batch, backend, **parameters).symbols
            errors[name][:, symbol] = np.count_nonzero(decisions(result) != bits[symbol], axis=(1, 2))
    return nmse, errors


def run_channel(channel, archive_path, design, baseline, output, backend, source, *, smoke=False):
    config = experiment_config(channel)
    count, length = PARAMETERS[channel]
    _, old = baseline_report(baseline, channel)
    expected_prior, expected_test = window_ids(config, smoke)
    if [(f["record"], f["start_seconds"]) for f in old["frames"]] != expected_test:
        raise ValueError("Old and new frame identities must match")
    if (old["smoke"] != smoke or old["config"]["snr_db"] != config["snr_db"]
            or old["config"]["seed"] != config["seed"]):
        raise ValueError("Old and new experiment scope/SNR mismatch")
    run = output/channel.lower()
    run.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(source=source, channel=channel, config=config, q=count, length=length,
                           smoke=smoke, test=expected_test))
    path = run/"metrics.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError("Refusing to mix another run into this output directory")
        if result["complete"]:
            return result, old
    else:
        result = dict(schema="four-data-rotated-mmse-v1", channel=channel, identity=identity,
                      complete=False, smoke=smoke, source=source, frames=[], source_physical_config=config,
                      parameters=dict(frame="D1-D2-D3-D4", bandwidth_hz=12, dictionary_q=27,
                                      effective_q=count, length=length, pilots_per_d=512,
                                      protection_per_d=256, data_per_d=256, observations=3072,
                                      unknown_coefficients=count*length, delay_block_size=32,
                                      csi_view="direct receiver estimate", ideal_compensation_db=0,
                                      layout="DDZPPPPZ over all 1024 bins; D shifts 0/1/3/5",
                                      prior_projection="independent prior FFT windows; full-frame temporal basis",
                                      snr_definition="mean clean power of all four received FFT windows / thermal noise power",
                                      csi_definition="effective channel including the actual transmitter normalization gain",
                                      power="same RF transmit energy as the paired old frame, including CP and transmit filters"))
    model, prior, prior_ids = load_rotated_prior(design, channel, config, count, length, smoke=smoke)
    if prior_ids != [(int(r), float(s)) for r, s in expected_prior]:
        raise ValueError("Temporal training identities differ from the fixed independent prior subset")
    result["prior_windows"] = prior_ids
    receiver = BlockedFourDataReceiver(config)
    unit_known, masks = full_band_layout(config)
    power = TransmitPower(receiver.chain).match(config, unit_known, masks["data"])
    known = unit_known*np.sqrt(power["pilot_symbol_energy"])
    result["power_calibration"] = power
    frozen_path = run/"frozen_prior.h5"
    if not frozen_path.exists():
        with h5py.File(frozen_path, "w") as saved:
            saved.attrs.update(identity=identity, fit_domain=prior.fit_domain, independent_records_json=json.dumps(prior_ids))
            saved.create_dataset("basis", data=model.basis)
            saved.create_dataset("coefficients", data=prior.coefficients)
    print(f"{channel} preparing Qeff={count}, L={length}, 3072-observation CSI estimator; pilot energy={power['pilot_symbol_energy']:.8f}", flush=True)
    begin = time.perf_counter()
    estimator = FourDataEstimator(prior, known, masks, receiver.chain.noise_covariance, backend)
    result["estimator"] = dict(estimator.diagnostics, preparation_seconds=time.perf_counter()-begin)
    noise = receiver.chain.noise_covariance
    whitener = make_whitener(noise, backend)
    archive = WatermarkArchive(archive_path, channel)
    complete = {(f["record"], f["start_seconds"]) for f in result["frames"]}
    snr = np.asarray(config["snr_db"], float)
    with h5py.File(run/"receiver_estimates.h5", "a") as saved:
        if "identity" in saved.attrs and saved.attrs["identity"] != identity:
            raise ValueError("Receiver cache identity mismatch")
        saved.attrs.update(identity=identity, complete=False, channel=channel, csi_view="raw")
        if "known" not in saved:
            saved.create_dataset("known", data=known)
            saved.create_dataset("snr_db", data=snr)
            saved.create_dataset("unit_noise_covariance", data=noise)
            for name, mask in masks.items():
                saved.create_dataset(f"masks/{name}", data=mask)
        for index, (record, start) in enumerate(expected_test):
            if (record, start) in complete:
                continue
            begin = time.perf_counter()
            frame = make_full_band_frame(config, record, start, known, masks)
            old_frame = make_four_data_frame(config, record, start)
            old_tx = receiver.chain.transmit(old_frame.baseband)
            new_tx = receiver.chain.transmit(frame.baseband)
            old_energy, before = float(np.sum(old_tx**2)), float(np.sum(new_tx**2))
            gain = float(np.sqrt(old_energy/before))
            frame = replace(frame, baseband=frame.baseband*gain)
            matched_energy = float(np.sum(receiver.chain.transmit(frame.baseband)**2))
            np.testing.assert_allclose(matched_energy, old_energy, rtol=2e-13, atol=0)
            sounding = archive.load(record)
            origin = start-config["guard_samples"]/config["fs_baseband"]
            clean = receiver.replay_frame(sounding, origin, frame)
            clean_y = receive_symbols(receiver.chain.receive(clean), config)
            noise_seed = (config["seed"], 2, record, round(start*1000), 1, 0)
            assert tuple(old["frames"][index]["payload_seed"]) == frame.seed
            assert tuple(old["frames"][index]["noise_seed"]) == noise_seed
            rng = np.random.default_rng(np.random.SeedSequence(noise_seed))
            standard = receive_symbols(receiver.chain.receive(rng.standard_normal(len(clean))), config)
            variances = float(np.mean(abs(clean_y)**2))/receiver.chain.noise_gain/10**(snr/10)
            y = clean_y[None]+np.sqrt(variances[:, None, None])*standard[None]
            refs = [receiver.reference_taps(sounding, start, d) for d in range(4)]
            truth = gain*np.stack([tap for tap, _ in refs])
            leakage = [value for _, value in refs]
            predicted = fft(np.sum(truth*input_windows(frame.x, config["nfft"]), axis=-1))
            model_error = [relative_energy(a-b, a) for a, b in zip(clean_y, predicted, strict=True)]
            # c_eff=g*c has prior covariance g^2 P. Dividing only the RF
            # variance by g^2 exactly reuses the precomputed LMMSE spectrum.
            fitted = estimator.estimate_coefficients(y, variances/gain**2)
            nmse, errors = score(model, fitted, truth, y, known, frame.bits, variances, masks,
                                 noise, backend, whitener, verify=index == 0)
            key = f"frames/{index:04d}"
            if key in saved:
                del saved[key]
            group = saved.create_group(key)
            group.attrs.update(record=record, start_seconds=start, tx_gain=gain,
                               payload_seed=json.dumps(frame.seed), noise_seed=json.dumps(noise_seed))
            arrays = dict(coefficients=fitted, clean_observations=clean_y, standard_noise=standard,
                          rf_variances=variances, bits=frame.bits, nmse_linear=nmse,
                          **{f"errors_{name}": value for name, value in errors.items()})
            for name, value in arrays.items():
                group.create_dataset(name, data=value)
            saved.flush()
            item = dict(record=record, start_seconds=start, payload_seed=frame.seed, noise_seed=noise_seed,
                        nmse_linear_per_d=nmse.tolist(), bit_errors={name:v.tolist() for name,v in errors.items()},
                        tx_gain=gain, old_rf_tx_energy=old_energy, new_rf_tx_energy=matched_energy,
                        rf_power_ratio=matched_energy/old_energy, intersymbol_leakage=leakage,
                        clean_circular_model_error=model_error, seconds=time.perf_counter()-begin)
            result["frames"].append(item)
            result["summary"] = _summary(result["frames"], config["snr_db"], frame.bits.size)
            write_json(path, result)
            print(f"{channel} {index+1}/{len(expected_test)} {record:03d}@{start}s: "
                  f"30dB NMSE={10*np.log10(nmse[-1].mean()):.3f} dB; "
                  f"BER single/full={errors['single_tap_mmse'][-1].sum()/frame.bits.size:.5f}/"
                  f"{errors['full_matrix_mmse'][-1].sum()/frame.bits.size:.5f}; RF ratio={matched_energy/old_energy:.12f}; "
                  f"{item['seconds']:.1f}s", flush=True)
        saved.attrs["complete"] = True
    result["complete"] = True
    write_json(path, result)
    return result, old


def comparison(output, results, source):
    report = dict(schema="four-data-old-new-mmse-v1", complete=True, source=source, channels={})
    for channel, (new, old) in results.items():
        report["channels"][channel] = dict(new=new["summary"], old=old["summary"],
            parameters=new["parameters"], power_calibration=new["power_calibration"],
            rf_power_ratio_min=min(f["rf_power_ratio"] for f in new["frames"]),
            rf_power_ratio_max=max(f["rf_power_ratio"] for f in new["frames"]))
    write_json(output/"metrics.json", report)
    with (output/"metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "scheme", "snr_db", "frames", "bits", "single_tap_errors", "single_tap_ber",
                         "full_matrix_errors", "full_matrix_ber", "mean_nmse_db"])
        for channel, row in report["channels"].items():
            for scheme in ("old", "new"):
                values = row[scheme]
                for si, snr in enumerate(values["snr_db"]):
                    single, full = (values["algorithms"][name] for name in ALGORITHMS)
                    writer.writerow([channel, scheme, snr, values["frame_count"], values["bits_per_snr"],
                        single["bit_errors"][si], single["ber"][si], full["bit_errors"][si], full["ber"][si],
                        values["mean_nmse_db"][si]])
    colors = {"NOF1":"#245a87", "NCS1":"#ce674b"}
    def draw(axis, channel, metric):
        row = report["channels"][channel]
        for scheme in ("old", "new"):
            values = row[scheme]
            color = "#7a8490" if scheme == "old" else colors[channel]
            label = "Old Q24/L176" if scheme == "old" else "New rotated basis"
            if metric == "ber":
                for name, style, marker, suffix in (("single_tap_mmse","--","o","single-tap"), ("full_matrix_mmse","-","s","full-matrix")):
                    ber = np.asarray(values["algorithms"][name]["ber"])
                    axis.semilogy(values["snr_db"], np.where(ber > 0, ber, .5/values["bits_per_snr"]),
                        linestyle=style, marker=marker, color=color, markersize=4, label=f"{label}: {suffix}")
            else:
                axis.plot(values["snr_db"], values["mean_nmse_db"], color=color,
                          marker="o" if scheme == "old" else "s", markersize=4, label=label)
        p = row["parameters"]
        axis.set(xlabel="SNR (dB)", ylabel="Bit error rate" if metric == "ber" else "Mean channel NMSE (dB)",
                 title=f"{channel} / {'BER-SNR' if metric == 'ber' else 'NMSE-SNR'} (Qeff={p['effective_q']}, L={p['length']})")
        axis.set_xticks(row["new"]["snr_db"])
        axis.grid(True, which="both", alpha=.22)
        axis.legend(fontsize=8)
    channels = list(report["channels"])
    fig, axes = plt.subplots(2, len(channels), figsize=(6*len(channels), 8), squeeze=False, layout="constrained")
    for ci, channel in enumerate(channels):
        draw(axes[0, ci], channel, "ber")
        draw(axes[1, ci], channel, "nmse")
    fig.suptitle("12 Hz | 4D, no P | direct receiver CSI\nMatched RF transmit power per paired frame", fontsize=12)
    fig.supxlabel("Same test windows; new scheme: 256 data / 512 pilots / 256 protection per D\nNew payload per frame is half the old payload", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"comparison.{suffix}", dpi=180)
    plt.close(fig)
    for metric in ("ber", "nmse"):
        fig, axes = plt.subplots(1, len(channels), figsize=(6*len(channels), 4.7), squeeze=False, layout="constrained")
        for ci, channel in enumerate(channels):
            draw(axes[0, ci], channel, metric)
        fig.suptitle("12 Hz, no P: Q24/L176 versus\nfull-band rotated BEM", fontsize=11)
        fig.supxlabel("Direct receiver CSI | equal RF power per paired frame\nNew payload per frame is halved", fontsize=9)
        for suffix in ("png", "pdf"):
            fig.savefig(output/f"{metric}_snr.{suffix}", dpi=180)
        plt.close(fig)
    return report


def run_experiment(archive, design, baseline, output, *, channels, device, smoke=False, report_only=False):
    output, design, baseline = Path(output), Path(design), Path(baseline)
    output.mkdir(parents=True, exist_ok=True)
    if report_only:
        pairs = {channel:(json.loads((output/channel.lower()/"metrics.json").read_text(encoding="utf-8")),
                          baseline_report(baseline, channel)[1]) for channel in channels}
        if not all(new["complete"] for new, _ in pairs.values()):
            raise ValueError("Cannot plot unfinished results")
        return comparison(output, pairs, next(iter(pairs.values()))[0]["source"])
    source = dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={name:file_sha256(ROOT/name) for name in CODE}, archive_sha256=file_sha256(archive),
        design_files={channel:{name:file_sha256(design/f"{channel.lower()}_{name}")
                              for name in ("temporal_basis.npz", "projection_cache.h5")} for channel in channels},
        baseline_metrics={channel:file_sha256(baseline_report(baseline, channel)[0]) for channel in channels},
        numerical_precision="complex128", device=device, smoke=smoke)
    write_json(output/"source_manifest.json", source)
    backend = DenseBackend(device)
    pairs = {channel:run_channel(channel, archive, design, baseline, output, backend, source, smoke=smoke)
             for channel in channels}
    return comparison(output, pairs, source)
