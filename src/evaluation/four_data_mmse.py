"""Small physical-link experiment for fixed 12 Hz / Q24 / L176 four-data CSI."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT, digest
from data_generation.src.pdd import load_pdd_config
from data_generation.src.receiver.chain import relative_energy, taps_to_matrix
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.four_data import (
    BlockedFourDataReceiver, FourDataEstimator, fixed_four_data_model, make_four_data_frame,
)
from data_generation.src.transmitter.ofdm import carrier_masks, decisions, fft, receive_symbols
from src.algorithms.full_matrix_lmmse.equalizer import equalize_batch as full_matrix_mmse
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.algorithms.single_tap_mmse.equalizer import equalize_batch as single_tap_mmse
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend


ALGORITHMS = {"single_tap_mmse": single_tap_mmse, "full_matrix_mmse": full_matrix_mmse}
CODE_FILES = (
    "data_generation/src/receiver/four_data.py", "src/evaluation/four_data_mmse.py",
    "scripts/evaluate/four_data_mmse.py", "data_generation/src/receiver/chain.py",
    "data_generation/src/receiver/dpss.py", "data_generation/src/receiver/estimation.py",
    "data_generation/src/receiver/pdd.py", "data_generation/src/transmitter/ofdm.py",
    "data_generation/src/channel/watermark/replay.py", "src/common/numerics.py",
    "src/algorithms/single_tap_mmse/equalizer.py", "src/algorithms/full_matrix_lmmse/equalizer.py",
    "src/algorithms/likelihood.py",
)


def experiment_config(channel):
    return dict(load_pdd_config(ROOT/f"configs/data/{channel.lower()}/pddp.json"),
                delay_length=176, dpss_expected_count=24, dpss_half_bandwidth_hz=12.0,
                csi_model_error_ratio=0.0)


def window_ids(config, smoke):
    prior = [(record, start) for record in config["prior_records"] for start in (2, 14)]
    test = [(record, start) for record in config["learning_dataset"]["records"]["test"]
            for start in (2, 14, 26)]
    if smoke:
        prior, test = prior[:2], test[:1]
    assert not {r for r, _ in prior} & {r for r, _ in test}
    return prior, test


def score_window(model, fitted, truth, observations, known, bits, variances, noise, backend,
                 whitener, *, verify_matrix_nmse=False):
    """Equalize identical CSI/observations and measure errors over all four D symbols."""
    data = np.flatnonzero(carrier_masks(model.c)["data"])
    nmse = np.empty((len(variances), 4))
    errors = {name: np.empty((len(variances), 4), dtype=np.int64) for name in ALGORITHMS}
    for symbol in range(4):
        estimate = model.reconstruct(fitted, symbol)
        reference = truth[symbol]
        energy = float(np.sum(abs(reference)**2))
        omitted = float(np.sum(abs(reference[:, model.length:])**2))
        nmse[:, symbol] = (np.sum(abs(estimate-reference[:, :model.length])**2, axis=(1, 2))+omitted)/energy
        matrices = np.stack([taps_to_matrix(taps) for taps in estimate])
        if verify_matrix_nmse:
            full_reference = taps_to_matrix(reference)
            for si in (0, len(variances)-1):
                direct = np.sum(abs(matrices[si]-full_reference)**2)/np.sum(abs(full_reference)**2)
                np.testing.assert_allclose(nmse[si, symbol], direct, rtol=2e-12, atol=1e-14)
        raw = observations[:, symbol]
        cancelled = raw-(matrices @ known[symbol, :, None])[..., 0]
        batch = EqualizerBatch(backend.array(matrices), backend.array(raw), backend.array(cancelled),
                               np.asarray(variances), data, noise)
        for name, equalizer in ALGORITHMS.items():
            kwargs = {"whitener": whitener} if name == "full_matrix_mmse" else {}
            estimate_symbols = equalizer(batch, backend, **kwargs).symbols
            errors[name][:, symbol] = np.count_nonzero(decisions(estimate_symbols) != bits[symbol], axis=(1, 2))
    return nmse, errors


def _summary(frames, snr, bits_per_frame):
    nmse = np.asarray([item["nmse_linear_per_d"] for item in frames]).mean(axis=(0, 2))
    total_bits = len(frames)*bits_per_frame
    algorithms = {}
    for name in ALGORITHMS:
        errors = np.asarray([item["bit_errors"][name] for item in frames], dtype=np.int64).sum(axis=(0, 2))
        algorithms[name] = dict(bit_errors=errors.tolist(), bit_count_per_snr=total_bits,
                                ber=(errors/total_bits).tolist())
    return dict(snr_db=snr, frame_count=len(frames), d_blocks_per_snr=len(frames)*4,
                bits_per_snr=total_bits, mean_nmse_linear=nmse.tolist(),
                mean_nmse_db=(10*np.log10(nmse)).tolist(), algorithms=algorithms)


def run_channel(channel, archive_path, output, backend, *, smoke, source):
    config = experiment_config(channel)
    prior_ids, test_ids = window_ids(config, smoke)
    model = fixed_four_data_model(config)
    masks = carrier_masks(config)
    run = output/channel.lower()
    run.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(config=config, prior=prior_ids, test=test_ids, source=source))
    metrics_path = run/"metrics.json"
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError(f"Refusing to mix another experiment into {run}")
        if result["complete"]:
            return result
    else:
        result = dict(schema="four-data-fixed-mmse-v1", identity=identity, channel=channel,
                      complete=False, smoke=smoke, prior_windows=prior_ids, test_windows=test_ids,
                      source=source, config=config, records={}, frames=[],
                      parameters=dict(frame="D1-D2-D3-D4", training_symbols=0, bandwidth_hz=12.0,
                                      q=24, delay_length=176, delay_block_size=32, frame_samples=5632,
                                      dpss_tail_ratio=model.tail_ratio, csi_view="raw receiver estimate",
                                      ideal_compensation_db=0, feedback_iterations=0,
                                      timing="known", noise="known simulated RF variance with colored receive covariance",
                                      estimation_rows="128 pilots + 256 protection carriers per D; all four D jointly",
                                      nmse="mean per-D linear full-matrix relative squared error, then 10 log10",
                                      reference_delay_length=1024, test_repeats=1,
                                      numerical_precision="complex128", snr_power="mean over all four received D FFT windows"))
    archive = WatermarkArchive(archive_path, channel)
    receiver = BlockedFourDataReceiver(config)
    frame = make_four_data_frame(config, *test_ids[0])
    prior_path = run/"frozen_prior.h5"
    if prior_path.exists():
        with h5py.File(prior_path, "r") as saved:
            if saved.attrs["identity"] != identity:
                raise ValueError("Frozen prior identity mismatch")
            np.testing.assert_array_equal(saved["basis"][:], model.basis)
            coefficients = saved["coefficients"][:]
    else:
        coefficients = []
        for index, (record, start) in enumerate(prior_ids):
            sounding = archive.load(record)
            result["records"][str(record)] = sounding.metadata()
            coefficients.append(receiver.prior_coefficients(sounding, start, model))
            print(f"{channel} prior {index+1}/{len(prior_ids)}: {record:03d} @ {start}s", flush=True)
        coefficients = np.stack(coefficients)
        partial = prior_path.with_suffix(".partial.h5")
        with h5py.File(partial, "w") as saved:
            saved.attrs.update(identity=identity, source_records_json=json.dumps(config["prior_records"]))
            saved.create_dataset("basis", data=model.basis)
            saved.create_dataset("coefficients", data=coefficients)
        partial.replace(prior_path)
    for record, _ in prior_ids:
        result["records"][str(record)] = archive.load(record).metadata()
    prior = FrozenPrior.fit_coefficients(model, coefficients, 0, 1, config["prior_records"],
                                         regularizer=config["relative_regularizer"], delay_block_size=32)
    begin = time.perf_counter()
    estimator = FourDataEstimator(prior, frame.known, masks, receiver.chain.noise_covariance, backend)
    result["estimator"] = dict(estimator.diagnostics, preparation_seconds=time.perf_counter()-begin)
    print(f"{channel} joint estimator ready: {estimator.diagnostics}, device={backend.device}", flush=True)
    noise = receiver.chain.noise_covariance
    whitener = make_whitener(noise, backend)
    completed = {(item["record"], item["start_seconds"]) for item in result["frames"]}
    snr = np.asarray(config["snr_db"], dtype=float)
    with h5py.File(run/"receiver_estimates.h5", "a") as saved:
        if "identity" in saved.attrs and saved.attrs["identity"] != identity:
            raise ValueError("Receiver estimates identity mismatch")
        saved.attrs.update(identity=identity, channel=channel, frame="DDDD", csi_view="raw", complete=False)
        if "known" not in saved:
            saved.create_dataset("known", data=frame.known)
            saved.create_dataset("snr_db", data=snr)
            saved.create_dataset("unit_noise_covariance", data=noise)
        for index, (record, start) in enumerate(test_ids):
            if (record, start) in completed:
                continue
            begin = time.perf_counter()
            sounding = archive.load(record)
            result["records"][str(record)] = sounding.metadata()
            frame = make_four_data_frame(config, record, start)
            origin = start-config["guard_samples"]/config["fs_baseband"]
            clean = receiver.replay_frame(sounding, origin, frame)
            clean_y = receive_symbols(receiver.chain.receive(clean), config)
            noise_seed = (config["seed"], 2, record, round(start*1000), 1, 0)
            rng = np.random.default_rng(np.random.SeedSequence(noise_seed))
            standard = receive_symbols(receiver.chain.receive(rng.standard_normal(len(clean))), config)
            power = float(np.mean(abs(clean_y)**2))
            variances = power/receiver.chain.noise_gain/10**(snr/10)
            observations = clean_y[None]+np.sqrt(variances[:, None, None])*standard[None]
            references = [receiver.reference_taps(sounding, start, symbol) for symbol in range(4)]
            truth = np.stack([taps for taps, _ in references])
            leakage = [value for _, value in references]
            predicted = fft(np.sum(truth*input_windows(frame.x, config["nfft"]), axis=-1))
            model_error = [relative_energy(y-p, y) for y, p in zip(clean_y, predicted, strict=True)]
            support = (np.sum(abs(truth[..., model.length:])**2, axis=(1, 2))/np.sum(abs(truth)**2, axis=(1, 2)))
            fitted = estimator.estimate_coefficients(observations, variances)
            nmse, errors = score_window(model, fitted, truth, observations, frame.known, frame.bits,
                                        variances, noise, backend, whitener, verify_matrix_nmse=index == 0)
            key = f"frames/{index:04d}"
            if key in saved:
                del saved[key]
            group = saved.create_group(key)
            group.attrs.update(record=record, start_seconds=start, payload_seed=json.dumps(frame.seed),
                               noise_seed=json.dumps(noise_seed))
            for name, value in dict(coefficients=fitted, clean_observations=clean_y, standard_noise=standard,
                                    rf_variances=variances, bits=frame.bits, nmse_linear=nmse,
                                    **{f"errors_{name}": value for name, value in errors.items()}).items():
                group.create_dataset(name, data=value)
            saved.flush()
            item = dict(record=record, start_seconds=start, payload_seed=frame.seed, noise_seed=noise_seed,
                        nmse_linear_per_d=nmse.tolist(), bit_errors={name: v.tolist() for name, v in errors.items()},
                        intersymbol_leakage=leakage, clean_circular_model_error=model_error,
                        reference_energy_outside_support=support.tolist(), rf_noise_variances=variances.tolist(),
                        seconds=time.perf_counter()-begin)
            result["frames"].append(item)
            result["summary"] = _summary(result["frames"], config["snr_db"], frame.bits.size)
            write_json(metrics_path, result)
            print(f"{channel} test {index+1}/{len(test_ids)}: {record:03d} @ {start}s; "
                  f"30dB NMSE={10*np.log10(nmse[-1].mean()):.3f} dB; "
                  f"BER single/full={errors['single_tap_mmse'][-1].sum()/frame.bits.size:.5f}/"
                  f"{errors['full_matrix_mmse'][-1].sum()/frame.bits.size:.5f}; {item['seconds']:.1f}s", flush=True)
        saved.attrs["complete"] = True
    result["complete"] = True
    result["summary"] = _summary(result["frames"], config["snr_db"], frame.bits.size)
    write_json(metrics_path, result)
    return result


def write_report(output, channels, source):
    report = dict(schema="four-data-fixed-mmse-comparison-v1", complete=True, source=source,
                  parameters=next(iter(channels.values()))["parameters"],
                  channels={channel: value["summary"] for channel, value in channels.items()})
    write_json(output/"metrics.json", report)
    with (output/"metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "snr_db", "frames", "bits", "single_tap_bit_errors", "single_tap_ber",
                         "full_matrix_bit_errors", "full_matrix_ber", "mean_nmse_linear", "mean_nmse_db"])
        for channel, result in report["channels"].items():
            single, full = (result["algorithms"][name] for name in ALGORITHMS)
            for si, snr in enumerate(result["snr_db"]):
                writer.writerow([channel, snr, result["frame_count"], result["bits_per_snr"],
                                 single["bit_errors"][si], single["ber"][si], full["bit_errors"][si], full["ber"][si],
                                 result["mean_nmse_linear"][si], result["mean_nmse_db"][si]])
    colors = {"NOF1": "#245a87", "NCS1": "#cd6549"}
    labels = {"single_tap_mmse": "Single-tap MMSE", "full_matrix_mmse": "Full-matrix MMSE"}

    def draw(axis, metric):
        for channel, result in report["channels"].items():
            if metric == "ber":
                for name, linestyle, marker in (("single_tap_mmse", "--", "o"), ("full_matrix_mmse", "-", "s")):
                    axis.semilogy(result["snr_db"], result["algorithms"][name]["ber"],
                                  color=colors[channel], linestyle=linestyle, marker=marker, markersize=4,
                                  label=f"{channel} / {labels[name]}")
            else:
                axis.plot(result["snr_db"], result["mean_nmse_db"], color=colors[channel], marker="o",
                          markersize=4, label=f"{channel} / shared receiver CSI")
        axis.set(xlabel="SNR (dB)", ylabel="Bit error rate" if metric == "ber" else "Mean channel NMSE (dB)",
                 title="BER-SNR" if metric == "ber" else "NMSE-SNR")
        axis.set_xticks(next(iter(report["channels"].values()))["snr_db"])
        axis.grid(True, which="both", alpha=0.22)
        axis.legend(fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25), layout="constrained")
    draw(axes[0], "ber")
    draw(axes[1], "nmse")
    frames = {value["frame_count"] for value in report["channels"].values()}
    count = str(next(iter(frames))) if len(frames) == 1 else "/".join(map(str, sorted(frames)))
    fig.suptitle("4D full-frame estimation | no P | 12 Hz | Q=24, L=176", fontsize=12)
    fig.supxlabel(f"Direct receiver CSI; {count} frames per channel per SNR; 4 data symbols per frame", fontsize=9)
    fig.savefig(output/"comparison.png", dpi=190)
    fig.savefig(output/"comparison.pdf")
    plt.close(fig)
    for metric in ("ber", "nmse"):
        single, axis = plt.subplots(figsize=(6, 4.6), layout="constrained")
        draw(axis, metric)
        single.suptitle("4D full-frame | no P | 12 Hz | Q=24, L=176", fontsize=10)
        single.supxlabel(f"Direct receiver CSI | {count} frames per channel per SNR\n4 data symbols per frame", fontsize=8)
        single.savefig(output/f"{metric}_snr.png", dpi=190)
        single.savefig(output/f"{metric}_snr.pdf")
        plt.close(single)
    write_json(output/"plot_manifest.json", dict(
        source_git_head=source["git_head"], plotting_code_sha256=file_sha256(Path(__file__)),
        created_utc=datetime.now(timezone.utc).isoformat()))
    return report


def render_saved_report(output):
    """Re-export finished metrics without repeating estimation or equalization."""
    output = Path(output).resolve()
    results = {}
    for channel in ("NOF1", "NCS1"):
        path = output/channel.lower()/"metrics.json"
        if not path.exists():
            path = output/f"{channel.lower()}_metrics.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if not result["complete"]:
                raise ValueError(f"Cannot export unfinished metrics: {path}")
            results[channel] = result
    if not results:
        raise FileNotFoundError("No completed four-data channel metrics")
    sources = [result["source"] for result in results.values()]
    if any(source != sources[0] for source in sources):
        raise ValueError("Channel results have different sources")
    return write_report(output, results, sources[0])


def run_experiment(archive, output, *, channels=("NOF1", "NCS1"), device="cpu", smoke=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    backend = DenseBackend(device)
    code = {name: file_sha256(ROOT/name) for name in CODE_FILES}
    metadata = dict(schema="four-data-fixed-mmse-source-v1", git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        archive_sha256=file_sha256(archive), code_sha256=code, smoke=smoke,
        environment=dict(python=platform.python_version(), numpy=np.__version__, device=device))
    # The timestamp is outside the resume identity.
    write_json(output/"source_manifest.json", dict(metadata, created_utc=datetime.now(timezone.utc).isoformat()))
    results = {}
    for channel in channels:
        results[channel] = run_channel(channel, archive, output, backend, smoke=smoke, source=metadata)
    return write_report(output, results, metadata)
