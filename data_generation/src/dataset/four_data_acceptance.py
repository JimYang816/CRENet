"""Independent disk acceptance and bounded BER diagnostics for formal DDDD data."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import h5py
import numpy as np

from data_generation.src.config import digest
from data_generation.src.dataset.four_data_math import array_sha256, coefficient_nmse, reference_statistics, sampled_grams
from data_generation.src.dataset.four_data_protocol import SCHEMA, SPLITS, SPLIT_CODES, validate_plan, windows
from data_generation.src.transmitter.ofdm import decisions, fft, make_frame
from data_generation.src.receiver.estimation import input_windows
from src.common.experiment import file_sha256, write_json
from src.data.four_data import FourDataDataset, data_system
from src.evaluation.four_data_csi_compensation import tap_nmse
from src.evaluation.four_data_lmmse import detect

LOG = logging.getLogger(__name__)


def _close(actual, expected, message, *, rtol=2e-9, atol=1e-11):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(message)


def ber_diagnostics(ds, backend, frame_limit):
    """Detectors see CSI/observations only; identities list the diagnostic subset."""
    h, n = ds._h, ds.config["nfft"]
    selected = np.unique(np.rint(np.linspace(0, ds.frame_count-1, min(frame_limit, ds.frame_count))).astype(int))
    errors = {view: {algorithm: np.zeros(len(ds.snr_db), np.int64) for algorithm in ("single", "full")}
              for view in ("raw", "reference_assisted", "reference")}
    noise = ds._model["unit_rf_noise_covariance"][:]
    alpha = h["compensation/error_scales"][:]
    from data_generation.src.dataset.four_data_math import assisted_taps
    for j, fi in enumerate(selected):
        observation = np.stack([ds.observations(si, int(fi)) for si in range(len(ds.snr_db))])
        variance = h["samples/rf_noise_variance"][:, fi]
        raw = np.stack([ds.raw_taps(si, int(fi)) for si in range(len(ds.snr_db))])
        truth = ds.reference_taps(int(fi))
        for view in errors:
            taps = raw if view == "raw" else truth if view == "reference" else assisted_taps(raw, truth, alpha)
            single, full = detect(taps, observation, ds.known, variance, noise, ds.masks["data"], backend)
            bits = h["samples/bits"][fi]
            for algorithm, symbols in (("single", single), ("full", full)):
                errors[view][algorithm] += np.count_nonzero(decisions(symbols) != bits[None], axis=(1, 2, 3))
            if view == "reference_assisted":
                del taps
        if backend.torch:
            backend.torch.cuda.empty_cache()
        if j == 0 or j+1 == len(selected):
            LOG.info("%s %s BER diagnostic %d/%d", ds.channel, ds.split, j+1, len(selected))
    bit_count = len(selected)*4*ds.config["data_count"]*2
    return dict(frame_indices=selected.tolist(), frame_identities=[ds.identity(int(fi)) for fi in selected],
        bits_per_snr=bit_count, diagnostic_only=(len(selected) != ds.frame_count),
        curves={view: {algorithm: dict(errors=value.tolist(), ber=(value/bit_count).tolist())
                       for algorithm, value in algorithms.items()} for view, algorithms in errors.items()})


def accept(directory, run, c, options, backend, *, hash_files=True):
    directory, run = Path(directory), Path(run)
    plan = validate_plan(c)
    report = dict(schema=SCHEMA, channel=c["channel"], complete=False, smoke=bool(c.get("smoke")),
        cp=c["cp"], cp_ms=1000*c["cp"]/c["fs_baseband"],
        default_input_view=options["default_input_view"], reference_assisted_csi_adjustment=(options["compensation_calibration"] != "none"),
        compensation_calibration=options["compensation_calibration"], config_hash=digest(c), counts=plan,
        snr_db=c["snr_db"], splits={})
    for split in SPLITS:
        LOG.info("%s %s independent full disk acceptance", c["channel"], split)
        with FourDataDataset(directory/f"{split}.h5") as ds:
            h, m = ds._h, ds._model
            if h.attrs["config_hash"] != digest(c) or ds.input_view != options["default_input_view"]:
                raise ValueError("Dataset settings differ from the requested formal configuration")
            expected_windows, repeats = windows(c, split), c["learning_dataset"]["repeats"][split]
            if list(zip(h["windows/record"][:], h["windows/start_seconds"][:])) != expected_windows:
                raise ValueError("Physical record/time identities differ from the configured split")
            if (not np.array_equal(h["samples/window_index"][:], np.repeat(np.arange(len(expected_windows)), repeats))
                    or not np.array_equal(h["samples/repeat"][:], np.tile(np.arange(repeats), len(expected_windows)))):
                raise ValueError("Payload/noise repeat identities differ from the formal plan")
            sampled, delays = ds.sampled_basis, ds.delays
            grams = sampled_grams(sampled)
            _close(grams, m["sampled_grams"][:], "Shared sampled-basis Gram matrix mismatch")
            raw_nmse = np.zeros((len(ds.snr_db), ds.frame_count, 4))
            noise_gain = np.trace(m["unit_rf_noise_covariance"][:]).real/c["nfft"]
            for wi, (record, start) in enumerate(expected_windows):
                reference_stored = h["windows/reference_taps"][wi]
                if array_sha256(reference_stored) != h["windows/reference_sha256"][wi].decode():
                    raise ValueError("Persisted full-support reference fingerprint mismatch")
                truth = reference_stored.astype(np.complex128)
                cross, energy = reference_statistics(sampled, truth, delays)
                _close(cross, h["windows/reference_cross"][wi], "Full-reference cross statistic mismatch")
                _close(energy, h["windows/reference_energy"][wi], "Full-reference energy mismatch")
                payload_hashes, noise_hashes = set(), set()
                for repeat in range(repeats):
                    fi = wi*repeats+repeat
                    clean, standard, bits = (h["samples/"+name][fi] for name in ("clean_observations", "standard_noise", "bits"))
                    variance = h["samples/rf_noise_variance"][:, fi]
                    if array_sha256(clean, standard, bits, variance) != h["samples/physical_sha256"][fi].decode():
                        raise ValueError("Physical observation, label or RF-noise fingerprint mismatch")
                    frame = make_frame(c, record, start, SPLIT_CODES[split], payload_repeat=repeat)
                    if (not np.array_equal(frame.bits, bits) or not np.array_equal(frame.known, ds.known)
                            or not np.array_equal(frame.seed, h["samples/payload_seed"][fi])
                            or not np.array_equal((c["seed"], SPLIT_CODES[split], record, round(start*1000), 1, repeat),
                                                  h["samples/noise_seed"][fi])):
                        raise ValueError("Public pilots, payload or independent noise seed mismatch")
                    payload_hashes.add(array_sha256(bits))
                    noise_hashes.add(array_sha256(standard))
                    _close(variance, np.mean(abs(clean)**2)/noise_gain/10**(np.asarray(c["snr_db"])/10), "Configured SNR variance mismatch")
                    prediction = fft(np.sum(truth*input_windows(frame.x, c["nfft"]), axis=-1))
                    _close(clean-prediction, h["samples/model_residual"][fi], "Physical model residual mismatch", atol=2e-10)
                    coefficients = h["samples/raw_coefficients"][:, fi]
                    raw_nmse[:, fi] = coefficient_nmse(coefficients, grams, cross, energy)
                    _close(raw_nmse[:, fi], h["samples/raw_nmse_linear"][:, fi], "Raw full-matrix NMSE mismatch")
                    realized = variance[:, None]*np.mean(abs(standard)**2, axis=1)[None]
                    _close(10*np.log10(np.mean(abs(clean)**2, axis=1)[None]/realized),
                        h["samples/actual_snr_db"][:, fi], "Realized SNR mismatch")
                if len(payload_hashes) != repeats or len(noise_hashes) != repeats:
                    raise ValueError("Repeated test payloads/noise are not independent")
                if wi == 0 or (wi+1) % 100 == 0 or wi+1 == len(expected_windows):
                    LOG.info("%s %s accepted physical window %d/%d", c["channel"], split, wi+1, len(expected_windows))
            means = raw_nmse.mean(axis=(1, 2))
            alpha = h["compensation/error_scales"][:]
            assisted_mean = means*alpha**2
            target = h["compensation/target_nmse_db"][:]
            _close(means, h["compensation/raw_mean_nmse_linear"][:], "Raw calibration mean mismatch")
            _close(assisted_mean, h["compensation/assisted_mean_nmse_linear"][:], "Assisted calibration mean mismatch")
            if options["compensation_calibration"] == "per_split" or (options["compensation_calibration"] == "train_only" and split == "train"):
                _close(10*np.log10(assisted_mean), target, "Requested target NMSE was not reached", rtol=0, atol=1e-8)
            # Actual full matrices verify the Gram identity and both reader views.
            for fi in sorted(set((0, ds.frame_count-1))):
                reference = ds.reference_taps(fi)
                for si in range(len(ds.snr_db)):
                    raw = ds.raw_taps(si, fi)
                    padded = np.pad(raw, ((0, 0), (0, 0), (0, c["nfft"]-raw.shape[-1])))
                    _close(tap_nmse(padded[None], reference)[0], raw_nmse[si, fi], "Direct raw CSI NMSE mismatch", rtol=2e-8)
                    current = ds.csi_taps(si, fi)
                    if ds.input_view == "reference_assisted":
                        _close(tap_nmse(current[None], reference)[0], raw_nmse[si, fi]*alpha[si]**2,
                            "Direct full-support assisted CSI NMSE mismatch", rtol=2e-8)
                    inputs = ds.model_input(si, fi)
                    if {"reference_csi", "true_nmse", "raw_nmse_linear", "bits"} & inputs.keys():
                        raise ValueError("Supervision or audit fields leaked into public model input")
                    labels = ds.labels(fi)
                    residual, matrix = data_system(inputs)
                    _close(matrix, inputs["csi"][..., inputs["data_indices"]], "Four-symbol data-column extraction mismatch")
                    tx = inputs["known"].copy()
                    tx[:, inputs["data_indices"]] = labels["symbols"]
                    _close(residual-np.einsum("dij,dj->di", matrix, labels["symbols"]),
                        inputs["observations"]-np.einsum("dij,dj->di", inputs["csi"], tx), "Four-symbol pilot cancellation mismatch")
            report["splits"][split] = dict(frames_per_snr=ds.frame_count, samples_all_snr=len(ds), checked_all_frames=True,
                mean_raw_nmse_db=(10*np.log10(means)).tolist(), mean_assisted_nmse_db=(10*np.log10(assisted_mean)).tolist(),
                target_nmse_db=target.tolist(), error_scales=alpha.tolist(),
                target_max_error_db=float(np.max(abs(10*np.log10(assisted_mean)-target))),
                ber=ber_diagnostics(ds, backend, min(options["ber_diagnostic_frames_per_split"], 1 if c.get("smoke") else 64)))
            report["basis_sha256"] = array_sha256(m["basis"][:])
            report["delay_indices"] = delays.tolist()
            report["model_identity"] = m.attrs["identity"]
    files = {}
    for path in sorted(directory.glob("*.h5")):
        entry = dict(bytes=path.stat().st_size)
        if hash_files:
            entry["sha256"] = file_sha256(path)
        files[path.name] = entry
    report.update(complete=True, files=files, total_file_bytes=sum(value["bytes"] for value in files.values()),
        checks=dict(independent_record_splits=True, nonoverlapping_windows=True, independent_repeats=True,
            every_reference_and_physical_fingerprint=True, every_frame_full_support_nmse=True,
            exact_target_calibration=True, explicit_raw_and_assisted_views=True, four_symbol_pilot_cancellation=True,
            full_matrix_spot_checks=True, diagnostic_ber_scope_explicit=True))
    write_json(run/"acceptance.json", report)
    write_json(directory/"manifest.json", report)
    plot_report(run, c, report)
    return report


def plot_report(run, c, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, split in zip(axes, SPLITS, strict=True):
        value = report["splits"][split]
        ax.plot(c["snr_db"], value["mean_raw_nmse_db"], "o:", color="0.5", label="Raw receiver CSI")
        ax.plot(c["snr_db"], value["target_nmse_db"], "x--", color="black", label="Preset target", markersize=9)
        ax.plot(c["snr_db"], value["mean_assisted_nmse_db"], "o-", color="#2166AC", label="Reference-assisted CSI")
        ax.set(title=f"{split}: {value['frames_per_snr']} frames/SNR", xlabel="SNR (dB)", ylabel="Mean full-matrix NMSE (dB)", xticks=c["snr_db"])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"{c['channel']} | DDDD, CP={c['cp']} ({1000*c['cp']/c['fs_baseband']:g} ms), Q=24, 192 raw taps | full-support reference assistance")
    for ext in ("png", "pdf"):
        fig.savefig(run/f"nmse_snr.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, split in zip(axes, SPLITS, strict=True):
        value = report["splits"][split]["ber"]
        for view, line, label in (("raw", ":", "raw"), ("reference_assisted", "-", "assisted"), ("reference", "--", "reference")):
            for algorithm, color, marker in (("single", "#D77A22", "o"), ("full", "#2166AC", "s")):
                measured = np.asarray(value["curves"][view][algorithm]["ber"])
                floor = .5/value["bits_per_snr"]
                ax.semilogy(c["snr_db"], np.maximum(measured, floor), linestyle=line, marker=marker,
                    color=color, label=f"{algorithm} / {label}", markersize=4)
                zero = measured == 0
                ax.scatter(np.asarray(c["snr_db"])[zero], np.full(zero.sum(), floor), marker=marker,
                    facecolors="white", edgecolors=color, zorder=5)
        ax.set(title=f"{split}: {len(value['frame_indices'])} diagnostic frames", xlabel="SNR (dB)", ylabel="BER", xticks=c["snr_db"])
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"{c['channel']} | bounded BER diagnostics | open markers: zero measured errors")
    for ext in ("png", "pdf"):
        fig.savefig(run/f"ber_snr.{ext}", dpi=170)
    plt.close(fig)
    with (run/"summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "split", "snr_db", "frames_per_snr", "raw_nmse_db", "assisted_nmse_db", "target_nmse_db",
            "diagnostic_bits", "diagnostic_raw_single_ber", "diagnostic_raw_full_ber", "diagnostic_assisted_single_ber", "diagnostic_assisted_full_ber"])
        for split, value in report["splits"].items():
            for i, snr in enumerate(c["snr_db"]):
                curves = value["ber"]["curves"]
                writer.writerow([c["channel"], split, snr, value["frames_per_snr"], value["mean_raw_nmse_db"][i],
                    value["mean_assisted_nmse_db"][i], value["target_nmse_db"][i], value["ber"]["bits_per_snr"],
                    curves["raw"]["single"]["ber"][i], curves["raw"]["full"]["ber"][i],
                    curves["reference_assisted"]["single"]["ber"][i], curves["reference_assisted"]["full"]["ber"][i]])
