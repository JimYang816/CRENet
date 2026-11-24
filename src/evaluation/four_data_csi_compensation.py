"""Reference-assisted CSI sensitivity experiment on the saved four-data frames.

The compensated channel is an oracle construction, not a receiver estimator.
All 1024 delay coordinates of the reference are included in the full matrix.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.pdd import JointEstimator
from data_generation.src.transmitter.ofdm import carrier_masks, decisions
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_lmmse import CODE_FILES, channel_config, detect, load_run_config
from src.evaluation.four_data_q_sweep import raw_nmse


CHANNELS = ("NOF1", "NCS1")
GAIN_DB = 5.0
ALPHA = 10 ** (-GAIN_DB / 20)
NEW_CODE = ("src/evaluation/four_data_csi_compensation.py",
            "scripts/evaluate/four_data_csi_compensation.py")


def compensate(estimated, reference):
    """Apply H_ref + alpha * (H_est - H_ref), after zero-extending raw taps."""
    estimated = np.asarray(estimated, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if (estimated.ndim != 4 or reference.ndim != 3
            or estimated.shape[1:3] != reference.shape[:2]
            or estimated.shape[-1] > reference.shape[-1]):
        raise ValueError("Estimated and reference CSI dimensions do not match")
    corrected = np.empty((estimated.shape[0], *reference.shape), dtype=np.complex128)
    np.multiply(reference[None], 1 - ALPHA, out=corrected)
    corrected[..., :estimated.shape[-1]] += ALPHA * estimated
    return corrected


def tap_nmse(taps, reference):
    """Per-SNR and per-symbol full-matrix NMSE via the unitary FFT identity."""
    if taps.ndim != 4 or taps.shape[1:] != reference.shape:
        raise ValueError("Expected one full-support CSI per SNR")
    result = np.empty((len(taps), len(reference)))
    for symbol in range(len(reference)):
        energy = np.sum(np.abs(reference[symbol]) ** 2)
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("Reference channel energy must be positive and finite")
        result[:, symbol] = np.sum(np.abs(taps[:, symbol] - reference[symbol]) ** 2, axis=(1, 2)) / energy
    return result


def verify_source(source):
    source = Path(source).resolve()
    manifest_path, metrics_path = source / "manifest.json", source / "metrics.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    options = load_run_config(ROOT / "configs/evaluation/four_data_lmmse.json")
    if (original.get("schema") != "four-data-raw-lmmse-v1" or original.get("csi_compensation") is not False
            or original.get("input_view") != "raw" or original.get("options") != options
            or not metrics.get("complete") or metrics.get("metadata") != original
            or set(metrics.get("channels", {})) != set(CHANNELS)):
        raise ValueError("Source is not the completed uncompensated four-data experiment")
    for name, expected in original["code_sha256"].items():
        payload = (ROOT / name).read_bytes()
        raw = hashlib.sha256(payload).hexdigest()
        normalized = hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()
        if expected not in (raw, normalized):
            raise ValueError(f"Baseline code changed: {name}")
    physical = {}
    for channel in CHANNELS:
        c, report = original["configs"][channel], metrics["channels"][channel]
        if (c != channel_config(channel) or c["cp"] != 256 or c["dpss_expected_count"] != 24
                or not report.get("complete") or report.get("q") != 24
                or report.get("channel") != channel or report.get("snr_db") != c["snr_db"]):
            raise ValueError(f"{channel}: source parameters or report do not match the current experiment")
        path = source / channel.lower() / "physical.h5"
        expected = digest(dict(schema=original["schema"], config=c, options=options,
                               archive=original["archive_sha256"], code=original["code_sha256"]))
        with h5py.File(path, "r") as h:
            groups = list(h["test"].values())
            identities = [(int(g.attrs["record"]), float(g.attrs["start_seconds"])) for g in groups]
            baseline = [(f["record"], f["start_seconds"]) for f in report["frames"]]
            if (not h.attrs.get("complete") or h.attrs.get("identity") != expected
                    or h.attrs.get("channel") != channel or h["basis"].shape != (5120, 24)
                    or not np.all(h["prior_complete"][:]) or identities != baseline
                    or [record for record, _ in identities] != options["test_records"]
                    or any(start != options["start_seconds"] for _, start in identities)
                    or any(not g.attrs.get("complete") for g in groups)):
                raise ValueError(f"{channel}: physical data identity or test frames differ from baseline")
        physical[channel] = dict(path=str(path), sha256=file_sha256(path), identity=expected)
    return original, metrics["channels"], physical


def experiment_metadata(source, original, physical, backend):
    return dict(schema="four-data-reference-assisted-plus5db-v1",
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                    cwd=ROOT, text=True).strip()),
                code_sha256={name: file_sha256(ROOT / name) for name in (*CODE_FILES, *NEW_CODE)},
                source=dict(path=str(source), manifest_sha256=file_sha256(source / "manifest.json"),
                            metrics_sha256=file_sha256(source / "metrics.json"),
                            source_git_head=original["git_head"], physical=physical),
                configs=original["configs"], options=original["options"], input_view="reference-assisted",
                csi_compensation=True, compensation_db=GAIN_DB, alpha=ALPHA,
                compensation="H_ref + alpha * (H_raw - H_ref); raw CSI zero-extended to full reference delay support",
                reference_use="oracle CSI construction, reference detector, independent prior, and NMSE scoring",
                comparison="identical saved physical frames, payloads and noise; raw/reference detector outcomes reverified",
                precision="complex128", device=backend.device,
                snr_definition=original["snr_definition"], nmse_definition=original["nmse_definition"])


def verify_reproduction(channel, name, actual, expected):
    if not np.array_equal(actual, np.asarray(expected)):
        raise ValueError(f"{channel}: {name} error counts differ from the saved raw/reference baseline")


def evaluate(channel, source, output, metadata, baseline, backend):
    c, options = metadata["configs"][channel], metadata["options"]
    result_path = output / channel.lower() / "metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(metadata=metadata, channel=channel))
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if report.get("identity") != identity or not report.get("complete"):
            raise ValueError(f"{channel}: cached result does not match this run")
        return report
    with h5py.File(source / channel.lower() / "physical.h5", "r") as h:
        model = DPSSModel(h["basis"][:], c["dpss_half_bandwidth_hz"], c["dpss_tail_ratio"], c["delay_length"], c)
        known, noise = h["known"][:], h["unit_noise"][:]
        groups = list(h["test"].values())
        observations = np.stack([g["clean"][:][None] + np.sqrt(g["variance"][:])[:, None, None]
                                 * g["standard"][:][None] for g in groups])
        variances = np.stack([g["variance"][:] for g in groups])
        prior = FrozenPrior.fit_coefficients(model, h["prior"][:], c["empirical_weight"], c["prior_scale"],
            options["prior_records"], regularizer=c["relative_regularizer"], delay_block_size=c["delay_block_size"],
            interference_scale=c["interference_scale"])
        layout = carrier_masks(c)
        print(f"{channel}: recomputing raw Q=24 CSI", flush=True)
        estimator = JointEstimator(prior, known, layout, noise, backend)
        coefficients = estimator.estimate_coefficients(observations.reshape(-1, 4, c["nfft"]), known,
            layout, variances.ravel()).reshape(len(groups), len(c["snr_db"]), model.count, model.length)
        del estimator, prior
        if backend.torch:
            backend.torch.cuda.empty_cache()
        sampled = model.sampled_basis.reshape(4, c["nfft"], model.count)
        frames = []
        max_nmse_relative_error = 0.0
        for i, (group, old) in enumerate(zip(groups, baseline["frames"], strict=True)):
            begin = time.perf_counter()
            estimated = np.einsum("dnq,sql->sdnl", sampled, coefficients[i], optimize=True)
            truth, bits = group["truth"][:], group["bits"][:]
            nmse = raw_nmse(model, coefficients[i], truth)
            if not np.allclose(nmse, old["raw_nmse_linear"], rtol=1e-9, atol=1e-12):
                raise ValueError(f"{channel}: raw CSI NMSE does not reproduce")
            for name, taps in (("raw", estimated), ("reference", truth)):
                single, full = detect(taps, observations[i], known, variances[i], noise, layout["data"], backend)
                errors = [np.count_nonzero(decisions(x) != bits[None], axis=(-1, -2)) for x in (single, full)]
                for algorithm, errors_this in zip(("single", "full"), errors, strict=True):
                    verify_reproduction(channel, f"{name}_{algorithm}", errors_this, old[f"{name}_{algorithm}_errors"])
            corrected = compensate(estimated, truth)
            corrected_nmse = tap_nmse(corrected, truth)
            expected_nmse = ALPHA**2 * nmse
            if not np.allclose(corrected_nmse, expected_nmse, rtol=1e-10, atol=1e-14):
                raise ValueError(f"{channel}: compensated full-matrix NMSE is not 5 dB below raw NMSE")
            max_nmse_relative_error = max(max_nmse_relative_error, float(np.max(
                np.abs(corrected_nmse - expected_nmse) / np.maximum(expected_nmse, 1e-300))))
            single, full = detect(corrected, observations[i], known, variances[i], noise, layout["data"], backend)
            errors = [np.count_nonzero(decisions(x) != bits[None], axis=(-1, -2)) for x in (single, full)]
            frames.append(dict(old, compensated_single_errors=errors[0].tolist(),
                               compensated_full_errors=errors[1].tolist(),
                               compensated_nmse_linear=corrected_nmse.tolist()))
            del corrected, estimated, truth
            if backend.torch:
                backend.torch.cuda.empty_cache()
            print(f"{channel}: frame {i+1}/{len(groups)} checked and compensated, {time.perf_counter()-begin:.1f}s", flush=True)
    bits_per_snr = baseline["bits_per_snr"]
    curves = dict(baseline["curves"])
    for algorithm in ("single", "full"):
        errors = np.asarray([f[f"compensated_{algorithm}_errors"] for f in frames]).sum(axis=(0, 2))
        curves[f"compensated_{algorithm}"] = dict(bit_errors=errors.tolist(), ber=(errors / bits_per_snr).tolist())
    mean = np.asarray([f["compensated_nmse_linear"] for f in frames]).mean(axis=(0, 2))
    report = dict(baseline, identity=identity, frames=frames, curves=curves,
                  mean_compensated_nmse_linear=mean.tolist(), mean_compensated_nmse_db=(10*np.log10(mean)).tolist(),
                  verification=dict(raw_reference_error_counts_identical=True, raw_nmse_reproduced=True,
                                    max_compensated_nmse_relative_error=max_nmse_relative_error))
    write_json(result_path, report)
    return report


def finish(output, metadata, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_json(output / "metrics.json", dict(complete=True, metadata=metadata, channels=reports))
    keys = tuple(f"{view}_{algorithm}" for view in ("raw", "compensated", "reference")
                 for algorithm in ("single", "full"))
    has_targets = all("target_nmse_db" in report for report in reports.values())
    extra_columns = ["target_nmse_db", "error_scale"] if has_targets else []
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "snr_db", "bits"] + [field for key in keys for field in (f"{key}_errors", f"{key}_ber")]
                        + ["raw_nmse_db", "compensated_nmse_db", "reference_nmse_linear", "reference_nmse_db"] + extra_columns)
        for channel, report in reports.items():
            for i, snr in enumerate(report["snr_db"]):
                row = [channel, snr, report["bits_per_snr"]]
                for key in keys:
                    row += [report["curves"][key]["bit_errors"][i], report["curves"][key]["ber"][i]]
                extra = [report["target_nmse_db"][i], report["error_scales"][i]] if has_targets else []
                writer.writerow(row + [report["mean_raw_nmse_db"][i], report["mean_compensated_nmse_db"][i], 0, "-inf"] + extra)
    assisted_label = metadata.get("assisted_label", "assisted (+5 dB)")
    styles = (("raw_single", "Single / raw", "#D77A22", "o", ":"),
              ("compensated_single", f"Single / {assisted_label}", "#D77A22", "o", "-"),
              ("reference_single", "Single / reference", "#D77A22", "o", "--"),
              ("raw_full", "Full / raw", "#2166AC", "s", ":"),
              ("compensated_full", f"Full / {assisted_label}", "#2166AC", "s", "-"),
              ("reference_full", "Full / reference", "#2166AC", "s", "--"))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    for ax, (channel, report) in zip(axes, reports.items(), strict=True):
        for key, label, color, marker, line in styles:
            values = np.asarray(report["curves"][key]["ber"])
            floor = .5 / report["bits_per_snr"]
            ax.plot(report["snr_db"], np.maximum(values, floor), linestyle=line, marker=marker, color=color,
                    label=label, linewidth=1.7, markersize=5)
            zero = values == 0
            if np.any(zero):
                ax.scatter(np.asarray(report["snr_db"])[zero], np.full(zero.sum(), floor), marker=marker,
                           facecolors="white", edgecolors=color, zorder=4)
        ax.set(title=f"{channel} ({report['bits_per_snr']:,} bits/SNR)", xlabel="SNR (dB)", ylabel="BER",
               yscale="log", xticks=report["snr_db"])
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(metadata.get("ber_plot_title", "CP=256, Q=24 | raw vs reference-assisted CSI vs reference | zero errors at half-count floor"))
    fig.savefig(output / "ber_snr.png", dpi=180)
    fig.savefig(output / "ber_snr.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout="constrained")
    for ax, (channel, report) in zip(axes, reports.items(), strict=True):
        ax.plot(report["snr_db"], report["mean_raw_nmse_db"], "o:", color="#777777", label="Raw CSI")
        if has_targets:
            ax.plot(report["snr_db"], report["target_nmse_db"], "x--", color="black", markersize=9, label="Preset target")
        ax.plot(report["snr_db"], report["mean_compensated_nmse_db"], "o-", color="#2166AC",
                label=f"Reference-{assisted_label}")
        ax.text(.03, .50, "Reference CSI: NMSE = 0 (−∞ dB)", transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=.95), fontsize=9)
        ax.set(title=channel, xlabel="SNR (dB)", ylabel="Mean full-matrix CSI NMSE (dB)", xticks=report["snr_db"])
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle(metadata.get("nmse_plot_title", "4D, CP=256 | 512 data + 256 pilots | 12 Hz / Q=24 / L=176"))
    fig.savefig(output / "nmse_snr.png", dpi=180)
    fig.savefig(output / "nmse_snr.pdf")
    plt.close(fig)


def run(source, output, backend):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source experiment")
    original, baseline, physical = verify_source(source)
    metadata = experiment_metadata(source, original, physical, backend)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory belongs to a different experiment")
    write_json(manifest, metadata)
    reports = {ch: evaluate(ch, source, output, metadata, baseline[ch], backend) for ch in CHANNELS}
    finish(output, metadata, reports)
    return output
