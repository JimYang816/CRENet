"""Paired Q=24 versus Q=32 raw-CSI LMMSE comparison on identical waveforms."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_lmmse import CODE_FILES, channel_config, evaluate, load_run_config

ORDERS = (24, 32)
CHANNELS = ("NOF1", "NCS1")


def validate_source_config(channel, source_config):
    """Permit only the order difference between current Q24 and the Q32 source."""
    active = channel_config(channel)
    if (source_config.get("dpss_expected_count") != 32
            or source_config != dict(active, dpss_expected_count=32)):
        raise ValueError("Q32 source differs from the active waveform parameters beyond DPSS order")


def verify_source(source, backend):
    """Authenticate the original run and bind both orders to its physical data."""
    source = Path(source).resolve()
    old = json.loads((source/"metrics.json").read_text(encoding="utf-8"))
    original = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    expected_options = load_run_config(ROOT/"configs/evaluation/four_data_lmmse.json")
    if (not old.get("complete") or old.get("metadata") != original
            or original.get("schema") != "four-data-raw-lmmse-v1"
            or original.get("input_view") != "raw" or original.get("csi_compensation") is not False
            or original.get("options") != expected_options or set(old.get("channels", {})) != set(CHANNELS)):
        raise ValueError("Q32 baseline is not the completed, uncompensated paired experiment")
    paths, tails = {}, {}
    for channel in CHANNELS:
        c = original["configs"][channel]
        validate_source_config(channel, c)
        path = source/channel.lower()/"physical.h5"
        expected_identity = digest(dict(schema=original["schema"], config=c, options=expected_options,
            archive=original["archive_sha256"], code=original["code_sha256"]))
        with h5py.File(path, "r") as h:
            if (not h.attrs.get("complete") or h.attrs.get("identity") != expected_identity
                    or h.attrs.get("channel") != channel or h["basis"].shape[1] != 32
                    or h["prior"].shape != (len(expected_options["prior_records"]), 32, c["delay_length"])
                    or not np.all(h["prior_complete"][:])):
                raise ValueError(f"{channel}: source physical identity or prior is invalid")
            groups = list(h["test"].values())
            records = [int(group.attrs["record"]) for group in groups]
            if (records != expected_options["test_records"] or any(not group.attrs.get("complete") for group in groups)
                    or any(float(group.attrs["start_seconds"]) != expected_options["start_seconds"] for group in groups)):
                raise ValueError(f"{channel}: source test frames differ from the original run")
            concentration = h["concentration"][:]
            nw = len(h["basis"])*c["dpss_half_bandwidth_hz"]/c["fs_baseband"]
            tails[channel] = {str(q): float(max(0., 1-np.sum(concentration[:q])/(2*nw))) for q in ORDERS}
        paths[channel] = dict(path=str(path), sha256=file_sha256(path), physical_identity=expected_identity)
    code = (*CODE_FILES, "src/evaluation/four_data_q24_compare.py", "scripts/evaluate/four_data_q24_compare.py")
    metadata = dict(schema="four-data-paired-q24-q32-v1",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
        code_sha256={name: file_sha256(ROOT/name) for name in code}, device=backend.device,
        options=expected_options, configs=original["configs"], orders=list(ORDERS),
        config_order_note="configs record the Q32 source; each evaluation overrides only dpss_expected_count with its explicit Q",
        input_view="raw", csi_compensation=False, archive_sha256=original["archive_sha256"],
        baseline_git_head=original["git_head"], baseline_manifest_sha256=file_sha256(source/"manifest.json"),
        baseline_metrics_sha256=file_sha256(source/"metrics.json"), physical=paths,
        basis_selection="first Q vectors of the same full-frame 12-Hz DPSS basis; Q is fixed explicitly",
        tail_ratio_threshold=original["configs"]["NOF1"]["dpss_tail_ratio"],
        tail_ratio_threshold_enforced_for_q24=False, measured_dpss_tail_ratio=tails,
        estimator="fresh prior fit and raw joint CSI estimate for each Q; no saved CSI or detector outputs used",
        reference_use="independent prior construction, ideal-reference detector and NMSE scoring; no CSI adjustment",
        paired="same five physical frames, data, standard noise, SNR, delay support and receiver model")
    return metadata, old["channels"]


def verify_q32(previous, current):
    """Reproduce every Q32 frame before using it as the paired baseline."""
    if (current["channel"] != previous["channel"] or current["q"] != 32
            or current["snr_db"] != previous["snr_db"] or current["bits_per_snr"] != previous["bits_per_snr"]
            or len(current["frames"]) != len(previous["frames"])):
        raise ValueError("Q32 baseline frame or SNR coverage changed")
    max_nmse_delta = 0.0
    for a, b in zip(current["frames"], previous["frames"], strict=True):
        if (a["record"], a["start_seconds"]) != (b["record"], b["start_seconds"]):
            raise ValueError("Q32 baseline frame identity changed")
        for field in ("raw_single_errors", "raw_full_errors", "reference_single_errors", "reference_full_errors"):
            if a[field] != b[field]:
                raise ValueError(f"Q32 baseline changed bit errors: {field}")
        first, second = np.asarray(a["raw_nmse_linear"]), np.asarray(b["raw_nmse_linear"])
        if not np.allclose(first, second, rtol=1e-9, atol=1e-12):
            raise ValueError("Q32 baseline NMSE does not reproduce")
        max_nmse_delta = max(max_nmse_delta, float(np.max(abs(first-second))))
    return dict(frames=len(current["frames"]), bit_errors_identical=True,
                max_absolute_linear_nmse_difference=max_nmse_delta)


def verify_reference(q24, q32):
    if (q24["snr_db"] != q32["snr_db"] or q24["bits_per_snr"] != q32["bits_per_snr"]
            or len(q24["frames"]) != len(q32["frames"])):
        raise ValueError("Q24 and Q32 do not have paired sample coverage")
    for a, b in zip(q24["frames"], q32["frames"], strict=True):
        if (a["record"], a["start_seconds"]) != (b["record"], b["start_seconds"]):
            raise ValueError("Q24 and Q32 use different test frames")
        for field in ("reference_single_errors", "reference_full_errors"):
            if a[field] != b[field]:
                raise ValueError("Reference CSI detection changed between paired runs")


def finish(output, metadata, rows, checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    write_json(output/"metrics.json", dict(complete=True, metadata=metadata, channels=rows, q32_reproduction=checks))
    with (output/"summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "snr_db", "bits", "q24_single_errors", "q24_single_ber",
            "q24_full_errors", "q24_full_ber", "q32_single_errors", "q32_single_ber", "q32_full_errors",
            "q32_full_ber", "reference_single_errors", "reference_single_ber", "reference_full_errors",
            "reference_full_ber", "q24_nmse_db", "q32_nmse_db", "reference_nmse_linear", "reference_nmse_db"])
        for channel, result in rows.items():
            q24, q32 = result["q24"], result["q32"]
            for i, snr in enumerate(q24["snr_db"]):
                line = [channel, snr, q24["bits_per_snr"]]
                for report, key in ((q24, "raw_single"), (q24, "raw_full"), (q32, "raw_single"),
                                    (q32, "raw_full"), (q24, "reference_single"), (q24, "reference_full")):
                    line += [report["curves"][key]["bit_errors"][i], report["curves"][key]["ber"][i]]
                writer.writerow(line+[q24["mean_raw_nmse_db"][i], q32["mean_raw_nmse_db"][i], 0, "-inf"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    styles = (("q24", "raw_single", "Q24 single-tap", "#DB761E", "o", "-"),
              ("q32", "raw_single", "Q32 single-tap", "#DB761E", "o", ":"),
              ("q24", "raw_full", "Q24 full-matrix", "#2468B4", "s", "-"),
              ("q32", "raw_full", "Q32 full-matrix", "#2468B4", "s", ":"),
              ("q24", "reference_single", "Reference single-tap", "#925128", "^", "--"),
              ("q24", "reference_full", "Reference full-matrix", "#243D71", "v", "--"))
    for ax, (channel, result) in zip(axes, rows.items(), strict=True):
        for order, key, label, color, marker, line in styles:
            report = result[order]
            values = np.asarray(report["curves"][key]["ber"])
            floor = .5/report["bits_per_snr"]
            ax.plot(report["snr_db"], np.maximum(values, floor), linestyle=line, marker=marker,
                    color=color, label=label, linewidth=1.6, markersize=5)
            zero = values == 0
            if np.any(zero):
                ax.scatter(np.asarray(report["snr_db"])[zero], np.full(zero.sum(), floor), marker=marker,
                           facecolors="white", edgecolors=color, zorder=4)
        ax.set(title=f"{channel} ({result['q24']['bits_per_snr']:,} bits/SNR)", xlabel="SNR (dB)", ylabel="BER",
               xticks=result["q24"]["snr_db"])
        ax.set_yscale("log")
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=7.5)
    fig.suptitle("Paired raw CSI: Q=24 vs Q=32 | zero errors at half-count floor")
    fig.savefig(output/"ber_snr.png", dpi=180)
    fig.savefig(output/"ber_snr.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout="constrained")
    for ax, (channel, result) in zip(axes, rows.items(), strict=True):
        for key, color, marker in (("q24", "#D97706", "o"), ("q32", "#2468B4", "s")):
            report = result[key]
            ax.plot(report["snr_db"], report["mean_raw_nmse_db"], marker=marker, color=color,
                    linewidth=1.7, label=f"Estimated CSI, Q={report['q']}")
        ax.text(.03, .05, "Reference CSI: NMSE = 0 (−∞ dB)", transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=.95), fontsize=9)
        ax.set(title=channel, xlabel="SNR (dB)", ylabel="Mean full-matrix CSI NMSE (dB)", xticks=result["q24"]["snr_db"])
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle("4D, 512 data + 256 pilots | 12 Hz / L=176 | no CSI compensation")
    fig.savefig(output/"nmse_snr.png", dpi=180)
    fig.savefig(output/"nmse_snr.pdf")
    plt.close(fig)


def run(source, output, backend):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Q comparison requires a separate output directory")
    metadata, previous = verify_source(source, backend)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory belongs to a different comparison")
    write_json(manifest, metadata)
    rows, checks = {}, {}
    for channel in CHANNELS:
        q32 = evaluate(channel, output/"q32_recheck", metadata, backend, physical_root=source, order=32)
        checks[channel] = verify_q32(previous[channel], q32)
        q24 = evaluate(channel, output/"q24", metadata, backend, physical_root=source, order=24)
        verify_reference(q24, q32)
        rows[channel] = dict(q24=q24, q32=q32)
        if file_sha256(source/channel.lower()/"physical.h5") != metadata["physical"][channel]["sha256"]:
            raise ValueError("Physical source changed during the comparison")
    finish(output, metadata, rows, checks)
    return output
