"""Paired Q=16, 24 and 32 raw-CSI LMMSE comparison on identical waveforms."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from data_generation.src.config import ROOT
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_lmmse import evaluate
from src.evaluation.four_data_q24_compare import CHANNELS, verify_q32, verify_reference, verify_source

ORDERS = (16, 24, 32)


def verify_q24_source(source, baseline, metadata):
    """Ensure the previous Q24 run used the same verified physical inputs."""
    baseline = Path(baseline).resolve()
    report = json.loads((baseline/"metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((baseline/"manifest.json").read_text(encoding="utf-8"))
    if (not report.get("complete") or report.get("metadata") != manifest
            or manifest.get("schema") != "four-data-paired-q24-q32-v1"
            or manifest.get("input_view") != "raw" or manifest.get("csi_compensation") is not False
            or manifest.get("options") != metadata["options"] or manifest.get("configs") != metadata["configs"]
            or manifest.get("physical") != metadata["physical"]
            or manifest.get("baseline_metrics_sha256") != metadata["baseline_metrics_sha256"]
            or manifest.get("baseline_manifest_sha256") != metadata["baseline_manifest_sha256"]
            or manifest.get("code_sha256") != metadata["code_sha256"]
            or set(report.get("channels", {})) != set(CHANNELS)):
        raise ValueError("Previous Q24 comparison is not bound to the same physical data and code")
    for channel in CHANNELS:
        if (set(report["channels"][channel]) != {"q24", "q32"}
                or not report["channels"][channel]["q24"].get("complete")
                or not report["channels"][channel]["q32"].get("complete")):
            raise ValueError("Previous Q24 comparison is incomplete")
    return report["channels"], dict(path=str(baseline),
        manifest_sha256=file_sha256(baseline/"manifest.json"),
        metrics_sha256=file_sha256(baseline/"metrics.json"))


def verify_order(previous, current, order):
    """Compare every frame and SNR against an earlier result for the same Q."""
    if (current["channel"] != previous["channel"] or current.get("q") != order or previous.get("q") != order
            or current["snr_db"] != previous["snr_db"] or current["bits_per_snr"] != previous["bits_per_snr"]
            or len(current["frames"]) != len(previous["frames"])):
        raise ValueError(f"Q{order} baseline frame or SNR coverage changed")
    max_nmse_delta = 0.0
    for a, b in zip(current["frames"], previous["frames"], strict=True):
        if (a["record"], a["start_seconds"]) != (b["record"], b["start_seconds"]):
            raise ValueError(f"Q{order} baseline frame identity changed")
        for field in ("raw_single_errors", "raw_full_errors", "reference_single_errors", "reference_full_errors"):
            if a[field] != b[field]:
                raise ValueError(f"Q{order} baseline changed bit errors: {field}")
        first, second = np.asarray(a["raw_nmse_linear"]), np.asarray(b["raw_nmse_linear"])
        if not np.allclose(first, second, rtol=1e-9, atol=1e-12):
            raise ValueError(f"Q{order} baseline NMSE does not reproduce")
        max_nmse_delta = max(max_nmse_delta, float(np.max(abs(first-second))))
    return dict(frames=len(current["frames"]), bit_errors_identical=True,
                max_absolute_linear_nmse_difference=max_nmse_delta)


def metadata_for_orders(source, q24_source, backend):
    metadata, previous32 = verify_source(source, backend)
    previous24, identity = verify_q24_source(source, q24_source, metadata)
    for channel in CHANNELS:
        verify_q32(previous32[channel], previous24[channel]["q32"])
        verify_reference(previous24[channel]["q24"], previous24[channel]["q32"])
        path = Path(metadata["physical"][channel]["path"])
        with h5py.File(path, "r") as h:
            c = metadata["configs"][channel]
            nw = len(h["basis"])*c["dpss_half_bandwidth_hz"]/c["fs_baseband"]
            metadata["measured_dpss_tail_ratio"][channel]["16"] = float(
                max(0., 1-np.sum(h["concentration"][:16])/(2*nw)))
    metadata.update(schema="four-data-paired-q16-q24-q32-v1", orders=list(ORDERS),
                    previous_q24=identity, tail_ratio_threshold_enforced_for_q16=False)
    for name in ("src/evaluation/four_data_q16_compare.py", "scripts/evaluate/four_data_q16_compare.py"):
        metadata["code_sha256"][name] = file_sha256(ROOT/name)
    return metadata, previous32, previous24


def finish(output, metadata, rows, checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    write_json(output/"metrics.json", dict(complete=True, metadata=metadata, channels=rows, reproduction=checks))
    with (output/"summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        header = ["channel", "snr_db", "bits"]
        for q in ORDERS:
            header += [f"q{q}_single_errors", f"q{q}_single_ber", f"q{q}_full_errors", f"q{q}_full_ber"]
        header += ["reference_single_errors", "reference_single_ber", "reference_full_errors",
                   "reference_full_ber"]+[f"q{q}_nmse_db" for q in ORDERS]+["reference_nmse_linear", "reference_nmse_db"]
        writer.writerow(header)
        for channel, result in rows.items():
            for i, snr in enumerate(result["q16"]["snr_db"]):
                line = [channel, snr, result["q16"]["bits_per_snr"]]
                for q in ORDERS:
                    for key in ("raw_single", "raw_full"):
                        curve = result[f"q{q}"]["curves"][key]
                        line += [curve["bit_errors"][i], curve["ber"][i]]
                for key in ("reference_single", "reference_full"):
                    curve = result["q16"]["curves"][key]
                    line += [curve["bit_errors"][i], curve["ber"][i]]
                writer.writerow(line+[result[f"q{q}"]["mean_raw_nmse_db"][i] for q in ORDERS]+[0, "-inf"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    colors = {16: "#D66B17", 24: "#13845A", 32: "#2766B3"}
    markers = {16: "o", 24: "s", 32: "D"}
    for ax, (channel, result) in zip(axes, rows.items(), strict=True):
        for q in ORDERS:
            report = result[f"q{q}"]
            for key, label, style in (("raw_single", "single-tap", ":"), ("raw_full", "full-matrix", "-")):
                values = np.asarray(report["curves"][key]["ber"])
                ax.plot(report["snr_db"], np.maximum(values, .5/report["bits_per_snr"]), linestyle=style,
                        marker=markers[q], markersize=4.4, color=colors[q], linewidth=1.45,
                        label=f"Q{q} {label}")
        report = result["q16"]
        for key, label, color, marker in (("reference_single", "Reference single-tap", "#805939", "^"),
                                           ("reference_full", "Reference full-matrix", "#303440", "v")):
            values = np.asarray(report["curves"][key]["ber"])
            floor = .5/report["bits_per_snr"]
            ax.plot(report["snr_db"], np.maximum(values, floor), linestyle="--", marker=marker,
                    color=color, label=label, linewidth=1.6, markersize=4.7)
            zero = values == 0
            if np.any(zero):
                ax.scatter(np.asarray(report["snr_db"])[zero], np.full(zero.sum(), floor), marker=marker,
                           facecolors="white", edgecolors=color, zorder=4)
        ax.set(title=f"{channel} ({report['bits_per_snr']:,} bits/SNR)", xlabel="SNR (dB)", ylabel="BER",
               xticks=report["snr_db"])
        ax.set_yscale("log")
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Paired raw CSI: Q=16, 24, 32 | zero errors at half-count floor")
    fig.savefig(output/"ber_snr.png", dpi=180)
    fig.savefig(output/"ber_snr.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout="constrained")
    for ax, (channel, result) in zip(axes, rows.items(), strict=True):
        for q in ORDERS:
            report = result[f"q{q}"]
            ax.plot(report["snr_db"], report["mean_raw_nmse_db"], marker=markers[q], color=colors[q],
                    linewidth=1.7, label=f"Estimated CSI, Q={q}")
        ax.text(.03, .05, "Reference CSI: NMSE = 0 (−∞ dB)", transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=.95), fontsize=9)
        ax.set(title=channel, xlabel="SNR (dB)", ylabel="Mean full-matrix CSI NMSE (dB)",
               xticks=result["q16"]["snr_db"])
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle("4D, 512 data + 256 pilots | 12 Hz / L=176 | no CSI compensation")
    fig.savefig(output/"nmse_snr.png", dpi=180)
    fig.savefig(output/"nmse_snr.pdf")
    plt.close(fig)


def run(source, q24_source, output, backend):
    source, q24_source, output = (Path(path).resolve() for path in (source, q24_source, output))
    if output in (source, q24_source):
        raise ValueError("Q comparison requires a separate output directory")
    metadata, previous32, previous24 = metadata_for_orders(source, q24_source, backend)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory belongs to a different comparison")
    write_json(manifest, metadata)
    rows, checks = {}, {}
    for channel in CHANNELS:
        result = {f"q{q}": evaluate(channel, output/f"q{q}", metadata, backend, physical_root=source, order=q)
                  for q in (32, 24, 16)}
        checks[channel] = dict(q32=verify_q32(previous32[channel], result["q32"]),
                               q24=verify_order(previous24[channel]["q24"], result["q24"], 24))
        verify_reference(result["q16"], result["q32"])
        verify_reference(result["q24"], result["q32"])
        rows[channel] = result
        if file_sha256(source/channel.lower()/"physical.h5") != metadata["physical"][channel]["sha256"]:
            raise ValueError("Physical source changed during the comparison")
    finish(output, metadata, rows, checks)
    return output
