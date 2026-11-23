"""Plot an existing CP=256 and CP=384 Q24 experiment on one figure.

This is a retrospective comparison of separately generated physical frames.
It does not rerun the channel or imply paired noise samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CHANNELS = ("NOF1", "NCS1")
CURVES = {
    "raw_single": ("Estimated CSI, single-tap", "#0072B2", "o"),
    "raw_full": ("Estimated CSI, full-matrix", "#D55E00", "s"),
    "reference_single": ("Reference CSI, single-tap", "#009E73", "^"),
    "reference_full": ("Reference CSI, full-matrix", "#CC79A7", "D"),
}
CP_STYLE = {256: ("-", True), 384: ("--", False)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source(directory, cp):
    manifest_path, metrics_path = directory / "manifest.json", directory / "metrics.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    schema = "four-data-raw-lmmse-v1" if cp == 256 else "four-data-paired-q24-q32-v1"
    if (metadata.get("schema") != schema or metrics.get("metadata") != metadata
            or not metrics.get("complete") or metadata.get("input_view") != "raw"
            or metadata.get("csi_compensation") is not False
            or set(metrics.get("channels", {})) != set(CHANNELS)):
        raise ValueError(f"CP={cp}: incompatible or incomplete experiment")
    reports = {}
    for channel in CHANNELS:
        config = metadata["configs"][channel]
        report = metrics["channels"][channel]
        if cp == 384:
            report = report["q24"]
        if (config["cp"] != cp or report.get("q") != 24 or report.get("channel") != channel
                or not report.get("complete") or report.get("reference_nmse_linear") != 0):
            raise ValueError(f"CP={cp}, {channel}: incompatible configuration or Q24 report")
        expected_frames = [(record, metadata["options"]["start_seconds"])
                           for record in metadata["options"]["test_records"]]
        frames = [(frame["record"], frame["start_seconds"]) for frame in report["frames"]]
        if frames != expected_frames:
            raise ValueError(f"CP={cp}, {channel}: reported test frames differ from the manifest")
        snr, bits = report["snr_db"], report["bits_per_snr"]
        if snr != list(range(0, 31, 5)) or bits <= 0 or len(report["mean_raw_nmse_db"]) != len(snr):
            raise ValueError(f"CP={cp}, {channel}: invalid SNR coverage or bit count")
        if not all(math.isfinite(v) for v in report["mean_raw_nmse_db"]):
            raise ValueError(f"CP={cp}, {channel}: invalid NMSE")
        for key in CURVES:
            curve = report["curves"][key]
            errors, bers = curve["bit_errors"], curve["ber"]
            if (len(errors) != len(snr) or len(bers) != len(snr)
                    or any(not 0 <= n <= bits or not math.isclose(b, n / bits, abs_tol=1e-14)
                           for n, b in zip(errors, bers, strict=True))):
                raise ValueError(f"CP={cp}, {channel}: inconsistent {key} BER")
        reports[channel] = report
    return metadata, reports


def compare_sources(older, newer):
    if (older["options"] != newer["options"] or older["archive_sha256"] != newer["archive_sha256"]
            or 24 not in older["orders"]):
        raise ValueError("Experiments do not share the same records, options and WATERMARK archive")
    for channel in CHANNELS:
        a, b = (dict(metadata["configs"][channel]) for metadata in (older, newer))
        if a.pop("cp") != 384 or b.pop("cp") != 256:
            raise ValueError("Unexpected CP values")
        # The historical manifest records the Q32 source, while its Q24 report is a fresh Q24 estimate.
        a.pop("dpss_expected_count")
        if b.pop("dpss_expected_count") != 24 or a != b:
            raise ValueError(f"{channel}: configuration changed beyond CP and the source Q override")


def write_summary(output, reports):
    fields = ["channel", "cp", "snr_db", "bits"]
    fields += [item for key in CURVES for item in (f"{key}_errors", f"{key}_ber")]
    fields += ["raw_nmse_db", "reference_nmse_linear", "reference_nmse_db"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for channel in CHANNELS:
            for cp in (256, 384):
                report = reports[cp][channel]
                for i, snr in enumerate(report["snr_db"]):
                    row = dict(channel=channel, cp=cp, snr_db=snr, bits=report["bits_per_snr"],
                               raw_nmse_db=report["mean_raw_nmse_db"][i],
                               reference_nmse_linear=0, reference_nmse_db="-inf")
                    for key in CURVES:
                        row[f"{key}_errors"] = report["curves"][key]["bit_errors"][i]
                        row[f"{key}_ber"] = report["curves"][key]["ber"][i]
                    writer.writerow(row)


def draw(output, reports):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11})
    report = reports[256][CHANNELS[0]]
    bits, frame_count = report["bits_per_snr"], len(report["frames"])
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.30, 1]})
    fig.subplots_adjust(left=.075, right=.975, bottom=.09, top=.79, hspace=.26, wspace=.23)
    fig.suptitle("CP length comparison | D-D-D-D | Q=24, L=176, 12 Hz", fontsize=17, y=.975)
    fig.text(.5, .935, f"{frame_count} frames per channel  |  {bits:,} bits per SNR  |  no CSI compensation",
             ha="center", fontsize=11, color="#414141")
    curve_handles = [Line2D([], [], color=color, marker=marker, linewidth=1.8, markersize=6,
                            label=label) for label, color, marker in CURVES.values()]
    fig.legend(handles=curve_handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=4,
               frameon=False, fontsize=10.5, columnspacing=1.6, handlelength=2.2)
    cp_handles = [Line2D([], [], color="#333333", linestyle=line, marker="o", linewidth=1.8,
                         markerfacecolor="#333333" if filled else "white", markersize=6,
                         label=f"CP = {cp}") for cp, (line, filled) in CP_STYLE.items()]
    fig.legend(handles=cp_handles, loc="upper center", bbox_to_anchor=(.5, .868), ncol=2,
               frameon=False, fontsize=10.5, columnspacing=3)

    for col, channel in enumerate(CHANNELS):
        ber_ax, nmse_ax = axes[0, col], axes[1, col]
        for cp, (line, filled) in CP_STYLE.items():
            report = reports[cp][channel]
            x, bits = report["snr_db"], report["bits_per_snr"]
            for key, (_, color, marker) in CURVES.items():
                values = report["curves"][key]["ber"]
                shown = [max(value, .5 / bits) for value in values]
                ber_ax.plot(x, shown, linestyle=line, color=color, marker=marker, linewidth=1.75,
                            markersize=5.5, markerfacecolor=color if filled else "white",
                            markeredgewidth=1.25)
            nmse_ax.plot(x, report["mean_raw_nmse_db"], linestyle=line, color="#333333",
                         marker="o", linewidth=1.75, markersize=5.5,
                         markerfacecolor="#333333" if filled else "white", markeredgewidth=1.25)
        ber_ax.set(title=f"{channel} · BER", ylabel="BER", yscale="log", ylim=(1.3e-5, 0.6))
        nmse_ax.set(title=f"{channel} · CSI NMSE", xlabel="SNR (dB)", ylabel="Mean full-matrix CSI NMSE (dB)",
                    xticks=report["snr_db"])
        nmse_ax.text(.035, .055, "Reference CSI: NMSE = 0 (−∞ dB)", transform=nmse_ax.transAxes,
                     fontsize=10, bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=.95))
        for ax in (ber_ax, nmse_ax):
            ax.grid(alpha=.22, which="both")
            ax.set_xlim(-1, 31)
    fig.text(.5, .03, f"Zero observed errors are shown at 0.5 / {bits:,}; overlapping curves may coincide.",
             ha="center", fontsize=10, color="#414141")
    fig.savefig(output / "cp_comparison.png", dpi=180)
    fig.savefig(output / "cp_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp256", type=Path, required=True, help="CP=256 experiment output directory")
    parser.add_argument("--cp384", type=Path, required=True, help="CP=384 Q24/Q32 experiment output directory")
    parser.add_argument("--output", type=Path, required=True, help="comparison output directory")
    args = parser.parse_args()
    old_meta, old_reports = read_source(args.cp384, 384)
    new_meta, new_reports = read_source(args.cp256, 256)
    compare_sources(old_meta, new_meta)
    reports = {256: new_reports, 384: old_reports}
    for channel in CHANNELS:
        old, new = old_reports[channel], new_reports[channel]
        if old["snr_db"] != new["snr_db"] or old["bits_per_snr"] != new["bits_per_snr"]:
            raise ValueError(f"{channel}: SNR grid or bit count differs")
    if len({report["bits_per_snr"] for group in reports.values() for report in group.values()}) != 1:
        raise ValueError("Channels must have the same bit count for this figure")
    args.output.mkdir(parents=True, exist_ok=True)
    write_summary(args.output, reports)
    draw(args.output, reports)
    manifest = {
        "schema": "four-data-cp-retrospective-comparison-v1",
        "script_sha256": sha256(Path(__file__)),
        "sources": {str(cp): {"path": str(path.resolve()),
                              "manifest_sha256": sha256(path / "manifest.json"),
                              "metrics_sha256": sha256(path / "metrics.json"),
                              "git_head": metadata["git_head"]}
                    for cp, path, metadata in ((256, args.cp256, new_meta), (384, args.cp384, old_meta))},
        "comparison": "separate physical frames, same record IDs and start time; noise samples are not paired",
        "q": 24,
        "csi_compensation": False,
        "zero_error_plot_value": "0.5 / bits_per_snr; CSV preserves zero errors",
        "reference_csi_nmse": "0 linear (-infinity dB)",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
