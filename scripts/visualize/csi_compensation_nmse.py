"""Extract accepted test-set CSI statistics and plot compensation comparisons.

The extract command only reads the compact HDF5 files and their acceptance
manifests. It writes JSON to standard output, so the small result can be
transferred without moving the full datasets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SNR_DB = np.arange(0, 31, 5)
CHANNEL_ENDPOINTS = {"NOF1": (-12.70, -18.30), "NCS1": (-6.20, -10.67)}
TEST_RECORDS = list(range(11, 48, 4))


def extract_channel(directory: Path, channel: str) -> dict:
    """Read paired raw and compensated test NMSE from accepted compact data."""
    import h5py

    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("protocol") != "watermark-pddp-compact-v2"
            or manifest.get("channel") != channel or not manifest.get("complete")
            or manifest.get("smoke") or not manifest.get("reference_assisted_construction")):
        raise ValueError(f"{channel}: invalid or unaccepted dataset manifest")

    with h5py.File(directory / "test.h5", "r") as data:
        if (data.attrs.get("schema") != manifest["protocol"]
                or data.attrs.get("channel") != channel or data.attrs.get("split") != "test"
                or not data.attrs.get("complete") or not data.attrs.get("compensated")
                or not data.attrs.get("reference_assisted_construction")):
            raise ValueError(f"{channel}: invalid or incomplete test dataset")
        snr = np.asarray(data["snr_db"][:])
        raw = np.asarray(data["samples/raw_nmse_linear"][:], dtype=float)
        saved_raw_mean = np.asarray(data["compensation/raw_mean_nmse_linear"][:], dtype=float)
        alpha = np.asarray(data["compensation/alpha"][:], dtype=float)
        records = np.unique(data["windows/record"][:]).tolist()
        if (not np.array_equal(snr, SNR_DB) or records != TEST_RECORDS
                or raw.shape != (len(SNR_DB), 600, 2)
                or saved_raw_mean.shape != SNR_DB.shape or alpha.shape != SNR_DB.shape
                or not np.all(np.isfinite(raw)) or np.any(raw <= 0)
                or not np.all(np.isfinite(alpha)) or np.any(alpha <= 0)):
            raise ValueError(f"{channel}: invalid SNR grid, test split, or NMSE arrays")
        raw_mean = raw.mean(axis=(1, 2))
        if not np.allclose(raw_mean, saved_raw_mean, rtol=1e-10, atol=1e-14):
            raise ValueError(f"{channel}: saved raw NMSE does not match per-block values")

        # Scaling H_raw - H_ref by alpha scales each linear NMSE by alpha**2.
        compensated_db = 10 * np.log10(raw_mean * alpha**2)
        accepted = np.asarray(manifest["splits"]["test"]["mean_nmse_db"], dtype=float)
        start, stop = CHANNEL_ENDPOINTS[channel]
        target = start + (stop - start) * (1 - np.exp(-snr / 10)) / (1 - np.exp(-3))
        if (accepted.shape != SNR_DB.shape
                or not np.allclose(compensated_db, accepted, rtol=0, atol=2e-8)
                or not np.allclose(compensated_db, target, rtol=0, atol=2e-8)
                or not np.allclose(alpha, manifest["splits"]["test"]["alpha"], rtol=0, atol=1e-12)
                or manifest["splits"]["test"]["checked_frames"] != raw.shape[1]):
            raise ValueError(f"{channel}: compensation differs from the accepted test report")

    raw_db = 10 * np.log10(raw_mean)
    return {
        "dataset_directory": str(directory),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "test_h5_sha256_recorded_in_manifest": manifest["files"]["test.h5"].get("sha256"),
        "records": records,
        "frames_per_snr": raw.shape[1],
        "d_blocks_per_snr": raw.shape[1] * raw.shape[2],
        "raw_mean_nmse_linear": raw_mean.tolist(),
        "compensation_alpha": alpha.tolist(),
        "raw_mean_nmse_db": raw_db.tolist(),
        "compensated_mean_nmse_db": compensated_db.tolist(),
        "reduction_db": (raw_db - compensated_db).tolist(),
        "max_manifest_difference_db": float(np.max(np.abs(compensated_db - accepted))),
    }


def extract(nof1: Path, ncs1: Path) -> dict:
    return {
        "split": "test",
        "snr_db": SNR_DB.tolist(),
        "metric": "10*log10(mean(per-D full-matrix linear NMSE))",
        "compensation": "H_comp = H_ref + alpha * (H_raw - H_ref)",
        "reference_assisted": True,
        "channels": {"NOF1": extract_channel(nof1, "NOF1"),
                     "NCS1": extract_channel(ncs1, "NCS1")},
    }


def plot(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if report.get("split") != "test" or report.get("snr_db") != SNR_DB.tolist():
        raise ValueError("Plot requires the accepted test split and SNR 0:5:30 dB")
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
        fig.subplots_adjust(left=.085, right=.98, bottom=.21, top=.77, wspace=.12)
        fig.suptitle("CSI NMSE before and after compensation", x=.085, y=.96,
                     ha="left", fontsize=16, fontweight="semibold")
        fig.text(.085, .89, "Test split  |  600 frames and 1,200 D blocks per SNR",
                 color="#526276", fontsize=10)
        for ax, channel in zip(axes, ("NOF1", "NCS1")):
            result = report["channels"][channel]
            for field, label, color, marker, style in (
                ("raw_mean_nmse_db", "Before compensation", "#B45A2A", "o", "--"),
                ("compensated_mean_nmse_db", "After compensation", "#2065A0", "s", "-"),
            ):
                values = np.asarray(result[field], dtype=float)
                if values.shape != SNR_DB.shape or not np.isfinite(values).all():
                    raise ValueError(f"{channel}: invalid {field}")
                ax.plot(SNR_DB, values, linestyle=style, marker=marker, color=color,
                        linewidth=2, markersize=5.5, markerfacecolor="white" if style == "--" else color,
                        markeredgewidth=1.2, label=label)
            ax.set_title(channel, loc="left", fontsize=12, fontweight="bold", pad=10)
            ax.set(xlabel="SNR (dB)", xticks=SNR_DB, xlim=(-1, 31), ylim=(-20, 0),
                   yticks=np.arange(-20, 1, 5))
            ax.grid(True, color="#D8DEE6", linewidth=.65)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#A8B1BC")
            legend_location = "upper right" if channel == "NOF1" else "lower left"
            ax.legend(loc=legend_location, frameon=True, facecolor="white",
                      edgecolor="#D8DEE6", framealpha=.95, fontsize=9)
        axes[0].set_ylabel("Mean full-matrix NMSE (dB)")
        fig.text(.085, .075, "Compensation uses reference CSI: Hcomp = Href + alpha (Hraw - Href).",
                 fontsize=9, color="#526276")
        fig.savefig(output / "csi_compensation_nmse_snr.png", dpi=220, facecolor="white")
        fig.savefig(output / "csi_compensation_nmse_snr.pdf", facecolor="white")
        plt.close(fig)

    with (output / "csi_compensation_nmse_snr.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["channel", "snr_db", "raw_mean_nmse_db", "compensated_mean_nmse_db", "reduction_db"])
        for channel, result in report["channels"].items():
            for row in zip(SNR_DB, result["raw_mean_nmse_db"],
                           result["compensated_mean_nmse_db"], result["reduction_db"]):
                writer.writerow([channel, *row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("extract", help="read statistics from accepted datasets")
    source.add_argument("--nof1", type=Path, required=True)
    source.add_argument("--ncs1", type=Path, required=True)
    figure = subparsers.add_parser("plot", help="plot extracted statistics")
    figure.add_argument("--input", type=Path, required=True)
    figure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        print(json.dumps(extract(args.nof1, args.ncs1), indent=2, ensure_ascii=True))
    else:
        plot(json.loads(args.input.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
