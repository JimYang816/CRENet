"""Plot NOF1 Q23 and NCS1 Q32 from the saved uncompensated-CSI test results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from data_generation.src.config import ROOT
from src.common.experiment import file_sha256, write_json


ORDERS = {"NOF1": 23, "NCS1": 32}
SNR_DB = [0, 5, 10, 15, 20, 25, 30]
COLORS = {"NOF1": "#24649a", "NCS1": "#c4513e"}
MARKERS = {"NOF1": "o", "NCS1": "s"}


def load_results(source):
    """Validate raw view and recompute the saved aggregates from per-frame records."""
    data = json.loads(source.read_text(encoding="utf-8"))
    meta = data["metadata"]
    if (not data.get("complete") or meta.get("input_view") != "raw"
            or meta.get("compensation") is not False or meta.get("snr_db") != SNR_DB
            or meta.get("bandwidth_hz") != 12.0 or meta.get("length") != 176):
        raise ValueError("Expected the complete raw-CSI 4D experiment with 12 Hz / L176")
    result = {}
    frames_by_channel = {}
    for channel, order in ORDERS.items():
        matches = [r for r in data["test"][channel] if r["q"] == order]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {channel} Q={order} test result")
        row = matches[0]
        if (row["input_view"] != "raw" or row["compensation"] is not False
                or row["split"] != "test"):
            raise ValueError("All plotted results must use unscaled raw test CSI")
        frames = row["frames"]
        errors = np.asarray([f["errors"] for f in frames], dtype=np.int64)
        nmse = np.asarray([f["nmse_linear"] for f in frames], dtype=np.float64)
        if (len(frames) != 10 or errors.shape != (10, len(SNR_DB), 4) or nmse.shape != errors.shape
                or np.any(errors < 0) or np.any(errors > 1024)
                or not np.all(np.isfinite(nmse)) or np.any(nmse <= 0)):
            raise ValueError("Invalid test-frame counts, bit errors or raw CSI NMSE")
        bit_count = len(frames)*4*512*2
        if row["bits_per_snr"] != bit_count:
            raise ValueError("Test bit count does not match the frames")
        error_counts = errors.sum(axis=(0, 2))
        linear = nmse.mean(axis=(0, 2))
        db = 10*np.log10(linear)
        np.testing.assert_array_equal(error_counts, row["bit_errors"])
        np.testing.assert_allclose(error_counts/bit_count, row["ber"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(linear, row["mean_nmse_linear"], rtol=1e-13, atol=0)
        np.testing.assert_allclose(db, row["mean_nmse_db"], rtol=0, atol=1e-11)
        frames_by_channel[channel] = [(f["record"], f["start_seconds"]) for f in frames]
        result[channel] = dict(q=order, frame_count=len(frames), bits_per_snr=bit_count,
                               bit_errors=error_counts.tolist(), ber=(error_counts/bit_count).tolist(),
                               mean_nmse_linear=linear.tolist(), mean_nmse_db=db.tolist())
    if frames_by_channel["NOF1"] != frames_by_channel["NCS1"]:
        raise ValueError("The two channels use different test-window identities")
    return dict(input_view="raw", compensation=False, snr_db=SNR_DB, channels=result,
                frame="D1-D2-D3-D4; 512 data and 256 comb pilots per D", bandwidth_hz=12.0,
                delay_length=176, equalizer="full-matrix LMMSE, 1024 receive rows and 512 data columns",
                nmse_aggregation="arithmetic mean of per-D linear NMSE, then 10 log10",
                test_windows=frames_by_channel["NOF1"], source_path=str(source.resolve()),
                source_sha256=file_sha256(source), source_git_head=meta["git_head"])


def draw(ax, data, metric):
    for channel, q in ORDERS.items():
        row = data["channels"][channel]
        values = row["ber"] if metric == "ber" else row["mean_nmse_db"]
        ax.plot(SNR_DB, values, color=COLORS[channel], marker=MARKERS[channel],
                linewidth=2.1, markersize=6, label=f"{channel}  (Q={q})")
    ax.set_xticks(SNR_DB)
    ax.set_xlabel("SNR (dB)")
    if metric == "ber":
        ax.set_title("BER–SNR")
        ax.set_ylabel("Bit error rate")
        ax.set_yscale("log")
        ax.set_ylim(.025, .4)
        ax.set_yticks([.03, .05, .1, .2, .3])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.set_title("Mean CSI NMSE–SNR")
        ax.set_ylabel("Mean NMSE (dB)")
        ax.set_ylim(-9.6, -2.2)
        ax.set_yticks([-9, -8, -7, -6, -5, -4, -3])
    ax.grid(True, which="major", alpha=.25)
    ax.grid(True, which="minor", alpha=.1)
    ax.legend(loc="best", frameon=True)


def render(data, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"metrics.json", data)
    with (output/"metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["snr_db", "nof1_q23_bit_errors", "nof1_bits", "nof1_q23_ber",
                         "nof1_q23_mean_nmse_db", "ncs1_q32_bit_errors", "ncs1_bits",
                         "ncs1_q32_ber", "ncs1_q32_mean_nmse_db"])
        a, b = data["channels"]["NOF1"], data["channels"]["NCS1"]
        for i, snr in enumerate(SNR_DB):
            writer.writerow([snr, a["bit_errors"][i], a["bits_per_snr"], a["ber"][i], a["mean_nmse_db"][i],
                             b["bit_errors"][i], b["bits_per_snr"], b["ber"][i], b["mean_nmse_db"][i]])
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
                         "legend.fontsize": 10, "savefig.facecolor": "white", "pdf.fonttype": 42})
    for metric in ("ber", "nmse"):
        fig, ax = plt.subplots(figsize=(6.5, 4.6), layout="constrained")
        draw(ax, data, metric)
        fig.suptitle("Uncompensated CSI · Full-matrix LMMSE", fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{metric}_snr.{ext}", dpi=220)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), layout="constrained")
    draw(axes[0], data, "ber")
    draw(axes[1], data, "nmse")
    fig.suptitle("NOF1 / NCS1 · Uncompensated CSI · Full-matrix LMMSE", fontsize=13)
    fig.supxlabel("4D frame · 12 Hz · L=176 · 10 test frames per channel · 40,960 bits per SNR", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"comparison.{ext}", dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT/"outputs/experiment/four_data_256p_raw_q16_32_20261123/metrics.json")
    p.add_argument("--output", type=Path, default=ROOT/"outputs/experiment/nof1_q23_ncs1_q32_raw_comparison_20261123")
    a = p.parse_args()
    output = a.output.resolve()
    if (ROOT/"outputs/experiment").resolve() not in output.parents:
        p.error("Output must remain under outputs/experiment")
    data = load_results(a.source.resolve())
    render(data, output)
    print(f"Wrote raw-CSI comparison to {output}")


if __name__ == "__main__":
    main()
