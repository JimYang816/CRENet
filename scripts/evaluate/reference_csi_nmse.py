"""Replot approximate Estimate CSI readings from the user-supplied reference image."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path,
                        default=Path("configs/evaluation/reference_csi_nmse.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/smoke/reference_csi_nmse_20261127"))
    parser.add_argument("--source-image", type=Path)
    args = parser.parse_args()
    content = args.points.read_bytes()
    points = json.loads(content)
    if points["kind"] != "approximate_points_digitized_from_reference_image":
        raise ValueError("This figure requires explicitly identified reference readings")
    snr = np.asarray(points["snr_db"], dtype=float)
    np.testing.assert_array_equal(snr, np.arange(0, 31, 5))
    for channel in ("NOF1", "NCS1"):
        values = np.asarray(points["nmse_db"][channel], dtype=float)
        if values.shape != snr.shape or not np.isfinite(values).all() or np.any(np.diff(values) >= 0):
            raise ValueError(f"Invalid reference readings for {channel}")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.source_image is not None:
        if hashlib.sha256(args.source_image.read_bytes()).hexdigest() != points["source_image_sha256"]:
            raise ValueError("Source image does not match the digitized reference")
        shutil.copy2(args.source_image, output / "source_reference.png")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.2, 4.65), layout="constrained")
    for channel, color, marker in (("NOF1", "#1F78B5", "o"), ("NCS1", "#C9144A", "^")):
        ax.plot(snr, points["nmse_db"][channel], color=color, marker=marker,
                linestyle=(0, (8, 5)), linewidth=1.4, markersize=6.5,
                markerfacecolor="white", markeredgewidth=1.4,
                label=f"{channel} - Estimate CSI")
    ax.set(xlabel="SNR (dB)", ylabel="NMSE (dB)",
           xlim=(-1.5, 31.5), ylim=(-22, 0), xticks=snr, yticks=np.arange(-22, 1, 2))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which="major", color="#A5A5A5", linewidth=.65)
    ax.grid(which="minor", color="#BFBFBF", linestyle=(0, (1, 4)), linewidth=.65)
    ax.tick_params(which="both", direction="out", top=True, right=True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=2, fontsize=9.5, framealpha=1,
              fancybox=False, edgecolor="#444444", columnspacing=1, handlelength=2.4)
    for extension in ("png", "pdf", "svg"):
        path = output / f"nmse_snr_reference.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        print(path.resolve())
    plt.close(fig)
    with (output / "reference_points.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["snr_db", "NOF1_estimate_csi_nmse_db", "NCS1_estimate_csi_nmse_db"])
        for index, value in enumerate(snr):
            writer.writerow([f"{value:g}"] + [f"{points['nmse_db'][ch][index]:.1f}"
                                              for ch in ("NOF1", "NCS1")])
    provenance = dict(points, points_file=str(args.points.resolve()),
                      points_file_sha256=hashlib.sha256(content).hexdigest(),
                      formal_datasets_modified=False, newly_measured=False)
    (output / "reference_replot.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
