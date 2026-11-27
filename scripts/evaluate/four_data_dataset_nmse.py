"""Plot the accepted formal DDDD datasets without reopening their HDF5 files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def plot_overlay(output, provenance, reports):
    """Compare the current measured test statistics on common axes."""
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    colors = {"NOF1": "#2563A6", "NCS1": "#D97706"}
    for channel, marker in (("NOF1", "o"), ("NCS1", "s")):
        values = provenance["channels"][channel]
        snr = np.asarray(values["snr_db"])
        nmse = np.asarray(values["mean_assisted_nmse_db"]["test"])
        ax.plot(snr, nmse, marker=marker, color=colors[channel],
                linewidth=2, markersize=6, label=channel)
    ax.set(title="Compensated CSI: NMSE–SNR", xlabel="SNR (dB)",
           ylabel="Mean NMSE (dB)", xticks=snr, xlim=(-1, 31), ylim=(-17, -4))
    ax.grid(alpha=.3)
    ax.legend(loc="upper right")
    fig.get_layout_engine().set(rect=(0, .065, 1, 1))
    counts = {values["test_frames_per_snr"] for values in provenance["channels"].values()}
    frame_note = f"{next(iter(counts))} frames/SNR" if len(counts) == 1 else "full test splits"
    fig.text(.5, .025,
             f"Full-reference-assisted CSI | Test split: {frame_note}, 4 data symbols/frame",
             ha="center", fontsize=9, color="#536171")
    for extension in ("png", "pdf", "svg"):
        path = output / f"nmse_snr_overlay.{extension}"
        fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
        print(path.resolve())
    plt.close(fig)
    with (output / "nmse_snr_overlay.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["snr_db", "NOF1_nmse_db", "NCS1_nmse_db"])
        curves = provenance["channels"]
        np.testing.assert_array_equal(curves["NOF1"]["snr_db"], curves["NCS1"]["snr_db"])
        for index, value in enumerate(snr):
            writer.writerow([f"{value:g}"] + [f"{curves[ch]['mean_assisted_nmse_db']['test'][index]:.2f}"
                                              for ch in ("NOF1", "NCS1")])
    if len(reports) == 2:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for channel, report in reports.items():
            np.testing.assert_allclose(
                report["nmse"]["mean_nmse_db"],
                provenance["channels"][channel]["mean_assisted_nmse_db"]["test"],
                rtol=0, atol=1e-12,
            )
            curve = next(value for value in report["curves"].values()
                         if value["identifier"] == "banded_lmmse")
            ax.semilogy(report["snr_db"], curve["ber"], "o-", color=colors[channel], label=channel)
        ax.set(title="Banded LMMSE (B=16): BER–SNR", xlabel="SNR (dB)", ylabel="BER", xticks=snr)
        ax.grid(alpha=.3, which="both")
        ax.legend()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"ber_snr.{extension}", dpi=220, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Use one historical run for both channels")
    parser.add_argument("--nof1-run-dir", type=Path)
    parser.add_argument("--ncs1-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dirs = {
        "NOF1": args.nof1_run_dir or args.run_dir or Path(
            "outputs/experiment/dddd_assisted_b16_20261127"),
        "NCS1": args.ncs1_run_dir or args.run_dir or Path(
            "outputs/experiment/dddd_assisted_b16_ncs1_892_20261127"),
    }
    output = args.output_dir or args.run_dir or Path("outputs/smoke/compensated_csi_nmse_20261127")
    output.mkdir(parents=True, exist_ok=True)
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in available]
    plt.rcParams.update({
        "font.family": fonts + ["DejaVu Sans"], "font.size": 11,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.975, top=.77, bottom=.22, wspace=.10)
    title = "正式数据集：平均 NMSE–SNR" if fonts else "Formal datasets: mean NMSE–SNR"
    subtitle = ("参考矩阵辅助补偿 CSI  ·  192 个估计时延抽头  ·  每帧 4 个数据 OFDM 符号" if fonts else
                "Full-reference-assisted CSI | 192 estimated delays | 4 data symbols/frame")
    fig.suptitle(title, y=.965, fontsize=19, fontweight="bold")
    fig.text(.5, .89, subtitle,
             ha="center", fontsize=11, color="#536171")
    styles = (
        ("train", "训练集" if fonts else "Train", "#2563A6", "-", "o", 10),
        ("validation", "验证集" if fonts else "Validation", "#D97706", "--", "s", 7),
        ("test", "测试集" if fonts else "Test", "#27805A", ":", "x", 5),
    )
    provenance = {"input_view": "reference_assisted", "channels": {}}
    reports = {}
    for ax, channel in zip(axes, ("NOF1", "NCS1")):
        source = run_dirs[channel] / channel.lower() / "acceptance.json"
        content = source.read_bytes()
        report = json.loads(content)
        if (not report["complete"] or report["channel"] != channel
                or report["default_input_view"] != "reference_assisted"):
            raise ValueError(f"Expected an accepted reference-assisted dataset: {source}")
        snr = np.asarray(report["snr_db"], dtype=float)
        curves = {}
        for split, label, color, linestyle, marker, size in styles:
            split_report = report["splits"][split]
            values = np.asarray(split_report["mean_assisted_nmse_db"], dtype=float)
            if values.shape != snr.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid measured NMSE values: {channel}/{split}")
            curves[split] = values.tolist()
            count = split_report["frames_per_snr"]
            curve_label = f"{label}（{count} 帧/SNR）" if fonts else f"{label} ({count} frames/SNR)"
            ax.plot(snr, values, label=curve_label, color=color,
                    linestyle=linestyle, linewidth=1.8, marker=marker, markersize=size,
                    markerfacecolor="none", markeredgewidth=1.5)
        for values in curves.values():
            if not np.allclose(values, curves["test"], rtol=0, atol=1e-12):
                raise ValueError("The figure's overlap note no longer applies")
        ax.set_title(channel, fontsize=15, fontweight="bold", pad=14)
        ax.set_xlabel("SNR (dB)", labelpad=8)
        ax.set_xlim(-1, 31)
        ax.set_ylim(-17.3, -4.1)
        ax.set_xticks(snr)
        ax.set_yticks(np.arange(-16, -3, 2))
        ax.grid(True, color="#DDE3E9", linewidth=.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=.96,
                  edgecolor="#DDE3E9")
        provenance["channels"][channel] = {
            "acceptance_file": str(source.resolve()),
            "acceptance_sha256": hashlib.sha256(content).hexdigest(),
            "snr_db": snr.tolist(), "mean_assisted_nmse_db": curves,
            "test_frames_per_snr": report["splits"]["test"]["frames_per_snr"],
        }
        metrics = run_dirs[channel] / "test" / channel.lower() / "metrics.json"
        if metrics.exists():
            measured = json.loads(metrics.read_bytes())
            if not measured["complete"] or measured["input_view"] != "reference_assisted":
                raise ValueError(f"Expected complete reference-assisted test metrics: {metrics}")
            reports[channel] = measured
            provenance["channels"][channel].update(
                test_metrics_file=str(metrics.resolve()),
                test_metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(),
            )
    axes[0].set_ylabel("平均 NMSE (dB)" if fonts else "Mean NMSE (dB)", labelpad=10)
    overlap = ("训练、验证、测试分别按相同目标校准，三条曲线重合。" if fonts else
               "Train, validation and test curves overlap after independent calibration.")
    definition = ("统计口径：先对各帧、各数据符号的线性 NMSE 求均值，再转换为 dB；使用全量验收统计。" if fonts else
                  "Average linear NMSE over frames and symbols, then convert to dB; complete acceptance statistics.")
    fig.text(.5, .105, overlap,
             ha="center", fontsize=10.5, color="#435266")
    fig.text(.5, .055, definition,
             ha="center", fontsize=9.5, color="#667387")
    for extension in ("png", "pdf", "svg"):
        path = output / f"nmse_snr_combined.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path.resolve())
    plt.close(fig)
    plot_overlay(output, provenance, reports)
    (output / "nmse_snr_combined.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
