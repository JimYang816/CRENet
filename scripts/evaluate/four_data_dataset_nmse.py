"""Plot the accepted formal DDDD datasets without reopening their HDF5 files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(
        "outputs/experiment/dddd_delay192_assisted_20261124"))
    args = parser.parse_args()
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in available]
    plt.rcParams.update({
        "font.family": fonts + ["DejaVu Sans"], "font.size": 11,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.975, top=.77, bottom=.22, wspace=.10)
    fig.suptitle("正式数据集：平均 NMSE–SNR", y=.965, fontsize=19, fontweight="bold")
    fig.text(.5, .89, "参考矩阵辅助补偿 CSI  ·  192 个估计时延抽头  ·  每帧 4 个数据 OFDM 符号",
             ha="center", fontsize=11, color="#536171")
    styles = (
        ("train", "训练集", "#2563A6", "-", "o", 10),
        ("validation", "验证集", "#D97706", "--", "s", 7),
        ("test", "测试集", "#27805A", ":", "x", 5),
    )
    provenance = {"input_view": "reference_assisted", "channels": {}}
    for ax, channel in zip(axes, ("NOF1", "NCS1")):
        source = args.run_dir / channel.lower() / "acceptance.json"
        content = source.read_bytes()
        report = json.loads(content)
        if not report["complete"] or report["default_input_view"] != "reference_assisted":
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
            ax.plot(snr, values, label=f"{label}（{count} 帧/SNR）", color=color,
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
        endpoints = np.asarray(curves["test"])[[0, -1]]
        for x, y, offset, align in zip(snr[[0, -1]], endpoints, ((9, 14), (-9, 14)), ("left", "right")):
            ax.annotate(f"{y:.2f} dB", (x, y), xytext=offset, textcoords="offset points",
                        ha=align, fontsize=11, fontweight="bold", color="#233348")
        provenance["channels"][channel] = {
            "acceptance_file": str(source.resolve()),
            "acceptance_sha256": hashlib.sha256(content).hexdigest(),
            "snr_db": snr.tolist(), "mean_assisted_nmse_db": curves,
        }
    axes[0].set_ylabel("平均 NMSE (dB)", labelpad=10)
    fig.text(.5, .105, "训练、验证、测试分别按相同目标校准，三条曲线重合。",
             ha="center", fontsize=10.5, color="#435266")
    fig.text(.5, .055, "统计口径：先对各帧、各数据符号的线性 NMSE 求均值，再转换为 dB；使用全量验收统计。",
             ha="center", fontsize=9.5, color="#667387")
    for extension in ("png", "pdf", "svg"):
        path = args.run_dir / f"nmse_snr_combined.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path.resolve())
    plt.close(fig)
    (args.run_dir / "nmse_snr_combined.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
