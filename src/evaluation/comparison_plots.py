"""Complete comparison reports and data-free plotting."""
from __future__ import annotations

import html
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.experiment import write_json, file_sha256
from src.evaluation.fingerprints import source_fingerprint
from src.evaluation.registry import AVAILABLE_ALGORITHMS, CURVES, IDENTIFIERS, normalize_report_labels
from src.evaluation.result_cache import FileLock


def plot_ber(report, path):
    report = normalize_report_labels(report)
    styles = dict(zip(CURVES, (("#667085", "o"), ("#2477b3", "s"))))
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        zero_seen = False
        for name, values in report["curves"].items():
            color, marker = styles[name]
            ber, bits = np.array(values["ber"]), np.array(values["total_bits"])
            line = dict(markersize=6)
            ax.semilogy(report["snr_db"], np.where(ber > 0, ber, np.nan), color=color,
                        marker=marker, linewidth=1.8, **line,
                        label=name)
            zero = ber == 0
            if np.any(zero):
                zero_seen = True
                ax.scatter(np.array(report["snr_db"])[zero], .5/bits[zero], marker="v", facecolors="none", edgecolors=color)
        ax.set(xlabel="SNR (dB)", ylabel="Uncoded BER", title=f'{report["channel"]} | Test set | QPSK')
        ax.set_xticks(report["snr_db"])
        ax.grid(True, which="major", alpha=.28)
        ax.grid(True, which="minor", alpha=.12)
        ax.legend(loc="best", framealpha=.95)
        view = report.get("input_view", "raw")
        caption = f'{report["frame_count"]:,} frames / SNR  |  {report["bits_per_snr"]:,} bits / SNR  |  CSI view: {view}'
        if report.get("smoke"):
            caption = "SMOKE TEST | "+caption
        if zero_seen:
            caption += "\nOpen triangles: zero observed errors, plotted at 0.5 / tested bits."
        fig.text(.5, .025, caption, ha="center", fontsize=8.5, color="#475467")
        fig.tight_layout(rect=(0, .07 if zero_seen else .045, 1, 1))
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)


def plot_nmse(report, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    ax.plot(report["snr_db"], report["nmse"]["mean_nmse_db"], "o-", color="#2477b3", label="Test CSI")
    ax.set(xlabel="SNR (dB)", ylabel="Mean NMSE (dB)",
           title=f'{report["channel"]} | Test set | mean(linear per-D NMSE)')
    ax.set_xticks(report["snr_db"])
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def validate_report(report):
    report = normalize_report_labels(report)
    names = set(report.get("curves", {}))
    if not report.get("complete") or names != set(CURVES):
        raise ValueError("Plotting requires a complete comparison report with both LMMSE baselines")
    snr = np.asarray(report["snr_db"], dtype=float)
    if snr.ndim != 1 or not len(snr) or not np.all(np.isfinite(snr)):
        raise ValueError("Invalid SNR values in comparison report")
    for name in report["curves"]:
        curve = report["curves"][name]
        for field in ("ber", "errors", "total_bits"):
            values = np.asarray(curve[field])
            if values.shape != snr.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid report curve: {name}/{field}")
        errors, bits = np.asarray(curve["errors"]), np.asarray(curve["total_bits"])
        if np.any(errors < 0) or np.any(bits <= 0) or np.any(errors > bits) or not np.array_equal(np.asarray(curve["ber"]), errors/bits):
            raise ValueError(f"Invalid BER counts: {name}")
    if "nmse" not in report:
        raise ValueError("Report has no test NMSE statistics; run the new evaluator once before offline plotting")
    values = np.asarray(report["nmse"]["mean_nmse_db"])
    if report["nmse"].get("split") != "test" or values.shape != snr.shape or not np.all(np.isfinite(values)):
        raise ValueError("Invalid test NMSE values in report")


def publish_report(run, report, *, redraw_nmse=False):
    """Stage both figures and JSON; roll back replacements on publication errors."""
    normalized = normalize_report_labels(report)
    report.clear()
    report.update(normalized)
    validate_report(report)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".publish.lock"), TemporaryDirectory(prefix=".comparison-", dir=run) as temporary:
        stage = Path(temporary)
        plot_source = source_fingerprint((__name__+":plot_ber", __name__+":plot_nmse"))
        plot_ber(report, stage/"ber_snr.png")
        previous = None
        try:
            previous = json.loads((run/"metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        if (not redraw_nmse and previous is not None and (run/"nmse_snr.png").is_file()
                and previous.get("nmse", {}).get("mean_nmse_db") == report["nmse"]["mean_nmse_db"]
                and previous.get("nmse", {}).get("provenance", {}).get("cache_key") == report["nmse"]["provenance"]["cache_key"]
                and previous.get("snr_db") == report["snr_db"]
                and previous.get("channel") == report["channel"]
                and previous.get("plot_source", {}).get(__name__+":plot_nmse") == plot_source[__name__+":plot_nmse"]
                and previous.get("figure_sha256", {}).get("nmse_snr.png") == file_sha256(run/"nmse_snr.png")):
            shutil.copyfile(run/"nmse_snr.png", stage/"nmse_snr.png")
        else:
            plot_nmse(report, stage/"nmse_snr.png")
        report["plot_source"] = plot_source
        report["figure_sha256"] = {name: file_sha256(stage/name) for name in ("ber_snr.png", "nmse_snr.png")}
        write_json(stage/"metrics.json", report)
        names = ("ber_snr.png", "nmse_snr.png", "metrics.json")
        backups = stage/"previous"
        backups.mkdir()
        for name in names:
            if (run/name).exists():
                shutil.copyfile(run/name, backups/name)
        replaced = []
        try:
            for name in names:
                (stage/name).replace(run/name)
                replaced.append(name)
        except BaseException:
            for name in reversed(replaced):
                if (backups/name).exists():
                    (backups/name).replace(run/name)
                else:
                    (run/name).unlink(missing_ok=True)
            raise


def _write_legacy_html(run, report):
    channel = html.escape(report["channel"])
    headings = "".join(f"<th>{s:g} dB</th>" for s in report["snr_db"])
    ber_rows = "".join(
        "<tr><th>"+html.escape(label)+"</th>"+"".join(f"<td>{value:.6g}</td>" for value in curve["ber"])+"</tr>"
        for label, curve in report["curves"].items()
    )
    nmse_cells = "".join(f"<td>{value:.3f}</td>" for value in report["nmse"]["mean_nmse_db"])
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{channel} 均衡算法对比</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dc;padding:8px}}</style>
<h1>{channel} LMMSE 评测</h1><p>每个 SNR 使用 {report['bits_per_snr']:,} 个测试比特。数据和评分来源见 metrics.json。</p>
<table><tr><th>算法 BER</th>{headings}</tr>{ber_rows}</table><img src="ber_snr.png" alt="BER-SNR">
<table><tr><th>输入 CSI 平均 NMSE (dB)</th>{headings}</tr><tr><th>输入 CSI</th>{nmse_cells}</tr></table>
<img src="nmse_snr.png" alt="NMSE-SNR">
<p>零误码以 0.5 / 测试比特数处的空心三角显示。</p></html>'''
    (run/"report.html").write_text(body, encoding="utf-8")


def replot(run, *, output=None, expected_channel=None, expected_view=None, required_algorithms=()):
    """Redraw the LMMSE curves from saved values, optionally in a new directory."""
    report = json.loads((Path(run)/"metrics.json").read_text(encoding="utf-8"))
    if expected_channel is not None and report.get("channel") != expected_channel:
        raise ValueError(f"Saved report does not match channel {expected_channel}")
    if expected_view is not None and report.get("input_view") != expected_view:
        raise ValueError(f"Saved report does not match CSI view {expected_view}")
    report = normalize_report_labels(report)
    retired = {curve.get("identifier") for label, curve in report["curves"].items()
               if label not in CURVES}
    report["curves"] = {label: report["curves"][label] for label in CURVES if label in report["curves"]}
    if "parameters" in report:
        report["parameters"] = {key: value for key, value in report["parameters"].items() if key in IDENTIFIERS}
    for identifier in retired:
        if identifier:
            report.pop(identifier, None)
    for field in ("checkpoint_paths", "checkpoints"):
        report.pop(field, None)
    if "nmse" in report:
        report["nmse"] = {key: value for key, value in report["nmse"].items()
                          if not key.endswith("_mean_nmse_db")}
        if report.get("schema") == "watermark-unified-comparison-v1":
            report["nmse"].pop("semantics", None)
    if "comparison_algorithms" in report:
        report["comparison_algorithms"] = list(report["curves"])
    required = {a.label for a in AVAILABLE_ALGORITHMS if a.identifier in required_algorithms}
    missing = required-report["curves"].keys()
    if missing:
        raise ValueError(f"Saved report lacks configured curves {sorted(missing)}. Run evaluation before plotting.")
    run = Path(run if output is None else output)
    publish_report(run, report, redraw_nmse=True)
    if report.get("schema") == "watermark-unified-comparison-v1":
        _write_legacy_html(run, report)
    return report
