"""Reference-assisted target-NMSE LMMSE on the saved CP=80-ms observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_cp80 import CODE as RAW_CODE, channel_settings, load_options
from src.evaluation.four_data_csi_compensation import finish
from src.evaluation.four_data_delay192_compensation import (
    CODE as COMPENSATION_CODE, CHANNELS, error_scales, evaluate, load_target_config, target_curve,
)


CODE = tuple(dict.fromkeys((*RAW_CODE, *COMPENSATION_CODE,
    "src/evaluation/four_data_cp80_compensation.py", "scripts/evaluate/four_data_cp80_compensation.py")))


def validate_protocol(original, reports):
    if (original.get("schema") != "four-data-cp80-lmmse-v1" or original.get("smoke") is not False
            or original.get("cp_ms") != 80 or original.get("input_view") != "raw"
            or original.get("csi_compensation") is not False or set(reports) != set(CHANNELS)):
        raise ValueError("Source must be the completed raw CP80 dual-channel experiment")
    for channel in CHANNELS:
        meta, report = original["channel_manifests"][channel], reports[channel]
        c, study, priors, tests = channel_settings(channel, load_options(), meta["archive_sha256"])
        if (original["options"] != load_options() or original["configs"][channel] != c
                or meta["config"] != c or meta["study"] != study or meta.get("smoke") is not False
                or meta.get("raw_csi") is not True or meta.get("csi_compensation") is not False
                or meta["prior_windows"] != priors or meta["test_windows"] != tests
                or report.get("complete") is not True or report["bits_per_snr"] != 20480
                or report["delay_count"] != 192 or report["coefficient_count"] != 4608
                or report["snr_db"] != list(range(0, 31, 5))
                or [[f["record"], f["start_seconds"]] for f in report["frames"]] != tests):
            raise ValueError(f"{channel}: CP80 settings, fixed delays or held-out windows changed")


def verify_source(source):
    source = Path(source)
    saved = json.loads((source/"metrics.json").read_text(encoding="utf-8"))
    original = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((source/"verification.json").read_text(encoding="utf-8"))
    transfer = json.loads((source/"transfer_manifest.json").read_text(encoding="utf-8"))
    if (saved.get("complete") is not True or saved.get("metadata") != original
            or acceptance.get("complete") is not True or acceptance.get("shared_dpss_basis") is not True
            or transfer.get("complete") is not True):
        raise ValueError("Expected accepted, matching CP80 source manifests and reports")
    validate_protocol(original, saved["channels"])
    physical = {}
    for channel in CHANNELS:
        directory = source/channel.lower()
        meta = original["channel_manifests"][channel]
        detail = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        frozen = json.loads((directory/"frozen_support192.json").read_text(encoding="utf-8"))
        if (json.loads((directory/"manifest.json").read_text(encoding="utf-8")) != meta
                or detail.get("metadata") != meta or detail.get("complete") is not True
                or frozen != meta["study"]["frozen_support"]
                or frozen["indices"] != detail["selection"]["candidates"]["selected192"]["indices"]):
            raise ValueError(f"{channel}: saved channel provenance or fixed support changed")
        for name, expected in meta["code_sha256"].items():
            # Verify the recorded source revision; defaults may evolve afterwards.
            payload = subprocess.check_output(["git", "show", f"{meta['git_head']}:{name}"], cwd=ROOT)
            if expected not in (hashlib.sha256(payload).hexdigest(), hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()):
                raise ValueError(f"Recorded CP80 source code does not match its committed revision: {name}")
        path = directory/"physical.h5"
        identity, checksum = digest(meta), file_sha256(path)
        expected = transfer["physical_data_remain_on_server"][channel.lower()]
        if checksum != expected["sha256"] or path.stat().st_size != expected["bytes"] or identity != expected["identity"]:
            raise ValueError(f"{channel}: physical source differs from the accepted CP80 file hash")
        with h5py.File(path, "r") as h:
            if (not h.attrs.get("complete") or h.attrs.get("identity") != identity
                    or h["basis"].shape != (6016, 24) or h["prior"].shape != (25, 24, 1024)
                    or [[int(g.attrs["record"]), float(g.attrs["start_seconds"])] for g in h["test"].values()] != meta["test_windows"]):
                raise ValueError(f"{channel}: CP80 physical shapes, identity or test windows changed")
        physical[channel] = dict(path=str(path), sha256=checksum, identity=identity,
            manifest_sha256=file_sha256(directory/"manifest.json"), frozen_support=frozen,
            frozen_support_sha256=file_sha256(directory/"frozen_support192.json"))
    return original, saved["channels"], physical


def verify_report(report):
    for algorithm in ("single", "full"):
        counts = np.sum([f[f"compensated_{algorithm}_errors"] for f in report["frames"]], axis=(0, 2))
        curve = report["curves"][f"compensated_{algorithm}"]
        np.testing.assert_array_equal(counts, curve["bit_errors"])
        np.testing.assert_allclose(counts/report["bits_per_snr"], curve["ber"], rtol=0, atol=1e-15)
    mean = np.mean([f["compensated_nmse_linear"] for f in report["frames"]], axis=(0, 2))
    achieved = 10*np.log10(mean)
    np.testing.assert_allclose(achieved, report["mean_compensated_nmse_db"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(achieved, report["target_nmse_db"], rtol=0, atol=1e-9)
    scales = np.asarray(report["error_scales"])
    for frame in report["frames"]:
        np.testing.assert_allclose(frame["compensated_nmse_linear"], scales[:, None]**2*np.asarray(frame["nmse_linear"]), rtol=1e-10, atol=1e-14)
    return dict(complete=True, bits_per_snr=report["bits_per_snr"], frame_count=len(report["frames"]),
        **report["verification"])


def plot_comparison(output, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), layout="constrained")
    styles = (("compensated_single", "Single-tap / compensated CSI", "#C8752B", "o", "-"),
              ("compensated_full", "Full-matrix / compensated CSI", "#2166AC", "s", "-"),
              ("reference_single", "Single-tap / perfect CSI", "#C8752B", "o", "--"),
              ("reference_full", "Full-matrix / perfect CSI", "#2166AC", "s", "--"))
    for col, (channel, r) in enumerate(reports.items()):
        ber, nmse = axes[:, col]
        x, floor = np.asarray(r["snr_db"]), .5/r["bits_per_snr"]
        for key, label, color, marker, line in styles:
            values = np.asarray(r["curves"][key]["ber"])
            ber.semilogy(x, np.maximum(values, floor), color=color, marker=marker, linestyle=line, label=label)
            zero = values == 0
            ber.scatter(x[zero], np.full(zero.sum(), floor), marker=marker, facecolors="white", edgecolors=color, zorder=5)
        ber.set(title=f"{channel} | 5 frames, {r['bits_per_snr']:,} bits/SNR", ylabel="BER")
        ber.legend(fontsize=8)
        nmse.plot(x, r["mean_compensated_nmse_db"], "o-", color="#2166AC", label="Compensated CSI")
        nmse.plot(x, r["target_nmse_db"], "x--", color="black", markersize=7, label="Preset target")
        nmse.text(.03, .07, "Perfect CSI: NMSE = 0 (−∞ dB)", transform=nmse.transAxes, fontsize=9)
        nmse.set(xlabel="SNR (dB)", ylabel="Mean full-matrix CSI NMSE (dB)")
        nmse.legend(fontsize=8)
        for ax in (ber, nmse):
            ax.set_xticks(x)
            ax.grid(alpha=.25, which="both")
    fig.suptitle("CP = 80 ms (480 samples) | D-D-D-D | FFT1024 | Q24 | 192 raw delays\n"
                 "Reference-assisted compensation uses full perfect CSI | open markers: zero observed errors", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"comparison.{ext}", dpi=180)
    plt.close(fig)


def run(source, output, backend, *, options=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the saved CP80 source")
    options = load_target_config() if options is None else options
    original, baseline, physical = verify_source(source)
    targets = {ch: target_curve(options["snr_db"], **options["targets"][ch], decay_db=options["decay_db"]).tolist() for ch in CHANNELS}
    scales = {ch: error_scales(baseline[ch]["mean_raw_nmse_linear"], targets[ch]).tolist() for ch in CHANNELS}
    metadata = dict(schema="four-data-cp80-reference-assisted-target-nmse-v1", options=options,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={name: file_sha256(ROOT/name) for name in CODE},
        source=dict(path=str(source), metrics_sha256=file_sha256(source/"metrics.json"), physical=physical,
            channel_manifests=original["channel_manifests"]), configs=original["configs"],
        target_nmse_db=targets, error_scales=scales, csi_compensation=True, input_view="reference_assisted", cp_ms=80,
        compensation="H_ref + alpha_s * (zero_extended_H_raw - H_ref)",
        scale_definition="one alpha per channel/SNR, calibrated from mean linear NMSE over all 5 frames and 4 symbols",
        reference_use="full-reference oracle assistance at all 1024 delays, NMSE scoring and perfect single-symbol detection",
        comparison="identical saved CP480 physical observations; raw and perfect counts reverified",
        precision="complex128", device=backend.device, assisted_label="compensated CSI",
        ber_plot_title="CP=80 ms / 480 samples | compensated, raw audit and perfect CSI | Q24",
        nmse_plot_title="CP=80 ms | full-support reference-assisted target NMSE | Q24")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output belongs to a different CP80 compensation experiment")
    write_json(manifest, metadata)
    started = time.perf_counter()
    reports = {ch: evaluate(ch, source, output, metadata, baseline[ch], backend) for ch in CHANNELS}
    checks = {ch: verify_report(r) for ch, r in reports.items()}
    finish(output, metadata, reports)
    plot_comparison(output, reports)
    write_json(output/"verification.json", dict(complete=True, channels=checks,
        source_physical_sha256={ch: p["sha256"] for ch, p in physical.items()},
        csi_compensation=True, input_view="reference_assisted", wall_seconds=time.perf_counter()-started))
    return output
