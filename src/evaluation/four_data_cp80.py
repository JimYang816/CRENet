"""Paired raw/perfect-CSI LMMSE experiment with an actual 80-ms CP."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest, frozen_support_matches_config, load_dddd_config
from data_generation.src.transmitter.ofdm import carrier_masks, decisions
from src.common.experiment import file_sha256, write_json
from src.evaluation import ncs1_delay_support as receiver
from src.evaluation.four_data_lmmse import basis_diagnostics, detect, finish, model_basis


CONFIG = "configs/evaluation/four_data_cp80.json"
CODE = tuple(dict.fromkeys((*receiver.CODE, "src/evaluation/four_data_cp80.py",
    "scripts/evaluate/four_data_cp80.py", CONFIG, "configs/data/nof1/delay_support192.json",
    "configs/data/nof1/dddd.json", "configs/data/ncs1/dddd.json")))
CURVES = ("raw_single", "raw_full", "reference_single", "reference_full")


def load_options(path=None):
    options = json.loads(Path(path or ROOT/CONFIG).read_text(encoding="utf-8"))
    if (options.get("channels") != ["NOF1", "NCS1"] or options.get("cp_ms") != 80
            or options.get("tap_count") != 192 or options.get("input_view") != "raw"
            or options.get("algorithms") != ["single_tap_mmse", "full_matrix_lmmse"]
            or options.get("support_files") != {
                "NOF1": "configs/data/nof1/delay_support192.json",
                "NCS1": "configs/data/ncs1/delay_support192.json"}):
        raise ValueError("CP80 requires both channels, frozen 192 taps and raw/perfect LMMSE")
    for key in ("prior_records", "prior_starts_seconds", "test_records"):
        values = options.get(key, [])
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Nonempty distinct {key} required")
    if set(options["prior_records"]) & set(options["test_records"]):
        raise ValueError("Independent prior and test records must be disjoint")
    return options


def channel_settings(channel, options, archive_sha256, smoke=False):
    # Validate the original configuration before applying this experiment's CP.
    base = load_dddd_config(channel, legacy_comb=True)
    if (base["frame_structure"], base["dpss_expected_count"], base["dpss_half_bandwidth_hz"],
            base["data_count"], base["pilot_count"]) != ("DDDD", 24, 12, 512, 256):
        raise ValueError("Expected the current DDDD / Q24 / 12-Hz frame")
    if (not set(options["prior_records"]).issubset(base["prior_records"])
            or not set(options["prior_starts_seconds"]).issubset(base["starts_seconds"])
            or not set(options["test_records"]).issubset(base["learning_dataset"]["records"]["test"])
            or options["start_seconds"] not in base["learning_dataset"]["starts_seconds"]):
        raise ValueError("Windows must follow the existing independent prior/test split")
    support_file = options["support_files"][channel]
    support = json.loads((ROOT/support_file).read_text(encoding="utf-8"))
    indices = np.asarray(support.get("indices", []))
    source_windows = support.get("prior_windows", [])
    if (support.get("schema") != f"{channel.lower()}-frozen-delay-support-v1"
            or support.get("channel") != channel or support.get("tap_count") != 192
            or indices.shape != (192,) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= base["nfft"]
            or not frozen_support_matches_config(support, base)
            or support.get("source_archive_sha256") != archive_sha256
            or source_windows != [[r, s] for r in base["prior_records"] for s in base["starts_seconds"]]):
        raise ValueError("Frozen support differs from the accepted independent-prior source")
    c = dict(base)
    cp_samples = options["cp_ms"]*c["fs_baseband"]/1000
    if not float(cp_samples).is_integer() or not 0 < cp_samples <= c["nfft"]:
        raise ValueError("CP duration must correspond to a valid integer sample count")
    c.update(cp=int(cp_samples), delay_length=192)
    if smoke:
        c["snr_db"] = [0, 15, 30]
    prior_records = options["prior_records"][:2] if smoke else options["prior_records"]
    prior_starts = options["prior_starts_seconds"][:1] if smoke else options["prior_starts_seconds"]
    tests = options["test_records"][:1] if smoke else options["test_records"]
    study = dict(channel=channel, tap_count=192, support_file=support_file, frozen_support=support)
    return c, study, [[r, s] for r in prior_records for s in prior_starts], [
        [r, options["start_seconds"]] for r in tests]


def channel_report(results, reference):
    r = results["selected192"]
    curves = {f"raw_{key}": dict(bit_errors=value["errors"], ber=value["ber"])
              for key, value in r["curves"].items()}
    curves.update({f"reference_{key}": dict(bit_errors=value["errors"], ber=value["ber"])
                   for key, value in reference.items()})
    return dict(q=24, complete=True, snr_db=r["snr_db"], bits_per_snr=r["bits_per_snr"],
        curves=curves, mean_raw_nmse_linear=r["mean_nmse_linear"], mean_raw_nmse_db=r["mean_nmse_db"],
        reference_nmse_linear=0., reference_nmse_db="-inf", delay_count=r["delay_count"],
        coefficient_count=r["coefficient_count"], support_span=r["support_span"],
        test_mean_coverage=r["test_mean_coverage"], test_min_coverage=r["test_min_coverage"], frames=r["frames"])


def verify_channel(path, meta, report, backend):
    c = meta["config"]
    frames = report["frames"]
    bits = len(frames)*4*c["data_count"]*2
    if c["cp"]/c["fs_baseband"] != .08 or report["bits_per_snr"] != bits:
        raise ValueError("CP duration or bit count mismatch")
    for key in ("single", "full"):
        errors = np.sum([f[f"{key}_errors"] for f in frames], axis=(0, 2))
        np.testing.assert_array_equal(errors, report["curves"][f"raw_{key}"]["bit_errors"])
    nmse = np.mean([f["nmse_linear"] for f in frames], axis=(0, 2))
    np.testing.assert_allclose(nmse, report["mean_raw_nmse_linear"], rtol=1e-13)
    np.testing.assert_allclose(10*np.log10(nmse), report["mean_raw_nmse_db"], atol=1e-12)
    for key in CURVES:
        values = report["curves"][key]
        errors = np.asarray(values["bit_errors"])
        if np.any((errors < 0) | (errors > bits)):
            raise ValueError("Invalid BER numerator")
        np.testing.assert_allclose(errors/bits, values["ber"], rtol=0, atol=1e-15)
    audit, ideal_counts = [], np.zeros(2, dtype=np.int64)
    with h5py.File(path, "r") as h:
        if not h.attrs["complete"] or h["basis"].shape != (4*(c["nfft"]+480), 24):
            raise ValueError("Physical file does not contain a complete CP480/Q24 frame")
        basis_sha = digest(h["basis"][:].tolist())
        noise, known = h["unit_noise"][:], h["known"][:]
        for g, expected in zip(h["test"].values(), meta["test_windows"], strict=True):
            if [g.attrs["record"], g.attrs["start_seconds"]] != expected:
                raise ValueError("Physical test window differs from the manifest")
            variance = g["variance"][-1:]
            observed = g["clean"][:][None]+np.sqrt(variance)[:, None, None]*g["standard"][:][None]
            single, full = detect(g["truth"][:], observed, known, variance, noise,
                                  carrier_masks(c)["data"], backend)
            errors = [np.count_nonzero(decisions(x)[0] != g["bits"][:], axis=-1).sum(axis=-1)
                      for x in (single, full)]
            ideal_counts += [int(value.sum()) for value in errors]
            audit.append(dict(record=int(g.attrs["record"]), snr_db=c["snr_db"][-1],
                perfect_single_errors_per_symbol=errors[0].tolist(),
                perfect_full_errors_per_symbol=errors[1].tolist(),
                model_error=json.loads(g.attrs["model_error"]),
                intersymbol_leakage=json.loads(g.attrs["intersymbol_leakage"])))
    np.testing.assert_array_equal(ideal_counts, [report["curves"][f"reference_{key}"]["bit_errors"][-1]
                                                for key in ("single", "full")])
    return dict(complete=True, cp_samples=480, cp_ms=80, bits_per_snr=bits,
        basis_sha256=basis_sha, frame_count=len(frames), perfect_csi_30db_audit=audit,
        max_model_error=max(max(f["model_error"]) for f in audit),
        max_intersymbol_leakage=max(max(f["intersymbol_leakage"]) for f in audit))


def plot_comparison(output, metadata, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), layout="constrained")
    styles = (("raw_single", "Single-tap / estimated CSI", "#C8752B", "o", "-"),
              ("raw_full", "Full-matrix / estimated CSI", "#2166AC", "s", "-"),
              ("reference_single", "Single-tap / perfect CSI", "#C8752B", "o", "--"),
              ("reference_full", "Full-matrix / perfect CSI", "#2166AC", "s", "--"))
    for col, (channel, r) in enumerate(reports.items()):
        ber, nmse = axes[:, col]
        floor = .5/r["bits_per_snr"]
        x = np.asarray(r["snr_db"])
        for key, label, color, marker, line in styles:
            values = np.asarray(r["curves"][key]["ber"])
            ber.semilogy(x, np.maximum(values, floor), color=color, marker=marker, linestyle=line, label=label)
            zero = values == 0
            ber.scatter(x[zero], np.full(zero.sum(), floor), marker=marker,
                        facecolors="white", edgecolors=color, zorder=5)
        ber.set(title=f"{channel} | {len(r['frames'])} frames, {r['bits_per_snr']:,} bits/SNR", ylabel="BER")
        ber.legend(fontsize=8)
        nmse.plot(x, r["mean_raw_nmse_db"], "o-", color="#2166AC", label="Estimated CSI (raw)")
        nmse.text(.03, .07, "Perfect CSI: NMSE = 0 (−∞ dB)", transform=nmse.transAxes, fontsize=9)
        nmse.set(xlabel="SNR (dB)", ylabel="Mean full-matrix CSI NMSE (dB)")
        nmse.legend(fontsize=8)
        for ax in (ber, nmse):
            ax.set_xticks(x)
            ax.grid(alpha=.25, which="both")
    fig.suptitle("CP = 80 ms (480 samples at 6 kHz) | D-D-D-D | FFT1024 | Q24 | 192 fixed delays\n"
                 "Same payload and noise across CSI/equalizer choices; open markers: zero observed errors", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"comparison.{ext}", dpi=180)
    plt.close(fig)


def run(output, archive_path, backend, smoke=False, *, options=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    options = load_options() if options is None else options
    archive_sha = file_sha256(archive_path)
    code_sha = {name: file_sha256(ROOT/name) for name in CODE}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metas = {}
    for channel in options["channels"]:
        c, study, priors, tests = channel_settings(channel, options, archive_sha, smoke)
        metas[channel] = dict(schema=f"{channel.lower()}-cp80-fixed192-v1", config=c,
            frame_config_sha256=digest(c), support_source_config_sha256=study["frozen_support"]["frame_config_sha256"],
            study=study, device=backend.device, smoke=smoke, prior_windows=priors, test_windows=tests,
            git_head=commit, code_sha256=code_sha, archive_sha256=archive_sha, precision="complex128",
            raw_csi=True, csi_compensation=False, cp_ms=80,
            selection="reuse accepted 192 physical delays; recompute independent full-frame priors at CP480",
            reference_use="complete perfect single-symbol CSI, no DPSS or delay truncation; actual intersymbol interference retained",
            nmse="mean frame/symbol full-matrix linear NMSE including omitted taps, then 10 log10",
            dpss_diagnostics=basis_diagnostics(c, model_basis(c)[1]))
    metadata = dict(schema="four-data-cp80-lmmse-v1", options=options, smoke=smoke, cp_ms=80,
        input_view="raw", csi_compensation=False, configs={ch: meta["config"] for ch, meta in metas.items()},
        channel_manifests=metas,
        snr_definition="mean received useful-window power / expected filtered RF thermal-noise power",
        reference_use="perfect current-symbol CSI; physical intersymbol interference is retained")
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output belongs to different CP80 settings or source; choose a new directory")
    write_json(manifest_path, metadata)
    started = time.perf_counter()
    reports, verification = {}, {}
    for channel, meta in metas.items():
        directory = output/channel.lower()
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory/"manifest.json", meta)
        write_json(directory/"frozen_support192.json", meta["study"]["frozen_support"])
        print(f"{channel}: CP480, preparing fresh priors and physical test frames", flush=True)
        path = receiver.prepare(directory, archive_path, meta)
        supports, selection = receiver.freeze_supports(path, directory, meta["study"])
        results, perfect = receiver.evaluate(path, directory, meta, supports, backend)
        receiver.finish(directory, meta, selection, results, perfect, path)
        reports[channel] = channel_report(results, perfect)
        verification[channel] = verify_channel(path, meta, reports[channel], backend)
        print(f"{channel}: verified all raw counts/NMSE and perfect-CSI 30-dB counts", flush=True)
    if len({value["basis_sha256"] for value in verification.values()}) != 1:
        raise ValueError("Channels do not share the same actual DPSS basis")
    finish(output, metadata, reports)
    plot_comparison(output, metadata, reports)
    write_json(output/"verification.json", dict(complete=True, channels=verification,
        shared_dpss_basis=True, wall_seconds=time.perf_counter()-started))
    return output
