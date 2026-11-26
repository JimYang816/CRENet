"""Compare full-frame DPSS-BEM and CE-BEM on paired raw CSI observations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest
from data_generation.src.receiver.ce_bem import ce_basis
from data_generation.src.transmitter.ofdm import carrier_masks
from src.common.experiment import file_sha256, write_json
from src.evaluation import ncs1_delay_support as receiver
from src.evaluation.four_data_cp80 import CODE as CP80_CODE, channel_settings, load_options


CE_HALF_BANDWIDTH = 16
CODE = (*CP80_CODE, "data_generation/src/receiver/ce_bem.py",
        "src/evaluation/four_data_bem_comparison.py", "scripts/evaluate/four_data_bem_comparison.py")
LABELS = {"dpss_bem": "DPSS-BEM (Q=24)", "ce_bem": "CE-BEM (±16 bins)"}
ALGORITHMS = {"single": "Single-tap LMMSE", "full": "Full-matrix LMMSE"}


def verify_channel(path, meta, reports):
    c = meta["config"]
    test_windows = meta["test_windows"]
    nframes, nsnr = len(test_windows), len(c["snr_db"])
    bits = nframes*4*c["data_count"]*2
    total = 4*(c["nfft"]+c["cp"])
    observations = 4*int(carrier_masks(c)["tracking"].sum())
    with h5py.File(path, "r") as h:
        if not h.attrs["complete"] or h.attrs["identity"] != digest(meta):
            raise ValueError("Incomplete or mismatched paired physical source")
        for name, settings in meta["bem_models"].items():
            basis = h[settings["prior_group"]]
            q = settings["q"]
            if (basis["basis"].shape != (total, q)
                    or basis["prior"].shape != (len(meta["prior_windows"]), q, c["nfft"])):
                raise ValueError(f"{name}: stored BEM or prior has the wrong dimensions")
    for name, report in reports.items():
        frames = report["frames"]
        if ([[f["record"], f["start_seconds"]] for f in frames] != test_windows
                or report["snr_db"] != c["snr_db"] or report["bits_per_snr"] != bits
                or report["delay_count"] != c["delay_length"]
                or report["coefficient_count"] != meta["bem_models"][name]["q"]*c["delay_length"]
                or report["estimator_diagnostics"][0]["observations"] != observations):
            raise ValueError(f"{name}: observations, coefficient count or test split changed")
        errors = np.array([frame["nmse_linear"] for frame in frames])
        projection = np.array([frame["projection_nmse_linear"] for frame in frames])
        if (errors.shape != (nframes, nsnr, 4) or projection.shape != (nframes, 4)
                or not np.all(np.isfinite(errors)) or not np.all(np.isfinite(projection))
                or np.any(errors+1e-12 < projection[:, None, :])):
            raise ValueError(f"{name}: invalid measured or projection NMSE")
        mean = errors.mean(axis=(0, 2))
        np.testing.assert_allclose(report["mean_nmse_linear"], mean, rtol=1e-13)
        np.testing.assert_allclose(report["mean_nmse_db"], 10*np.log10(mean), atol=1e-12)
        report["projection_floor_db"] = float(10*np.log10(projection.mean()))
        for algorithm in ALGORITHMS:
            counts = np.array([frame[f"{algorithm}_errors"] for frame in frames])
            if counts.shape != (nframes, nsnr, 4) or np.any((counts < 0) | (counts > 2*c["data_count"])):
                raise ValueError(f"{name}: invalid per-symbol error counts")
            counts = counts.sum(axis=(0, 2))
            np.testing.assert_array_equal(counts, report["curves"][algorithm]["errors"])
            np.testing.assert_allclose(counts/bits, report["curves"][algorithm]["ber"], rtol=0, atol=1e-15)
    return dict(complete=True, frame_count=nframes, bits_per_snr=bits, observation_rows=observations,
                shared_physical_sha256=file_sha256(path), linear_nmse_and_ber_reaggregated=True)


def write_summary(output, channels):
    with (output/"summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "model", "snr_db", "mean_nmse_linear", "mean_nmse_db", "projection_floor_db",
                         "single_bit_errors", "single_ber", "full_bit_errors", "full_ber", "bits_per_snr"])
        for channel, results in channels.items():
            for name, r in results["models"].items():
                for i, snr in enumerate(r["snr_db"]):
                    writer.writerow([channel, name, snr, r["mean_nmse_linear"][i], r["mean_nmse_db"][i],
                        r["projection_floor_db"], r["curves"]["single"]["errors"][i], r["curves"]["single"]["ber"][i],
                        r["curves"]["full"]["errors"][i], r["curves"]["full"]["ber"][i], r["bits_per_snr"]])


def plot_results(output, channels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nmse_figure, nmse_axes = plt.subplots(1, 2, figsize=(10.6, 4.2), layout="constrained")
    ber_figure, ber_axes = plt.subplots(2, 2, figsize=(10.6, 7.0), layout="constrained")
    styles = {"dpss_bem": ("#2166AC", "o"), "ce_bem": ("#C8752B", "s")}
    for col, (channel, results) in enumerate(channels.items()):
        nmse = nmse_axes[col]
        for name, r in results["models"].items():
            color, marker = styles[name]
            snr = np.asarray(r["snr_db"])
            nmse.plot(snr, r["mean_nmse_db"], color=color, marker=marker, label=LABELS[name])
            for row, algorithm in enumerate(ALGORITHMS):
                ber = ber_axes[row, col]
                values = np.asarray(r["curves"][algorithm]["ber"])
                floor, zeros = .5/r["bits_per_snr"], values == 0
                ber.semilogy(snr, np.maximum(values, floor), color=color, marker=marker, label=LABELS[name])
                if np.any(zeros):
                    ber.scatter(snr[zeros], np.full(zeros.sum(), floor), marker=marker,
                                facecolors="white", edgecolors=color, zorder=5)
        nmse.set(title=channel, ylabel="Mean full-matrix NMSE (dB)")
        for row, label in enumerate(ALGORITHMS.values()):
            ber_axes[row, col].set(title=f"{channel} · {label}", ylabel="BER")
        for ax in (nmse, *ber_axes[:, col]):
            ax.set(xlabel="SNR (dB)", xticks=snr)
            ax.grid(alpha=.25, which="both")
            ax.legend(fontsize=8)
    nmse_figure.suptitle("Raw CSI estimation · CP 80 ms · 224 pilots · 192 fixed delays")
    ber_figure.suptitle("Paired BER with raw CSI · open markers represent zero observed errors")
    for name, figure in (("nmse_snr", nmse_figure), ("ber_snr", ber_figure)):
        figure.savefig(output/f"{name}.png", dpi=180)
        figure.savefig(output/f"{name}.pdf")
        plt.close(figure)


def run(output, archive_path, backend, *, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    options = load_options()
    archive_sha = file_sha256(archive_path)
    channel_metadata = {}
    for channel in options["channels"]:
        c, study, priors, tests = channel_settings(channel, options, archive_sha, smoke)
        models = dict(
            dpss_bem=dict(prior_group="/", q=c["dpss_expected_count"], half_bandwidth_hz=c["dpss_half_bandwidth_hz"]),
            ce_bem=dict(prior_group="bem/ce_bem", q=2*CE_HALF_BANDWIDTH+1, tail_ratio=None,
                        half_bandwidth_bins=CE_HALF_BANDWIDTH,
                        half_bandwidth_hz=CE_HALF_BANDWIDTH*c["fs_baseband"]/c["nfft"]),
        )
        channel_metadata[channel] = dict(config=c, study=study, prior_windows=priors, test_windows=tests, bem_models=models)
    metadata = dict(schema="four-data-bem-comparison-v1", input_view="raw", smoke=smoke, options=options,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={name: file_sha256(ROOT/name) for name in CODE}, archive_sha256=archive_sha,
        device=backend.device, gpu=backend.torch.cuda.get_device_name() if backend.torch else None, precision="complex128",
        ce_time_model="exp(2j*pi*k*n/nfft)/sqrt(frame_samples), k=-16..16; one coefficient vector per delay per frame",
        nmse_definition="mean frame/symbol full-matrix linear NMSE including omitted delays, then 10*log10",
        projection_floor="least-squares projection of held-out reference onto each BEM at receiver FFT-window samples",
        channel_manifests=channel_metadata)
    manifest = output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output belongs to a different BEM study")
    write_json(manifest, metadata)

    started = time.perf_counter()
    channels, verification = {}, {}
    for channel, settings in channel_metadata.items():
        directory = output/channel.lower()
        directory.mkdir(exist_ok=True)
        c = settings["config"]
        meta = dict(settings, run_identity=digest(metadata))
        delays = np.asarray(settings["study"]["frozen_support"]["indices"])
        basis = ce_basis(c, CE_HALF_BANDWIDTH)
        print(f"{channel}: preparing {len(meta['prior_windows'])} prior windows and {len(meta['test_windows'])} paired frames",
              flush=True)
        physical = receiver.prepare(directory, archive_path, meta, extra_bases={"ce_bem": basis})
        reports, reference = receiver.evaluate(physical, directory, meta, dict.fromkeys(LABELS, delays), backend)
        verification[channel] = verify_channel(physical, meta, reports)
        channels[channel] = dict(models=reports, perfect_csi=reference)
        print(f"{channel}: checked paired results and full-matrix NMSE for both bases", flush=True)

    write_summary(output, channels)
    plot_results(output, channels)
    write_json(output/"metrics.json", dict(complete=True, metadata=metadata, channels=channels))
    write_json(output/"verification.json", dict(complete=True, channels=verification,
                                                elapsed_seconds=time.perf_counter()-started))
    return output
