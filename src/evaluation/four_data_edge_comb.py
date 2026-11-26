"""Compare the two DDDD carrier allocations under fixed single-symbol CSI settings."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT
from data_generation.src.dataset.four_data_math import array_sha256, reference_statistics
from data_generation.src.dataset.four_data_physical import initialize_worker, replay_window, window_results
from data_generation.src.dataset.four_data_protocol import channel_config, windows
from data_generation.src.receiver.single_symbol import SymbolPilotDesign, symbol_basis
from data_generation.src.transmitter.ofdm import carrier_masks, decisions, make_frame
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_delay192 import select_prior_delays
from src.evaluation.four_data_lmmse import detect
from src.evaluation.four_data_q_sweep import BlockedReceiver
from src.evaluation.four_data_single_symbol import (
    dataset_manifest, db, estimate_frames, fit_independent_prior, fit_model, projection_nmse,
    verify_file, verify_split, write_csv,
)
from src.evaluation.ncs1_delay_support import full_nmse


CONFIG = "configs/evaluation/four_data_edge_comb.json"


def load_options(path=None):
    options = json.loads(Path(path or ROOT/CONFIG).read_text(encoding="utf-8"))
    if (options["channels"] != ["NOF1", "NCS1"] or options["null_per_edge"] != 128
            or type(options["delay_count"]) is not int or not 1 <= options["delay_count"] <= 1024
            or not 0 <= options["model_error_ratio"] <= 1
            or type(options["ber_frames_per_channel"]) is not int or options["ber_frames_per_channel"] < 1):
        raise ValueError("Invalid fixed single-symbol CP80 carrier experiment")
    return options


def allocation(c):
    layout = carrier_masks(c)
    n = c["nfft"]
    result = {kind: (np.flatnonzero(layout[kind])-n//2).tolist()
              for kind in ("pilot", "data", "protection", "edge", "tracking")}
    return dict(nfft=n, centered_indices=result, counts={kind: len(values) for kind, values in result.items()},
                subcarrier_spacing_hz=c["fs_baseband"]/n, null_per_edge=c["null_per_edge"])


def compare_source(baseline, options, archive_sha):
    path = baseline/"manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(temporal_orders=[[options["bandwidth_hz"], [options["q"]]]],
        delay_counts=[options["delay_count"]], model_error_ratios=[options["model_error_ratio"]],
        ber_frames_per_channel=options["ber_frames_per_channel"])
    if (manifest["smoke"] or manifest["archive_sha256"] != archive_sha
            or any(manifest["options"][key] != value for key, value in expected.items())):
        raise ValueError("The historical carrier layout must use matching raw single-symbol CSI settings")
    return dict(path=str(baseline), manifest_sha256=file_sha256(path), previous_commit=manifest["git_head"])


def replay_observations(packet, old, frame):
    clean, standard, variance, bits = (packet[key] for key in ("clean", "standard", "variance", "bits"))
    changed = 0
    if old is not None:
        np.testing.assert_array_equal(bits, old["samples/bits"][frame])
        np.testing.assert_allclose(standard, old["samples/standard_noise"][frame], rtol=0, atol=1e-12)
        changed = int(np.count_nonzero(np.any(abs(clean-old["samples/clean_observations"][frame]) > 1e-12, axis=(1, 2))))
    y = clean[None] + np.sqrt(variance)[..., None, None]*standard[None]
    return y, variance, bits, changed


def split_result(split, c, old, replay, archive, receiver, options, estimators, known, backend, smoke):
    plan = windows(c, split)
    repeats = c["learning_dataset"]["repeats"][split]
    n_snr, n_frames = len(c["snr_db"]), len(plan)*repeats
    diag = dict(validation=0, test=1 if smoke else options["ber_frames_per_channel"])[split]
    diag_ids = np.linspace(0, n_frames-1, min(n_frames, diag), dtype=int)
    symbol_count = known.shape[0]
    nmse = np.full((n_snr, n_frames, symbol_count), np.nan)
    counts = np.zeros((2, n_snr), dtype=np.int64)
    model = estimators[0].model
    basis, delays = model.basis, model.delays
    sampled_basis = np.broadcast_to(basis, (symbol_count, *basis.shape))
    masks = carrier_masks(c)
    noise = receiver.chain.noise_covariance
    floor = options["model_error_ratio"]
    max_nmse_difference, changed_frames = 0., 0
    for wi, packet in replay:
        record, start = plan[wi]
        frame = slice(wi*repeats, (wi+1)*repeats)
        if old is None:
            sounding = archive.load(record)
            truth = np.stack([receiver.reference(sounding, start, s)[0] for s in range(symbol_count)]).astype(np.complex64)
        else:
            truth = old["windows/reference_taps"][wi]
            # The reference is a physical linear operator, independent of transmitted tones.
            if wi == 0:
                sounding = archive.load(record)
                direct_truth = np.stack([receiver.reference(sounding, start, s)[0] for s in range(symbol_count)])
                np.testing.assert_allclose(truth, direct_truth.astype(truth.dtype), rtol=1e-6, atol=1e-8)
        projection, energy = reference_statistics(sampled_basis, truth, delays)
        y, variance, bits, changed = replay_observations(packet, old, frame)
        changed_frames += changed
        coeff = estimate_frames(estimators, y, variance, floor)
        score = projection_nmse(coeff, projection, energy)
        nmse[:, frame] = score
        if wi == 0:
            isolated = np.zeros_like(y[:1, :1])
            for s in range(symbol_count):
                isolated.fill(0)
                isolated[0, 0, s] = y[0, 0, s]
                by_symbol = estimate_frames(estimators, isolated, variance[:1, :1], floor)[0, 0, s]
                np.testing.assert_allclose(by_symbol, coeff[0, 0, s], rtol=1e-10, atol=1e-11)
        for fi in diag_ids[(diag_ids >= frame.start) & (diag_ids < frame.stop)]:
            offset = fi-frame.start
            selected = basis @ coeff[:, offset]
            direct = full_nmse(selected, truth.astype(np.complex128), delays)
            difference = float(np.max(abs(score[:, offset]-direct)))
            max_nmse_difference = max(max_nmse_difference, difference)
            np.testing.assert_allclose(score[:, offset], direct, rtol=1e-10, atol=1e-11)
            taps = np.zeros(selected.shape[:-1]+(int(delays[-1])+1,), complex)
            taps[..., delays] = selected
            symbols = detect(taps, y[:, offset], known, variance[:, offset], noise, masks["data"], backend)
            counts += np.count_nonzero(decisions(np.asarray(symbols)) != bits[offset], axis=(2, 3, 4))
        if wi == 0 or (wi+1) % 25 == 0 or wi+1 == len(plan):
            print(f"{c['channel']} {split}: replayed {frame.stop}/{n_frames} frames per SNR", flush=True)
    if not np.all(np.isfinite(nmse)) or (old is not None and changed_frames != n_frames):
        raise ValueError("Carrier comparison must score every frame after replaying its changed waveform")
    denominator = len(diag_ids)*symbol_count*c["data_count"]*2
    result = dict(snr_db=c["snr_db"], frames_per_snr=n_frames, symbols_per_snr=n_frames*symbol_count,
        mean_nmse_db=db(nmse.mean(axis=(1, 2))).tolist(), all_snr_nmse_db=float(db(nmse.mean())),
        nmse_db_by_position=db(nmse.mean(axis=1)).tolist(), diagnostic_ber_frames=diag_ids.tolist(),
        diagnostic_bits_per_snr=denominator, bit_errors=counts.tolist(),
        ber=(counts/denominator).tolist() if denominator else None,
        checks=dict(independent_observations=True, max_direct_nmse_difference=max_nmse_difference,
                    waveform_changed_frames=changed_frames if old is not None else None,
                    paired_noise_and_payload=old is not None))
    return result, nmse


def previous_test(baseline, c, accepted, prior, test):
    root = baseline/c["channel"].lower()
    selection = json.loads((root/"selection.json").read_text(encoding="utf-8"))
    old = json.loads((root/"test.json").read_text(encoding="utf-8"))
    if (selection["source_model_identity"] != accepted["model_identity"]
            or selection["basis_sha256"] != array_sha256(prior.model.basis)
            or selection["delay_indices"] != prior.model.delays.tolist()
            or old["diagnostic_ber_frames"] != test["diagnostic_ber_frames"]
            or old["diagnostic_bits_per_snr"] != test["diagnostic_bits_per_snr"]):
        raise ValueError("The historical result must use the same prior support and paired test data")
    repeats = c["learning_dataset"]["repeats"]["test"]
    records, starts = zip(*windows(c, "test"), strict=True)
    path = root/"paired_test_nmse.npz"
    with np.load(path) as pairs:
        for key, value in dict(record=np.repeat(records, repeats), start_seconds=np.repeat(starts, repeats),
                               repeat=np.tile(np.arange(repeats), len(records)), snr_db=c["snr_db"]).items():
            np.testing.assert_array_equal(pairs[key], value)
        nmse = pairs["single_symbol_nmse_linear"]
    if nmse.shape != (len(c["snr_db"]), test["frames_per_snr"], 4) or not np.all(np.isfinite(nmse)):
        raise ValueError("The historical per-symbol scores do not match the current test plan")
    np.testing.assert_allclose(db(nmse.mean(axis=(1, 2))), old["single_symbol_nmse_db"], rtol=0, atol=1e-12)
    return dict(mean_nmse_db=old["single_symbol_nmse_db"], all_snr_nmse_db=old["single_symbol_all_snr_nmse_db"],
        bit_errors=old["bit_errors"][0], ber=old["ber"][0], source_sha256=file_sha256(path)), nmse


def run_channel(output, data, archive_path, archive_sha, baseline, channel, options, backend, workers, smoke):
    base_c = channel_config(channel, smoke)
    c = dict(base_c, null_per_edge=options["null_per_edge"])
    layout = allocation(c)
    if layout["counts"] != dict(pilot=256, data=512, protection=0, edge=256, tracking=256):
        raise ValueError("Expected 512 data, 256 comb pilots and 256 edge nulls")
    write_json(output/"carrier_allocation.json", layout)
    archive, receiver = WatermarkArchive(archive_path, channel), BlockedReceiver(c)
    noise = receiver.chain.noise_covariance
    accepted, source_sha = (None, None) if smoke else dataset_manifest(data, base_c, archive_sha)
    if accepted is not None:
        with h5py.File(data/"model.h5", "r") as h:
            np.testing.assert_allclose(noise, h["unit_rf_noise_covariance"][:], rtol=1e-10, atol=1e-12)
    band, q = options["bandwidth_hz"], options["q"]
    basis = symbol_basis(c["nfft"], c["fs_baseband"], band, q)
    coeffs, powers, records, prior_ids = fit_independent_prior(c, archive, {band: basis}, smoke)
    delays = select_prior_delays(powers, options["delay_count"])
    prior = fit_model(c, basis, delays, coeffs[band][..., delays], records)
    known = make_frame(c, 1, 2).known
    design = SymbolPilotDesign(basis, delays, known, carrier_masks(c), noise, backend)
    matrices = backend.numpy(design.responses).reshape(len(known), len(design.rows), -1)
    ranks = np.linalg.matrix_rank(matrices)
    estimators = design.build(prior)
    del design, matrices
    prior_info = dict(basis_sha256=array_sha256(basis), delay_indices=delays.tolist(),
        prior_windows=prior_ids, prior_delay_coverage=float(powers[:, delays].sum()/powers.sum()),
        estimated_coefficients=q*len(delays), observation_rows=layout["counts"]["tracking"], ranks=ranks.tolist())
    print(f"{channel}: {prior_info['observation_rows']} pilot observations, ranks {ranks.tolist()}", flush=True)
    results, scores = {}, {}
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                initializer=initialize_worker, initargs=(c, archive_path, 1))
            if workers > 1 else nullcontext())
    with pool as workers_pool:
        for split in ("validation", "test"):
            path = data/f"{split}.h5"
            if accepted is not None:
                verify_file(path, accepted)
            repeats = c["learning_dataset"]["repeats"][split]
            plan = windows(c, split)
            jobs = [(wi, split, record, start, repeats) for wi, (record, start) in enumerate(plan)]
            replay = window_results(jobs, c, archive, receiver, pool=workers_pool, prefetch=2*workers, compute=replay_window)
            with (nullcontext() if smoke else h5py.File(path, "r")) as old:
                if old is not None:
                    verify_split(old, base_c, split, accepted)
                results[split], scores[split] = split_result(split, c, old, replay, archive, receiver,
                    options, estimators, known, backend, smoke)
            write_json(output/f"{split}.json", results[split])
    comparison, previous_nmse = (None, None) if smoke else previous_test(baseline, c, accepted, prior, results["test"])
    records, starts = zip(*windows(c, "test"), strict=True)
    repeats = c["learning_dataset"]["repeats"]["test"]
    arrays = dict(snr_db=c["snr_db"], record=np.repeat(records, repeats), start_seconds=np.repeat(starts, repeats),
                  repeat=np.tile(np.arange(repeats), len(records)), edge_comb_nmse_linear=scores["test"],
                  ber_diagnostic_frames=results["test"]["diagnostic_ber_frames"])
    if comparison is not None:
        arrays["inside_null_nmse_linear"] = previous_nmse
    np.savez_compressed(output/"test_nmse.npz", **arrays)
    return dict(prior=prior_info, carrier_counts=layout["counts"], **results, baseline=comparison,
                physical_source_manifest_sha256=source_sha)


def finish(output, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for measure, ylabel in (("nmse", "Mean full-matrix CSI NMSE (dB)"), ("ber", "Bit error rate")):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained", sharey=True)
        for ax, (channel, report) in zip(axes, reports.items(), strict=True):
            test, old = report["test"], report["baseline"]
            snr = test["snr_db"]
            ax.set(xlabel="SNR (dB)", ylabel=ylabel, xticks=snr)
            if measure == "nmse":
                current = np.asarray(test["mean_nmse_db"])
                ax.plot(snr, current, "o-", label="256 edge nulls", color="#146C7C")
                if old:
                    ax.plot(snr, old["mean_nmse_db"], "s--", label="128 in-band nulls", color="#A45534")
                ax.set_title(f"{channel} · {test['symbols_per_snr']:,} symbols/SNR")
                for i, s in enumerate(snr):
                    previous = old["mean_nmse_db"][i] if old else None
                    rows.append(dict(channel=channel, snr_db=s, symbols_per_snr=test["symbols_per_snr"],
                        edge_comb_nmse_db=current[i], inside_null_nmse_db=previous,
                        nmse_delta_db=current[i]-previous if old else None,
                        diagnostic_bits_per_snr=test["diagnostic_bits_per_snr"],
                        edge_comb_single_tap_ber=test["ber"][0][i], inside_null_single_tap_ber=old["ber"][0][i] if old else None,
                        edge_comb_full_matrix_ber=test["ber"][1][i], inside_null_full_matrix_ber=old["ber"][1][i] if old else None))
            else:
                ax.set_title(f"{channel} · {test['diagnostic_bits_per_snr']:,} bits/SNR")
                lower = .5/test["diagnostic_bits_per_snr"]
                for i, (algorithm, color, marker) in enumerate((("Single-tap", "#A45534", "o"), ("Full-matrix", "#146C7C", "s"))):
                    curves = [(test["ber"][i], "256 edge nulls", "-")]
                    if old:
                        curves.append((old["ber"][i], "128 in-band nulls", "--"))
                    for values, name, line in curves:
                        ax.semilogy(snr, np.maximum(values, lower), marker=marker, linestyle=line,
                                    color=color, label=f"{algorithm} / {name}")
            ax.grid(alpha=.23, which="both")
            ax.legend(fontsize=8)
        fig.suptitle("Single-symbol raw CSI · CP 80 ms" if measure == "nmse" else "Paired LMMSE detection · diagnostic subset")
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{measure}_snr.{ext}", dpi=170)
        plt.close(fig)
    write_csv(output/"summary.csv", rows)


def run(output, data, archive_path, baseline, options, backend, *, workers=1, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    archive_sha = file_sha256(archive_path)
    baseline_meta = None if smoke else compare_source(baseline, options, archive_sha)
    manifest = dict(schema="cp80-single-symbol-edge-comb-v1", options=options, git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), device=backend.device, smoke=smoke,
        archive_sha256=archive_sha, baseline=baseline_meta, accepted_physical_source=None if smoke else str(data),
        carrier_pattern="128 edge nulls, then 256 repetitions of pilot/data/data, then 128 edge nulls",
        snr_definition="Each allocation's frame-mean useful received power divided by expected filtered thermal power",
        pairing="Same physical windows, QPSK streams and standard RF noise; clean waveform replayed for new carriers",
        csi="Current-symbol comb-pilot LMMSE with independent physical priors; full reference used for scoring")
    path = output/"manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output belongs to a different carrier experiment")
    write_json(path, manifest)
    reports = {}
    for channel in options["channels"]:
        directory = output/channel.lower()
        directory.mkdir(exist_ok=True)
        reports[channel] = run_channel(directory, data/(channel.lower()+"_dddd"), archive_path, archive_sha, baseline,
                                       channel, options, backend, workers, smoke)
    if len({r["prior"]["basis_sha256"] for r in reports.values()}) != 1:
        raise ValueError("The two carriers must use the same single-symbol time basis")
    finish(output, reports)
    write_json(output/"metrics.json", dict(complete=True, channels=reports, wall_seconds=time.perf_counter()-began))
    return reports
