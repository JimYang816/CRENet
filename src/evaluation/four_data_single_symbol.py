"""Validation-selected, raw single-symbol CSI on the accepted CP80 splits."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT, digest
from data_generation.src.dataset.four_data_math import array_sha256, reconstruct_taps
from data_generation.src.dataset.four_data_protocol import MODEL_SCHEMA, SCHEMA, channel_config, counts, windows
from data_generation.src.receiver.dpss import SINGLE_SYMBOL_DOMAIN
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.single_symbol import SymbolDPSSModel, SymbolPilotDesign, symbol_basis
from data_generation.src.transmitter.ofdm import carrier_masks, decisions
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_delay192 import select_prior_delays
from src.evaluation.four_data_lmmse import detect
from src.evaluation.four_data_q_sweep import BlockedReceiver
from src.evaluation.ncs1_delay_support import full_nmse


CONFIG = "configs/evaluation/four_data_single_symbol.json"
CODE = (CONFIG, "src/evaluation/four_data_single_symbol.py", "scripts/evaluate/four_data_single_symbol.py",
        "data_generation/src/receiver/single_symbol.py", "data_generation/src/receiver/estimation.py",
        "data_generation/src/receiver/pdd.py")


def load_options(path=None, *, smoke=False):
    options = json.loads(Path(path or ROOT/CONFIG).read_text(encoding="utf-8"))
    specs = options["temporal_orders"]
    delays, floors = options["delay_counts"], options["model_error_ratios"]
    if (options["channels"] != ["NOF1", "NCS1"] or not specs or not delays or not floors
            or len({band for band, _ in specs}) != len(specs)
            or any(not orders or len(set(orders)) != len(orders) for _, orders in specs)
            or sorted(set(delays)) != delays or any(type(k) is not int or not 1 <= k <= 1024 for k in delays)
            or len(set(floors)) != len(floors) or any(not 0 <= ratio <= 1 for ratio in floors)
            or type(options["ber_frames_per_channel"]) is not int or options["ber_frames_per_channel"] < 1):
        raise ValueError("Invalid single-symbol search grid")
    for band, orders in specs:
        for order in orders:
            symbol_basis(1024, 6000, band, order)
    if smoke:
        options.update(temporal_orders=[[0, [1]], [12, [2, 4]]], delay_counts=[32, 128],
                       model_error_ratios=[0, .03], ber_frames_per_channel=1)
    return options


def db(linear):
    return 10*np.log10(np.maximum(linear, 1e-300))


def fit_independent_prior(c, archive, bases, smoke):
    records = c["prior_records"][:2] if smoke else c["prior_records"]
    starts = c["starts_seconds"][:1] if smoke else c["starts_seconds"]
    receiver = BlockedReceiver(c)
    transform = np.concatenate(list(bases.values()), axis=1).conj().T.copy()
    projections, powers, identities = [], [], []
    for record in records:
        sounding = archive.load(record)
        for start in starts:
            for symbol in range(4):
                truth, _ = receiver.reference(sounding, start, symbol)
                projections.append(transform @ truth)
                powers.append(np.mean(abs(truth)**2, axis=0))
            identities.append(dict(record=record, start_seconds=start, sounding_sha256=sounding.sha256))
        print(f"{c['channel']}: prior recording {record}, {len(projections)} symbols fitted", flush=True)
    offsets = np.cumsum([basis.shape[1] for basis in bases.values()])
    coefficients = dict(zip(bases, np.split(np.stack(projections), offsets[:-1], axis=1), strict=True))
    return coefficients, np.stack(powers), records, identities


def dataset_manifest(directory, c, archive_sha):
    source = directory/"manifest.json"
    accepted = json.loads(source.read_text(encoding="utf-8"))
    with h5py.File(directory/"model.h5", "r") as model:
        if (not accepted["complete"] or model.attrs["schema"] != MODEL_SCHEMA or not model.attrs["complete"]
                or model.attrs["identity"] != accepted["model_identity"]
                or json.loads(model.attrs["config_json"]) != c
                or json.loads(model.attrs["provenance_json"])["archive_sha256"] != archive_sha):
            raise ValueError(f"Expected an accepted CP80 dataset matching this archive: {directory}")
    verify_file(directory/"model.h5", accepted)
    return accepted, file_sha256(source)


def verify_file(path, accepted):
    entry = accepted["files"][path.name]
    if path.stat().st_size != entry["bytes"] or ("sha256" in entry and file_sha256(path) != entry["sha256"]):
        raise ValueError(f"Dataset differs from the accepted manifest: {path}")


def verify_split(h, c, split, accepted):
    plan = windows(c, split)
    repeats = c["learning_dataset"]["repeats"][split]
    if (h.attrs["schema"] != SCHEMA or h.attrs["config_hash"] != digest(c) or h.attrs["split"] != split
            or h.attrs["model_identity"] != accepted["model_identity"] or not h.attrs["complete"]
            or not h.attrs["raw_complete"] or not np.all(h["samples/complete"][:])
            or list(zip(h["windows/record"][:], h["windows/start_seconds"][:], strict=True)) != plan
            or not np.array_equal(h["snr_db"][:], c["snr_db"])):
        raise ValueError(f"Dataset split does not match the accepted CP80 protocol: {h.filename}")
    np.testing.assert_array_equal(h["samples/window_index"][:], np.repeat(np.arange(len(plan)), repeats))
    np.testing.assert_array_equal(h["samples/repeat"][:], np.tile(np.arange(repeats), len(plan)))
    np.testing.assert_allclose(db(h["samples/raw_nmse_linear"][:].mean(axis=(1, 2))),
                               accepted["splits"][split]["mean_raw_nmse_db"], rtol=0, atol=1e-10)


def observations(h, sample_ids):
    variance = h["samples/rf_noise_variance"][:, sample_ids]
    y = h["samples/clean_observations"][sample_ids][None] + (
        np.sqrt(variance)[..., None, None]*h["samples/standard_noise"][sample_ids][None])
    return y, variance


def project_references(h, window_ids, basis, delays):
    """Project selected delays and keep energy on the entire physical support."""
    projections, energies = [], []
    transform = basis.conj().T.copy()
    for wi in window_ids:
        truth = h["windows/reference_taps"][wi].astype(np.complex128)
        projections.append(transform @ truth[..., delays])
        energies.append(np.sum(abs(truth)**2, axis=(-1, -2)))
    energy = np.stack(energies)
    if np.any(energy <= 0) or not np.all(np.isfinite(energy)):
        raise ValueError("Invalid physical reference energy")
    return np.stack(projections), energy


def estimate_frames(estimators, y, variance, floor):
    ns, nf, _, n = y.shape
    return np.stack([estimator.estimate_coefficients(y[:, :, s].reshape(-1, n), variance.ravel(), floor)
                     .reshape(ns, nf, estimator.model.count, estimator.model.length)
                     for s, estimator in enumerate(estimators)], axis=2)


def projection_nmse(estimate, projection, energy):
    error = np.sum(abs(estimate-projection[None])**2, axis=(-1, -2))
    omitted = np.maximum(energy-np.sum(abs(projection)**2, axis=(-1, -2)), 0)
    return (error+omitted[None])/energy[None]


def fit_model(c, basis, delays, coefficients, records):
    model = SymbolDPSSModel(c, basis, delays)
    return FrozenPrior.fit_coefficients(model, coefficients, 0, 1, records, c["relative_regularizer"],
        fit_domain=SINGLE_SYMBOL_DOMAIN, delay_block_size=c["delay_block_size"])


def write_csv(path, rows):
    partial = path.with_suffix(".partial.csv")
    with partial.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def search(h, c, options, bases, prior_coefficients, powers, records, known, noise, backend, directory):
    layout = carrier_masks(c)
    supports = {k: select_prior_delays(powers, k) for k in options["delay_counts"]}
    longest = supports[max(supports)]
    ids = h["samples/window_index"][:]
    y, variance = observations(h, np.arange(len(ids)))
    packed = np.concatenate(list(bases.values()), axis=1)
    projections, energy = project_references(h, np.unique(ids), packed, longest)
    columns = {k: np.searchsorted(longest, delays) for k, delays in supports.items()}
    reference_slices = np.split(projections, np.cumsum([v.shape[1] for v in bases.values()])[:-1], axis=-2)
    trials = []
    progress = time.perf_counter()
    for (band, orders), references in zip(options["temporal_orders"], reference_slices, strict=True):
        design = SymbolPilotDesign(bases[band], longest, known, layout, noise, backend)
        for q in orders:
            for k, delays in supports.items():
                prior = fit_model(c, bases[band][:, :q], delays, prior_coefficients[band][:, :q, delays], records)
                estimators = design.build(prior)
                projection = references[ids, :, :q][..., columns[k]]
                for floor in options["model_error_ratios"]:
                    coeff = estimate_frames(estimators, y, variance, floor)
                    score = projection_nmse(coeff, projection, energy[ids]).mean(axis=(1, 2))
                    trials.append(dict(bandwidth_hz=band, q=q, delay_count=k, model_error_ratio=floor,
                        validation_nmse_linear=float(score.mean()), validation_nmse_db=float(db(score.mean())),
                        **{f"nmse_db_{snr}dB": value for snr, value in zip(c["snr_db"], db(score), strict=True)}))
                del estimators
        del design
        write_csv(directory/"validation_search.csv", trials)
        print(f"{c['channel']}: validated {len(trials)} settings through {band} Hz ({time.perf_counter()-progress:.1f}s)", flush=True)
    best = min(trials, key=lambda item: (item["validation_nmse_linear"], item["q"]*item["delay_count"],
                                        item["bandwidth_hz"], item["model_error_ratio"]))
    delays = supports[best["delay_count"]]
    return best, delays, len(trials)


def evaluate_test(h, c, options, prior, known, noise, model, backend, floor, directory):
    masks = carrier_masks(c)
    basis, delays = prior.model.basis, prior.model.delays
    estimators = SymbolPilotDesign(basis, delays, known, masks, noise, backend).build(prior)
    baseline = h["samples/raw_nmse_linear"][:]
    new = np.empty_like(baseline)
    ids = h["samples/window_index"][:]
    n_frames, n_snr = len(ids), len(c["snr_db"])
    diag_ids = np.linspace(0, n_frames-1, min(n_frames, options["ber_frames_per_channel"]), dtype=int)
    error_counts = np.zeros((2, 2, n_snr), np.int64)
    audit_error = 0.
    audit_independence = False
    for first in range(0, n_frames, 24):
        sample_ids = np.arange(first, min(first+24, n_frames))
        unique, inverse = np.unique(ids[sample_ids], return_inverse=True)
        projection, energy = project_references(h, unique, basis, delays)
        y, variance = observations(h, sample_ids)
        coeff = estimate_frames(estimators, y, variance, floor)
        new[:, sample_ids] = projection_nmse(coeff, projection[inverse], energy[inverse])
        if first == 0:
            changed = y[:1, :1].copy()
            for symbol in range(4):
                changed[...] = 0
                changed[:, :, symbol] = y[:1, :1, symbol]
                isolated = estimate_frames(estimators, changed, variance[:1, :1], floor)
                np.testing.assert_allclose(isolated[0, 0, symbol], coeff[0, 0, symbol], rtol=1e-10, atol=1e-11)
            audit_independence = True
        for fi in np.intersect1d(sample_ids, diag_ids):
            local = fi-first
            selected = basis @ coeff[:, local]
            truth = h["windows/reference_taps"][ids[fi]].astype(np.complex128)
            direct = full_nmse(selected, truth, delays)
            audit_error = max(audit_error, float(np.max(abs(direct-new[:, fi]))))
            np.testing.assert_allclose(direct, new[:, fi], rtol=1e-10, atol=1e-11)
            taps = np.zeros(selected.shape[:-1]+(int(delays[-1])+1,), complex)
            taps[..., delays] = selected
            joint = reconstruct_taps(model["sampled_basis"], h["samples/raw_coefficients"][:, fi], model["delays"])
            bits = h["samples/bits"][fi]
            for method, channel_taps in enumerate((taps, joint)):
                symbols = detect(channel_taps, y[:, local], known, variance[:, local], noise, masks["data"], backend)
                error_counts[method] += np.count_nonzero(decisions(np.array(symbols)) != bits, axis=(2, 3, 4))
        if first == 0 or first+24 >= n_frames or (first+24) % 120 == 0:
            print(f"{c['channel']}: scored {min(first+24, n_frames)}/{n_frames} test frames per SNR", flush=True)
    if not np.all(np.isfinite(new)):
        raise ValueError("Nonfinite held-out CSI error")
    bits_per_snr = len(diag_ids)*4*c["data_count"]*2
    np.savez_compressed(directory/"paired_test_nmse.npz", single_symbol_nmse_linear=new, joint_nmse_linear=baseline,
        record=h["windows/record"][:][ids], start_seconds=h["windows/start_seconds"][:][ids],
        repeat=h["samples/repeat"][:], snr_db=c["snr_db"], ber_diagnostic_frames=diag_ids)
    return dict(snr_db=c["snr_db"], frames_per_snr=n_frames, symbols_per_snr=n_frames*4,
        single_symbol_nmse_db=db(new.mean(axis=(1, 2))).tolist(), joint_nmse_db=db(baseline.mean(axis=(1, 2))).tolist(),
        single_symbol_nmse_db_by_position=db(new.mean(axis=1)).tolist(), joint_nmse_db_by_position=db(baseline.mean(axis=1)).tolist(),
        single_symbol_all_snr_nmse_db=float(db(new.mean())), joint_all_snr_nmse_db=float(db(baseline.mean())),
        diagnostic_ber_frames=diag_ids.tolist(), diagnostic_bits_per_snr=bits_per_snr,
        bit_errors=error_counts.tolist(), ber=(error_counts/bits_per_snr).tolist(),
        checks=dict(independent_observations=audit_independence, max_direct_nmse_difference=audit_error,
                    paired_full_support=True, baseline_matches_accepted_raw_nmse=True))


def run_channel(directory, data, archive_path, archive_sha, c, options, backend, smoke):
    channel = c["channel"]
    accepted, source_sha = dataset_manifest(data, c, archive_sha)
    with h5py.File(data/"model.h5", "r") as h:
        known, noise = h["known"][:], h["unit_rf_noise_covariance"][:]
        joint = dict(sampled_basis=h["sampled_basis"][:], delays=h["delay_indices"][:])
        np.testing.assert_array_equal(joint["delays"], accepted["delay_indices"])
        if array_sha256(h["basis"][:]) != accepted["basis_sha256"]:
            raise ValueError("The paired joint baseline uses a different time basis")
    bases = {band: symbol_basis(c["nfft"], c["fs_baseband"], band, max(orders))
             for band, orders in options["temporal_orders"]}
    for basis in bases.values():
        np.testing.assert_allclose(basis.conj().T @ basis, np.eye(basis.shape[1]), rtol=0, atol=1e-12)
    archive = WatermarkArchive(archive_path, channel)
    coefficients, powers, records, prior_ids = fit_independent_prior(c, archive, bases, smoke)
    verify_file(data/"validation.h5", accepted)
    with h5py.File(data/"validation.h5", "r") as h:
        verify_split(h, c, "validation", accepted)
        best, delays, evaluated = search(h, c, options, bases, coefficients, powers, records,
                                          known, noise, backend, directory)
    q, band = best["q"], best["bandwidth_hz"]
    prior = fit_model(c, bases[band][:, :q], delays, coefficients[band][:, :q, delays], records)
    selection = dict(best=best, evaluated_settings=evaluated, snr_db=c["snr_db"],
        validation_frames_per_snr=counts(c)["validation"]["frames_per_snr"],
        prior_symbols=len(powers), prior_windows=prior_ids, delay_indices=delays.tolist(),
        basis_sha256=array_sha256(prior.model.basis), prior_sha256=array_sha256(prior.coefficients),
        prior_mean_delay_energy=powers.mean(axis=0).tolist(),
        prior_delay_coverage=float(powers[:, delays].sum()/powers.sum()),
        validation_joint_nmse_db=accepted["splits"]["validation"]["mean_raw_nmse_db"],
        source_model_identity=accepted["model_identity"], source_manifest_sha256=source_sha,
        source_files={name: accepted["files"][name] for name in ("model.h5", "validation.h5", "test.h5")})
    write_json(directory/"selection.json", selection)
    print(f"{channel}: selected {band} Hz / Q={q} / K={len(delays)} / floor={best['model_error_ratio']:g}, "
          f"validation NMSE {best['validation_nmse_db']:.3f} dB", flush=True)
    verify_file(data/"test.h5", accepted)
    with h5py.File(data/"test.h5", "r") as h:
        verify_split(h, c, "test", accepted)
        test = evaluate_test(h, c, options, prior, known, noise, joint, backend, best["model_error_ratio"], directory)
    write_json(directory/"test.json", test)
    return dict(selection=selection, test=test)


def finish(output, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = []
    nmse_fig, nmse_axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained", sharey=True)
    ber_fig, ber_axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained", sharey=True)
    for col, (channel, report) in enumerate(reports.items()):
        trial, test = report["selection"]["best"], report["test"]
        snr, ber = test["snr_db"], np.asarray(test["ber"])
        estimated, joint = np.array([test["single_symbol_nmse_db"], test["joint_nmse_db"]])
        for i, s in enumerate(snr):
            tables.append(dict(channel=channel, snr_db=s, frames_per_snr=test["frames_per_snr"],
                symbols_per_snr=test["symbols_per_snr"], bandwidth_hz=trial["bandwidth_hz"], q=trial["q"],
                delay_count=trial["delay_count"], model_error_ratio=trial["model_error_ratio"],
                single_symbol_nmse_db=estimated[i], joint_nmse_db=joint[i], nmse_delta_db=estimated[i]-joint[i],
                diagnostic_bits_per_snr=test["diagnostic_bits_per_snr"], single_symbol_single_tap_ber=ber[0, 0, i],
                joint_single_tap_ber=ber[1, 0, i], single_symbol_full_matrix_ber=ber[0, 1, i], joint_full_matrix_ber=ber[1, 1, i]))
        ax_nmse, ax_ber = nmse_axes[col], ber_axes[col]
        ax_nmse.plot(snr, estimated, "o-", color="#176B78", label="Single-symbol CSI")
        ax_nmse.plot(snr, joint, "s--", color="#9B522A", label="Joint-frame CSI")
        ax_nmse.set_title(f"{channel} · {test['symbols_per_snr']:,} symbols/SNR")
        ax_nmse.set_ylabel("Mean full-matrix CSI NMSE (dB)")
        minimum = .5/test["diagnostic_bits_per_snr"]
        for method, (name, line) in enumerate((("Single-symbol", "-"), ("Joint-frame", "--"))):
            for detector, (algorithm, color, marker) in enumerate((("Single-tap", "#9B522A", "o"), ("Full-matrix", "#176B78", "s"))):
                curve = ber[method, detector]
                ax_ber.semilogy(snr, np.maximum(curve, minimum), color=color, marker=marker, linestyle=line,
                                label=f"{algorithm} / {name}")
                zero = curve == 0
                ax_ber.scatter(np.asarray(snr)[zero], np.full(zero.sum(), minimum), marker=marker,
                               facecolor="white", edgecolor=color, zorder=5)
        ax_ber.set_title(f"{channel} · {test['diagnostic_bits_per_snr']:,} bits/SNR")
        ax_ber.set_ylabel("Bit error rate")
        for ax in (ax_nmse, ax_ber):
            ax.set(xlabel="SNR (dB)", xticks=snr)
            ax.grid(alpha=.23, which="both")
            ax.legend(fontsize=8)
    nmse_fig.suptitle("Held-out channel estimation · raw CSI · CP 80 ms")
    ber_fig.suptitle("Paired LMMSE detection · diagnostic subset")
    for fig, name in ((nmse_fig, "nmse_snr"), (ber_fig, "ber_snr")):
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{name}.{ext}", dpi=170)
        plt.close(fig)
    write_csv(output/"summary.csv", tables)


def run(output, data, archive_path, backend, options, *, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    archive_sha = file_sha256(archive_path)
    metadata = dict(schema="cp80-single-symbol-csi-v1", smoke=smoke, options=options, device=backend.device,
        archive_sha256=archive_sha, dataset_directory=str(data),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={name: file_sha256(ROOT/name) for name in CODE},
        observations="Current-symbol pilot and protection carriers; model-error variance uses current-symbol power",
        noise_variance="Identical injected RF variance supplied as benchmark metadata to both estimators",
        selection="Mean linear full-matrix NMSE across validation frames, symbols and SNR; ties prefer fewer coefficients",
        prior="Independent physical FFT windows; energy-ranked physical delays and 32-sample local covariance blocks",
        reference_use="Physical prior fitting and held-out scoring only; raw CSI without reference assistance",
        ber_order=dict(csi=["single_symbol", "joint"], equalizer=["single_tap_lmmse", "full_matrix_lmmse"]))
    manifest = output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output belongs to a different single-symbol evaluation")
    write_json(manifest, metadata)
    reports = {}
    for channel in options["channels"]:
        directory = output/channel.lower()
        directory.mkdir(exist_ok=True)
        c = channel_config(channel, smoke)
        reports[channel] = run_channel(directory, data/(channel.lower()+"_dddd"), archive_path, archive_sha,
                                       c, options, backend, smoke)
    finish(output, reports)
    write_json(output/"metrics.json", dict(complete=True, channels=reports, wall_seconds=time.perf_counter()-began))
    print(f"Completed paired single-symbol CSI evaluation: {output}", flush=True)
    return reports
