"""Paired CSI-estimator comparison on the protected-pilot diagnostic frames."""
from __future__ import annotations

from functools import partial
import json
import subprocess
import time

import numpy as np
from scipy.linalg import pinv

from data_generation.src.config import ROOT
from data_generation.src.dataset.four_data_math import reference_statistics
from data_generation.src.dataset.four_data_physical import window_results
from data_generation.src.dataset.four_data_protocol import windows
from data_generation.src.transmitter.ofdm import carrier_masks, decisions
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_edge_comb import allocation, replay_observations
from src.evaluation.four_data_lmmse import detect
from src.evaluation.four_data_protected_pilots import (
    accepted_split, cluster_config, load_options, prepare, prepare_design, replay_cluster_window, replay_pool,
)
from src.evaluation.four_data_single_symbol import db, estimate_frames, projection_nmse, write_csv


ACCEPTED = "outputs/experiment/four_data_protected_pilots_20261126"
METHODS = ("ls", "lmmse")


def minimum_norm_inverses(responses):
    """Take the Moore-Penrose inverse of each pattern's selected response."""
    n_patterns, rows, q, k = responses.shape
    matrices = responses.reshape(n_patterns, rows, q*k)
    factors = [pinv(matrix, return_rank=True) for matrix in matrices]
    inverses, ranks = zip(*factors, strict=True)
    return matrices, np.stack(inverses), [int(rank) for rank in ranks]


def least_squares(design, y, matrices, inverses):
    """Apply the corresponding independent symbol inverse to each sample."""
    ns, nf, patterns, _ = y.shape
    q, k = design.basis.shape[1], len(design.delays)
    selected = y[..., design.rows].transpose(2, 0, 1, 3).reshape(patterns, ns*nf, len(design.rows))
    fitted = selected @ inverses.swapaxes(-1, -2)
    residual = selected-fitted @ matrices.swapaxes(-1, -2)
    relative = np.sum(abs(residual)**2)/max(float(np.sum(abs(selected)**2)), 1e-300)
    return fitted.reshape(patterns, ns, nf, q, k).transpose(1, 2, 0, 3, 4), float(relative)


def accepted_frames(source, channel, plan, repeats, snr):
    root = source/channel.lower()
    accepted = json.loads((root/"test.json").read_text(encoding="utf-8"))
    with np.load(root/"paired_test_nmse.npz") as h:
        ids = h["ber_diagnostic_frames"]
        nmse = h["protected_nmse_linear"][:, ids]
        records, starts = zip(*plan, strict=True)
        np.testing.assert_array_equal(h["record"], np.repeat(records, repeats))
        np.testing.assert_array_equal(h["start_seconds"], np.repeat(starts, repeats))
        np.testing.assert_array_equal(h["repeat"], np.tile(np.arange(repeats), len(plan)))
        np.testing.assert_array_equal(h["snr_db"], snr)
    np.testing.assert_array_equal(ids, accepted["test"]["diagnostic_ber_frames"])
    return ids, nmse, np.asarray(accepted["test"]["bit_errors"][1]), accepted["ranks"]


def compare_channel(channel, archive, archive_sha, data, accepted, accepted_layout, options, backend, workers, smoke):
    study = prepare(channel, data, archive, archive_sha, options, smoke)
    spec = options["candidates"][0]
    c = cluster_config(study.config, spec)
    if accepted_layout is not None and allocation(c) != accepted_layout:
        raise ValueError("The accepted diagnostic used a different carrier allocation")
    prior, known, design = prepare_design(study, c, backend)
    lmmse = design.build(prior)
    matrices, inverses, ranks = minimum_norm_inverses(backend.numpy(design.responses))
    repeats = c["learning_dataset"]["repeats"]["test"]
    plan, snr = windows(c, "test"), c["snr_db"]
    if smoke:
        ids = np.array([0])
    else:
        ids, baseline_nmse, baseline_bits, accepted_rank = accepted_frames(accepted, channel, plan, repeats, snr)
        np.testing.assert_array_equal(ranks, accepted_rank)
    jobs = [(int(wi), "test", *plan[wi], repeats) for wi in np.unique(ids//repeats)]
    nmse = np.full((2, len(snr), len(ids), 4), np.nan)
    errors = np.zeros((2, len(snr)), dtype=np.int64)
    max_ls_residual = 0.
    model = prior.model
    basis, delays = model.basis, model.delays
    sampled = np.broadcast_to(basis, (4, *basis.shape))
    mask = carrier_masks(c)["data"]
    noise = study.receiver.chain.noise_covariance
    with accepted_split(study, "test") as old, replay_pool(study, workers) as pool:
        source = window_results(jobs, study.config, study.archive, study.receiver, pool=pool, prefetch=2*workers,
                                compute=partial(replay_cluster_window, spec=spec))
        for wi, packet in source:
            begin, end = wi*repeats, (wi+1)*repeats
            take = np.flatnonzero((ids >= begin) & (ids < end))
            local = ids[take]-begin
            y, variances, bits, changed = replay_observations(packet, old, slice(begin, end))
            if old is not None:
                if changed != repeats:
                    raise ValueError("Expected each waveform to be replayed on the protected pilot layout")
                truth = old["windows/reference_taps"][wi]
            else:
                record, start = plan[wi]
                sounding = study.archive.load(record)
                truth = np.stack([study.receiver.reference(sounding, start, s)[0] for s in range(4)]).astype(np.complex64)
            projection, energy = reference_statistics(sampled, truth, delays)
            y, variances, bits = y[:, local], variances[:, local], bits[local]
            ls, residual = least_squares(design, y, matrices, inverses)
            max_ls_residual = max(max_ls_residual, residual)
            estimate = estimate_frames(lmmse, y, variances, options["model_error_ratio"])
            for method, coeff in enumerate((ls, estimate)):
                nmse[method][:, take] = projection_nmse(coeff, projection, energy)
                for frame in range(len(take)):
                    selected = basis @ coeff[:, frame]
                    taps = np.zeros(selected.shape[:-1]+(int(delays[-1])+1,), complex)
                    taps[..., delays] = selected
                    _, full = detect(taps, y[:, frame], known, variances[:, frame], noise, mask, backend)
                    errors[method] += np.count_nonzero(decisions(full) != bits[frame], axis=(1, 2, 3))
            print(f"{channel}: compared {int(take[-1])+1}/{len(ids)} paired frames", flush=True)
    if not np.all(np.isfinite(nmse)) or (min(ranks) == len(design.rows) and max_ls_residual > 1e-20):
        raise ValueError("Expected complete paired estimates and a near-zero full-row-rank LS residual")
    if not smoke:
        np.testing.assert_allclose(nmse[1], baseline_nmse, rtol=1e-10, atol=1e-11)
        np.testing.assert_array_equal(errors[1], baseline_bits)
    bits_per_snr = len(ids)*4*c["data_count"]*2
    frames = [dict(index=int(i), record=plan[i//repeats][0], start_seconds=plan[i//repeats][1], repeat=int(i%repeats)) for i in ids]
    return dict(snr_db=snr, frames=frames, symbols_per_snr=len(ids)*4, bits_per_snr=bits_per_snr,
        observations=len(design.rows), coefficients=model.count*model.length, rank_by_symbol=ranks,
        max_ls_relative_observation_residual=max_ls_residual, accepted_lmmse_match=not smoke,
        per_symbol_nmse_linear=nmse.tolist(), methods={method:dict(mean_nmse_db=db(nmse[i].mean(axis=(1, 2))).tolist(),
            all_snr_nmse_db=float(db(nmse[i].mean())), bit_errors=errors[i].tolist(), ber=(errors[i]/bits_per_snr).tolist())
            for i, method in enumerate(METHODS)})


def finish(output, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for channel, report in reports.items():
        methods = report["methods"]
        for i, snr in enumerate(report["snr_db"]):
            values = {f"{key}_nmse_db": methods[key]["mean_nmse_db"][i] for key in METHODS}
            rows.append(dict(channel=channel, snr_db=snr, symbols_per_snr=report["symbols_per_snr"],
                bits_per_snr=report["bits_per_snr"], **values,
                ls_minus_lmmse_nmse_db=values["ls_nmse_db"]-values["lmmse_nmse_db"],
                **{f"{key}_full_matrix_ber": methods[key]["ber"][i] for key in METHODS}))
    write_csv(output/"summary.csv", rows)
    for metric, field, ylabel, title in (
            ("nmse", "mean_nmse_db", "Mean full-matrix CSI NMSE (dB)", "Single-symbol DPSS-BEM estimation"),
            ("ber", "ber", "Bit error rate", "Same full-matrix LMMSE equalizer for both CSI estimates")):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained", sharey=True)
        for ax, (channel, report) in zip(axes, reports.items(), strict=True):
            snr = report["snr_db"]
            for method, color, marker, style in (("ls", "#A45534", "s", "--"), ("lmmse", "#146C7C", "o", "-")):
                values = np.asarray(report["methods"][method][field])
                if metric == "ber":
                    lower = .5/report["bits_per_snr"]
                    ax.semilogy(snr, np.maximum(values, lower), linestyle=style, marker=marker, color=color,
                                label=f"{method.upper()} CSI")
                    zero = values == 0
                    ax.scatter(np.asarray(snr)[zero], np.full(zero.sum(), lower), marker=marker, facecolors="white",
                               edgecolors=color, zorder=5)
                else:
                    ax.plot(snr, values, linestyle=style, marker=marker, color=color, label=f"{method.upper()} CSI")
            ax.set(title=f"{channel} · {report['symbols_per_snr']} symbols/SNR", xlabel="SNR (dB)", ylabel=ylabel, xticks=snr)
            ax.legend(fontsize=8)
            ax.grid(alpha=.23, which="both")
        fig.suptitle(title)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{metric}_snr.{ext}", dpi=170)
        plt.close(fig)


def run(output, data, archive, accepted, backend, *, workers=1, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    options = load_options(smoke=smoke)
    archive_sha = file_sha256(archive)
    accepted_layout = accepted_sha = None
    if not smoke:
        path = accepted/"manifest.json"
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["archive_sha256"] != archive_sha or any(previous["options"][key] != options[key]
                for key in ("bandwidth_hz", "q", "delay_count", "model_error_ratio")):
            raise ValueError("Use the accepted archive and matching single-symbol settings")
        accepted_sha = file_sha256(path)
        accepted_layout = json.loads((accepted/"carrier_allocation.json").read_text(encoding="utf-8"))
    manifest = dict(schema="paired-single-symbol-ls-lmmse-v1", smoke=smoke, options=options, device=backend.device,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        archive_sha256=archive_sha, accepted_manifest_sha256=accepted_sha,
        frames="The accepted protected-pilot run's 16 diagnostic frames per channel",
        ls="SVD Moore-Penrose minimum coefficient norm; identical pilot and in-band null observation rows",
        lmmse="Frozen independent prior; model-error floor and colored interference and thermal covariance",
        metrics="Ratio of squared full-matrix errors per symbol, linear mean then dB; shared full-matrix LMMSE detection")
    path = output/"manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output belongs to a different LS/LMMSE comparison")
    write_json(path, manifest)
    reports = {channel:compare_channel(channel, archive, archive_sha, data, accepted, accepted_layout,
        options, backend, workers, smoke) for channel in options["channels"]}
    finish(output, reports)
    write_json(output/"metrics.json", dict(complete=True, channels=reports, wall_seconds=time.perf_counter()-begin))
    return reports
