"""Causal CSI-error, thermal-noise and covariance audit of the saved 4D run."""
from __future__ import annotations

import json
from copy import copy
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy import linalg

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT
from data_generation.src.receiver.chain import frequency_matrix, resample_operator, taps_to_matrix
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.four_data import BlockedFourDataReceiver, FourDataEstimator
from data_generation.src.receiver.four_data_rotated import waveform
from data_generation.src.transmitter.ofdm import Frame, decisions, fft, qpsk, receive_symbols, window_indices
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.evaluation.four_data_mmse import ALGORITHMS, experiment_config


def coefficient_power(basis, coefficients):
    """Squared full-CSI norm per D, using unitary Fourier invariance."""
    return np.stack([np.einsum("...ql,qr,...rl->...", coefficients.conj(),
                     v.conj().T @ v, coefficients, optimize=True).real for v in basis], axis=-1)


def coefficient_nmse(basis, coefficients, cross, energy):
    power = coefficient_power(basis, coefficients)
    correlation = np.stack([np.einsum("...ql,ql->...", coefficients.conj(), x,
                                     optimize=True).real for x in cross], axis=-1)
    return np.maximum((power-2*correlation+energy)/energy, 0)


def thermal_mode_energy(estimator, basis):
    """Exact per-D CSI error energy of each noise-whitened gain column."""
    b = estimator.backend
    q, length = estimator.prior.model.count, estimator.prior.model.length
    gain = estimator.gain.reshape(q, length, -1)
    result = []
    for v in basis:
        weighted = (b.array(np.asarray(v.conj().T @ v, dtype=complex)) @ gain.reshape(q, -1)).reshape(gain.shape)
        product = (gain.conj()*weighted).real
        result.append(b.numpy(product.sum(dim=(0, 1)) if b.torch else product.sum(axis=(0, 1))))
    return np.maximum(np.stack(result), 0)


def exact_noise_audit(receiver, config, recorded):
    """Independent finite-RF-operator covariance; retain mixing pseudocovariance."""
    chain = receiver.chain
    down = resample_operator(len(chain.phase_carrier), 1, 16, config)
    first = down[window_indices(config, 0)]
    time_covariance = (4*(first @ first.T)).toarray()
    exact = frequency_matrix(time_covariance)
    other = down[window_indices(config, 1)]
    cross = (4*(first @ other.T)).toarray()
    phase = np.exp(-4j*np.pi*config["carrier_hz"]*np.arange(down.shape[1])/config["fs_passband"])
    pseudo = (4*(first.multiply(phase) @ first.T)).toarray()
    return dict(covariance_relative_frobenius=float(np.linalg.norm(exact-recorded)/np.linalg.norm(exact)),
                covariance_max_absolute_error=float(np.max(abs(exact-recorded))),
                exact_noise_gain=float(np.trace(exact).real/config["nfft"]),
                recorded_noise_gain=chain.noise_gain,
                cross_D_noise_covariance_norm=float(np.linalg.norm(cross)),
                pseudocovariance_relative_frobenius=float(np.linalg.norm(pseudo)/np.linalg.norm(time_covariance)))


def direct_solve_audit(estimator, observations, variances):
    b = estimator.backend
    moments = estimator.captured_moments
    covariance = b.array(moments["observation_covariance"])
    noise = b.array(estimator.selected_noise)
    cross = b.array(moments["cross_covariance"])
    selected = np.concatenate([observations[:, s, r] for s, r in enumerate(estimator.rows)], axis=-1)
    spectral = estimator.estimate_coefficients(observations, variances)
    errors = []
    for index in (0, 4, 6):
        weights = b.solve(covariance+(variances[index]+estimator.ridge)*noise, b.array(selected[index]))
        actual = b.numpy(cross @ weights).reshape(spectral[index].shape)
        errors.append(float(np.linalg.norm(actual-spectral[index])/np.linalg.norm(actual)))
    return dict(snr_db=[0, 20, 30], direct_vs_spectral_relative_error=errors)


def replace_data_covariance(estimator, empirical):
    """Prior-only diagnostic: keep P and A fixed, replace only R_data."""
    b, moments = estimator.backend, estimator.captured_moments
    result = copy(estimator)
    factor = linalg.cholesky(estimator.selected_noise, lower=True, check_finite=False)
    w = b.array(linalg.solve_triangular(factor, np.eye(len(factor)), lower=True, check_finite=False))
    covariance = moments["observation_covariance"]-moments["data_interference"]+empirical
    whitened = w @ b.array(covariance) @ w.conj().T
    whitened = .5*(whitened+whitened.conj().T)
    values, vectors = (b.eigh(whitened) if b.torch else linalg.eigh(whitened, check_finite=False, driver="evr"))
    result.values = b.array(np.maximum(b.numpy(values), 0))
    # Hold the original numerical ridge fixed to isolate R_data.
    result.projected = w.conj().T @ vectors
    result.gain = b.array(moments["cross_covariance"]) @ result.projected
    return result


def prior_covariance_audit(receiver, dataset, model, prior, known, masks, estimator, identities):
    """Independent-prior physical residuals; no test-fitted covariance floor."""
    n, q, length = model.c["nfft"], model.count, model.length
    basis = model.sampled_basis.reshape(4, n, q)
    model_diagonal = np.diag(estimator.captured_moments["data_interference"]).real
    empirical = np.zeros_like(estimator.captured_moments["data_interference"])
    in_space, physical = np.zeros_like(model_diagonal), np.zeros_like(model_diagonal)
    residual_pilot, residual_data = [], []
    for index, (record, start) in enumerate(identities):
        sounding = dataset.load(record)
        truth = np.stack([receiver.reference_taps(sounding, start, s)[0] for s in range(4)])
        projected = np.stack([v @ prior.coefficients[index] for v in basis])
        frame = Frame(known, known, np.zeros((4, n//4, 2), np.uint8), waveform(known, model.c), ())
        clean = receiver.replay_frame(sounding, start-model.c["guard_samples"]/model.c["fs_baseband"], frame)
        pilot_physical = receive_symbols(receiver.chain.receive(clean), model.c)
        pilot_model = fft(np.sum(projected*input_windows(known, length), axis=-1))
        pilot_residual = np.concatenate([(pilot_physical[s]-pilot_model[s])[r] for s,r in enumerate(estimator.rows)])
        residual_pilot.append(float(np.sum(abs(pilot_residual)**2)))
        offset, data_power = 0, 0.0
        for s, rows in enumerate(estimator.rows):
            indices = np.flatnonzero(masks["data"][s])
            full = taps_to_matrix(truth[s])[np.ix_(rows, indices)]
            limited = taps_to_matrix(projected[s])[np.ix_(rows, indices)]
            target = slice(offset, offset+len(rows))
            in_space[target] += np.sum(abs(limited)**2, axis=1)/len(identities)
            z = estimator.backend.array(limited)
            empirical[target, target] += estimator.backend.numpy(z @ z.conj().T)/len(identities)
            physical[target] += np.sum(abs(full)**2, axis=1)/len(identities)
            data_power += float(np.sum(abs(full-limited)**2))
            offset += len(rows)
        residual_data.append(data_power)
        print(f"{dataset.channel} prior covariance {index+1}/{len(identities)} {record:03d}@{start}s", flush=True)
    trace_noise = float(np.trace(estimator.selected_noise).real)
    delta = float((np.mean(residual_pilot)+np.mean(residual_data))/trace_noise)
    report = dict(modeled_data_trace=float(model_diagonal.sum()),
                empirical_in_space_data_trace=float(in_space.sum()),
                empirical_physical_self_block_data_trace=float(physical.sum()),
                modeled_vs_empirical_in_space_diagonal_relative_error=float(np.linalg.norm(model_diagonal-in_space)/np.linalg.norm(in_space)),
                modeled_vs_physical_diagonal_relative_error=float(np.linalg.norm(model_diagonal-physical)/np.linalg.norm(physical)),
                modeled_vs_empirical_in_space_covariance_relative_error=float(np.linalg.norm(estimator.captured_moments["data_interference"]-empirical)/np.linalg.norm(empirical)),
                prior_pilot_model_residual_power=float(np.mean(residual_pilot)),
                prior_data_model_residual_power=float(np.mean(residual_data)),
                equivalent_rf_residual_variance=delta,
                note="Prior-only scalar trace proxy. Pilot residual includes physical cross-block leakage; data residual trace here excludes cross-block data leakage.")
    return report, empirical


def audit_channel(channel, run, output, archive_path, backend, *, frame_limit=None):
    config = experiment_config(channel)
    report = json.loads((run/channel.lower()/"metrics.json").read_text(encoding="utf-8"))
    if not report["complete"]:
        raise ValueError("Need completed receiver experiment")
    q, length = report["parameters"]["effective_q"], report["parameters"]["length"]
    with h5py.File(run/channel.lower()/"frozen_prior.h5", "r") as saved:
        model = DPSSModel(saved["basis"][:], 12, 8.22586010451598e-6, length, config)
        coefficients = saved["coefficients"][:]
        identities = json.loads(saved.attrs["independent_records_json"])
    prior = FrozenPrior.fit_coefficients(model, coefficients, 0, 1, config["prior_records"],
                                         regularizer=1e-6, delay_block_size=32)
    basis = model.sampled_basis.reshape(4, config["nfft"], q)
    receiver, dataset = BlockedFourDataReceiver(config), WatermarkArchive(archive_path, channel)
    result = dict(channel=channel, complete=False, snr_db=config["snr_db"], frames=[],
                  source_identity=report["identity"], controls="Counterfactual observation changes affect CSI estimation only; BER detection always uses original received data")
    with h5py.File(run/channel.lower()/"receiver_estimates.h5", "r") as saved:
        known, noise = saved["known"][:], saved["unit_noise_covariance"][:]
        masks = {name:saved[f"masks/{name}"][:] for name in ("data", "tracking", "pilot", "protection")}
        estimator = FourDataEstimator(prior, known, masks, noise, backend, capture_moments=True)
        result["thermal_covariance"] = exact_noise_audit(receiver, config, noise)
        result["prior_covariance"], empirical_data = prior_covariance_audit(receiver, dataset, model, prior, known, masks, estimator, identities)
        empirical_estimator = replace_data_covariance(estimator, empirical_data)
        modes = thermal_mode_energy(estimator, basis)
        values = backend.numpy(estimator.values)
        delta = result["prior_covariance"]["equivalent_rf_residual_variance"]
        whitener = make_whitener(noise, backend)
        old_frames = report["frames"][:frame_limit]
        for index, source in enumerate(old_frames):
            record, start = source["record"], source["start_seconds"]
            group = saved[f"frames/{index:04d}"]
            gain = float(group.attrs["tx_gain"])
            clean = group["clean_observations"][:]/gain
            variances = group["rf_variances"][:]/gain**2
            standard, bits = group["standard_noise"][:], group["bits"][:]
            y = clean[None]+np.sqrt(variances[:, None, None])*standard[None]
            current = estimator.estimate_coefficients(y, variances)
            recorded = group["coefficients"][:]/gain
            reproduction = float(np.linalg.norm(current-recorded)/np.linalg.norm(recorded))
            if reproduction > 1e-9:
                raise AssertionError(f"Saved coefficient reproduction failed: {reproduction}")
            if index == 0:
                result["direct_solver"] = direct_solve_audit(estimator, y, variances)
            sounding = dataset.load(record)
            truth = np.stack([receiver.reference_taps(sounding, start, s)[0] for s in range(4)])
            energy = np.sum(abs(truth)**2, axis=(1, 2))
            cross = np.einsum("dnq,dnl->dql", basis.conj(), truth[..., :length], optimize=True)
            optimum = cross.sum(axis=0)
            projected = basis @ optimum
            data_x = np.zeros_like(known)
            for s in range(4):
                data_x[s, masks["data"][s]] = qpsk(bits[s])
            def apply(taps, x):
                return fft(np.sum(taps*input_windows(x, taps.shape[-1]), axis=-1))
            true_p, true_d = apply(truth, known), apply(truth, data_x)
            project_p, project_d = apply(projected, known), apply(projected, data_x)
            residual_p, residual_d = true_p-project_p, true_d-project_d
            physical = clean-true_p-true_d
            cases = dict(actual=current,
                remove_pilot_mismatch=estimator.estimate_coefficients(y-residual_p, variances),
                remove_data_mismatch=estimator.estimate_coefficients(y-residual_d, variances),
                remove_physical_residual=estimator.estimate_coefficients(y-physical, variances),
                matched_bem_observations=estimator.estimate_coefficients(y-residual_p-residual_d-physical, variances),
                pilot_only_bem_observations=estimator.estimate_coefficients(y-true_d-residual_p-physical, variances),
                empirical_in_space_data_covariance=empirical_estimator.estimate_coefficients(y, variances),
                thermal_floor_20db=estimator.estimate_coefficients(y, np.maximum(variances, variances[4])),
                prior_residual_floor=estimator.estimate_coefficients(y, variances+delta))
            curves = {name:coefficient_nmse(basis, c, cross, energy).mean(axis=-1).tolist() for name,c in cases.items()}
            clean_coeff = estimator.estimate_coefficients(np.broadcast_to(clean, y.shape), variances)
            clean_error = coefficient_nmse(basis, clean_coeff, cross, energy)
            denominator = values[None]+variances[:, None]+estimator.ridge
            thermal_error = variances[:, None]*(modes[None]/denominator[:, None, :]**2).sum(axis=-1)/energy
            curves["noise_averaged_expected"] = (clean_error+thermal_error).mean(axis=-1).tolist()
            curves["noise_free_input"] = clean_error.mean(axis=-1).tolist()
            curves["expected_thermal_contribution"] = thermal_error.mean(axis=-1).tolist()
            matched_clean = estimator.estimate_coefficients(np.broadcast_to(clean-residual_p-residual_d-physical, y.shape), variances)
            curves["matched_bem_noise_averaged_expected"] = (coefficient_nmse(basis, matched_clean, cross, energy)+thermal_error).mean(axis=-1).tolist()
            curves["oracle_projection"] = [float(coefficient_nmse(basis, optimum, cross, energy).mean())]*7
            effects = {}
            for name, residual in (("pilot",residual_p),("data",residual_d),("physical",physical),("in_space_data",project_d)):
                perturbation = estimator.estimate_coefficients(np.broadcast_to(residual, y.shape), variances)
                effects[name] = (coefficient_power(basis, perturbation)/energy).mean(axis=-1).tolist()
            bit_errors = {}
            decode_cases = [(name, cases[name][-1], variances[-1]) for name in
                ("actual", "thermal_floor_20db", "prior_residual_floor", "remove_pilot_mismatch", "remove_physical_residual", "matched_bem_observations", "pilot_only_bem_observations", "empirical_in_space_data_covariance")]
            decode_cases.extend((("detector_thermal_floor_20db", current[-1], variances[4]),
                                 ("detector_prior_residual_floor", current[-1], variances[-1]+delta)))
            for name, c, detector_variance in decode_cases:
                matrix = np.stack([taps_to_matrix(v @ c) for v in basis])
                errors = {}
                # EqualizerBatch has one data-index vector, so score each D
                # separately rather than incorrectly sharing staggered indices.
                for algorithm, equalizer in ALGORITHMS.items():
                    per_d = []
                    for s in range(4):
                        batch = EqualizerBatch(backend.array(matrix[s:s+1]), backend.array(y[-1,s:s+1]),
                            backend.array((y[-1,s]-matrix[s] @ known[s])[None]), np.array([detector_variance]),
                            np.flatnonzero(masks["data"][s]), noise)
                        kwargs = {"whitener":whitener} if algorithm == "full_matrix_mmse" else {}
                        detected = equalizer(batch, backend, **kwargs).symbols[0]
                        per_d.append(int(np.count_nonzero(decisions(detected) != bits[s])))
                    errors[algorithm] = per_d
                bit_errors[name] = errors
            for name, values_error in bit_errors["actual"].items():
                np.testing.assert_array_equal(values_error, source["bit_errors"][name][-1])
            selected_energy = lambda z:float(sum(np.sum(abs(z[s,r])**2) for s,r in enumerate(estimator.rows)))
            item = dict(record=record, start_seconds=start, coefficient_reproduction_error=reproduction,
                nmse_linear=curves, residual_csi_perturbation=effects, bit_errors_30db=bit_errors,
                selected_observation_power=dict(clean=selected_energy(clean), pilot_residual=selected_energy(residual_p),
                    data_residual=selected_energy(residual_d), physical_residual=selected_energy(physical)),
                rf_variances_normalized=variances.tolist())
            result["frames"].append(item)
            write_json(output/f"{channel.lower()}_audit.json", result)
            print(f"{channel} covariance controls {index+1}/{len(old_frames)} {record:03d}@{start}s", flush=True)
    names = result["frames"][0]["nmse_linear"]
    result["mean_nmse_db"] = {name:(10*np.log10(np.mean([f["nmse_linear"][name] for f in result["frames"]],axis=0))).tolist() for name in names}
    result["mean_residual_csi_perturbation_linear"] = {name:np.mean([f["residual_csi_perturbation"][name] for f in result["frames"]],axis=0).tolist() for name in ("pilot","data","physical","in_space_data")}
    result["ber_30db"] = {case:{algorithm:sum(sum(f["bit_errors_30db"][case][algorithm]) for f in result["frames"])/(len(result["frames"])*2048)
        for algorithm in ("single_tap_mmse","full_matrix_mmse")} for case in result["frames"][0]["bit_errors_30db"]}
    result["complete"] = True
    write_json(output/f"{channel.lower()}_audit.json", result)
    return result


def run_audit(run, output, archive, *, device, channels=("NOF1", "NCS1"), frame_limit=None):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    backend = DenseBackend(device)
    results = {channel:audit_channel(channel, run, output, archive, backend, frame_limit=frame_limit)
               for channel in channels}
    report = dict(complete=True, code_sha256=file_sha256(Path(__file__)), git_head=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), channels={channel:{k:v for k,v in result.items() if k != "frames"} for channel,result in results.items()})
    write_json(output/"audit.json", report)
    return report


def plot_audit(run, output, *, channels=("NOF1", "NCS1")):
    """Show measured curves and separately marked 30-dB causal controls."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, output = Path(run), Path(output)
    report = json.loads((output/"audit.json").read_text(encoding="utf-8"))
    results = {channel:report["channels"][channel] for channel in channels}
    if not report["complete"] or not all(d["complete"] for d in results.values()):
        raise ValueError("Cannot report an incomplete covariance audit")
    original = json.loads((run/"metrics.json").read_text(encoding="utf-8"))["channels"]
    fig, axes = plt.subplots(2, len(channels), squeeze=False, figsize=(6.5*len(channels), 8.6), layout="constrained")
    for col, (channel, data) in enumerate(results.items()):
        snr = data["snr_db"]
        ax = axes[0, col]
        for scheme in ("old", "new"):
            for algorithm, style, color in (("single_tap_mmse", "--", "#245a87"), ("full_matrix_mmse", "-", "#ce674b")):
                label = f"{'Original' if scheme == 'old' else 'New'} {'single-tap' if algorithm == 'single_tap_mmse' else 'full-matrix'}"
                ax.semilogy(snr, original[channel][scheme]["algorithms"][algorithm]["ber"], style,
                            color="#8c9198" if scheme == "old" else color, marker="o", markersize=3, label=label)
                if scheme == "new":
                    ax.semilogy([30], [data["ber_30db"]["prior_residual_floor"][algorithm]], "*", color=color,
                                markersize=11, label=f"Prior-only floor, {'single' if algorithm == 'single_tap_mmse' else 'full'} (30 dB)")
        ax.set(title=f"{channel}: BER-SNR", ylabel="Bit error rate", xlabel="SNR (dB)")
        ax.legend(fontsize=7.2)
        ax = axes[1, col]
        for name, label, color, style in (
            ("actual", "Actual receiver CSI", "#24282d", "-"),
            ("noise_averaged_expected", "Exact thermal-noise expectation", "#24282d", ":"),
            ("remove_pilot_mismatch", "Remove pilot BEM residual (oracle)", "#245a87", "--"),
            ("matched_bem_observations", "Remove all model residuals (oracle)", "#278560", "-"),
            ("empirical_in_space_data_covariance", "Prior empirical in-space data covariance", "#9154a3", ":"),
            ("prior_residual_floor", "Prior-only residual variance floor", "#ce674b", "-")):
            ax.plot(snr, data["mean_nmse_db"][name], style, color=color, marker="o", markersize=3, label=label)
        ax.set(title=f"{channel}: CSI-error controls", ylabel="Mean channel NMSE (dB)", xlabel="SNR (dB)")
        ax.legend(fontsize=7.2)
        for ax in axes[:, col]:
            ax.set_xticks(snr)
            ax.grid(True, which="both", alpha=.2)
    fig.suptitle("12 Hz, no P, 4D: high-SNR covariance audit\nSame saved observations; oracle changes apply only to CSI estimation", fontsize=12)
    fig.supxlabel("BER controls shown only at measured 30-dB points; original BER curves retained\nPrior-only floor is a diagnostic control; original experiment is unchanged", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"covariance_audit.{suffix}", dpi=180)
    plt.close(fig)
    with (output/"audit_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("channel", "control", "snr_db", "mean_nmse_db", "single_tap_ber", "full_matrix_ber"))
        for channel, data in results.items():
            for name, curve in data["mean_nmse_db"].items():
                for snr, nmse in zip(data["snr_db"], curve, strict=True):
                    ber = data["ber_30db"].get(name, {}) if snr == 30 else {}
                    writer.writerow((channel, name, snr, nmse, ber.get("single_tap_mmse", ""), ber.get("full_matrix_mmse", "")))
