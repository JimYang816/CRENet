"""Isolated receiver-only DPSS group OMP versus joint LMMSE experiment."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace
import time

import h5py
import numpy as np

from data_generation.src.dataset.compact_math import matrix_to_taps
from data_generation.src.dataset.compact_protocol import sample_identities, validate_plan
from data_generation.src.receiver.estimation import input_windows
from data_generation.src.receiver.pdd import JointEstimator
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.transmitter.ofdm import decisions
from src.algorithms.dpss_bomp.estimator import GroupOMP, build_dictionary, whiten_observations
from src.algorithms.full_matrix_lmmse.equalizer import equalize_batch
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256, write_json
from src.evaluation.fingerprints import DataIdentity, digest, numerical_environment
from src.evaluation.four_data_q_sweep import time_noise_covariance
from src.evaluation.delay_sparsity import _identity_check
from src.data.compact import CompactDataset

LOG = logging.getLogger(__name__)
ALGORITHMS = ("joint_lmmse", "group_omp")


def restore_receiver_lmmse(ds, backend):
    """Restore frozen prior factors and recompute coefficients from observations."""
    h = ds._shared
    estimator = JointEstimator.__new__(JointEstimator)
    estimator.prior = SimpleNamespace(model=ds._model)
    estimator.backend = backend
    estimator.model_error_ratio = float(ds._config.get("csi_model_error_ratio", 0))
    estimator.known = ds._known.copy()
    estimator.masks = {k: v.copy() for k, v in ds._masks.items()}
    estimator.noise_gain = float(h["estimator"].attrs["noise_gain"])
    estimator.ridge = float(h["estimator"].attrs["ridge"])
    estimator.rows = [h[f"estimator/rows/{s}"][:] for s in range(4)]
    estimator.offsets = h["estimator/offsets"][:]
    estimator.values = backend.array(h["estimator/values"][:])
    estimator.projected = backend.array(h["estimator/projected"][:])
    estimator.gain = backend.array(h["estimator/gain"][:])
    return estimator


class ReferenceScore:
    """Full-matrix NMSE using Gram identities; only used after prediction."""
    def __init__(self, model, reference):
        self.basis = model.sampled_basis.reshape(4, model.c["nfft"], model.count)[[1, 2]]
        truth = matrix_to_taps(np.asarray(reference, dtype=np.complex128))
        self.energy = np.sum(abs(truth)**2, axis=(1, 2))
        if np.any(self.energy <= 0):
            raise ValueError("Zero-energy reference matrix")
        self.gram = np.stack([x.conj().T @ x for x in self.basis])
        self.cross = np.stack([self.basis[d].conj().T @ truth[d, :, :model.length] for d in range(2)])

    def nmse(self, coefficients):
        c = np.asarray(coefficients, dtype=np.complex128)
        if c.ndim != 3 or c.shape[1:] != self.cross.shape[1:]:
            raise ValueError("Unexpected coefficient shape")
        result = []
        for d in range(2):
            transformed = np.einsum("qr,brl->bql", self.gram[d], c, optimize=True)
            energy = np.sum((c.conj()*transformed).real, axis=(1, 2))
            cross = np.sum((c.conj()*self.cross[d]).real, axis=(1, 2))
            value = (self.energy[d]+energy-2*cross)/self.energy[d]
            if np.any(value < -1e-8) or not np.all(np.isfinite(value)):
                raise ValueError("Invalid channel estimation error")
            result.append(np.maximum(value, 0))
        return np.stack(result, axis=1)


class FullMatrixDetector:
    """The same full-ICI, full colored-noise LMMSE detector for both estimates."""
    def __init__(self, dataset, backend):
        self.model, self.known, self.indices = dataset._model, dataset._known, dataset._indices
        self.noise = dataset._shared["unit_rf_noise_covariance"][:]
        self.backend = backend
        if backend.torch is None:
            self.whitener = make_whitener(self.noise, backend)
            return
        t, n = backend.torch, self.model.c["nfft"]
        carriers = np.fft.fftshift(np.arange(n))[dataset._masks["data"]]
        self.carriers = t.as_tensor(carriers, device="cuda")
        self.basis = t.as_tensor(self.model.sampled_basis.reshape(4, n, self.model.count)[[1, 2]],
                                 dtype=t.complex128, device="cuda")
        phase = np.exp(2j*np.pi/n*np.arange(n)[:, None]*carriers)/np.sqrt(n)
        self.phase = t.as_tensor(phase, dtype=t.complex128, device="cuda")
        known_time = input_windows(self.known[[1, 2]], self.model.length)
        self.known_time = t.as_tensor(known_time, dtype=t.complex128, device="cuda")
        covariance = time_noise_covariance(self.noise)
        self.chol = t.linalg.cholesky(t.as_tensor((covariance+covariance.conj().T)/2, device="cuda"))

    def detect(self, coefficients, observations, variances):
        b, n = len(variances), self.model.c["nfft"]
        if self.backend.torch is None:
            result = np.empty((b, 2, len(self.indices)), dtype=complex)
            for i in range(b):
                for d, symbol in enumerate((1, 2)):
                    h = taps_to_matrix(self.model.reconstruct(coefficients[i], symbol))
                    y = observations[i, symbol]
                    batch = EqualizerBatch(h[None], y[None], (y-h@self.known[symbol])[None],
                                           np.asarray(variances[i:i+1]), self.indices, self.noise)
                    result[i, d] = equalize_batch(batch, self.backend, whitener=self.whitener).symbols[0]
            return result
        t = self.backend.torch
        coeff = t.as_tensor(coefficients, dtype=t.complex128, device="cuda")
        taps = t.einsum("dnq,bql->bdnl", self.basis, coeff)
        response = t.fft.fft(taps, n=n, dim=-1).index_select(-1, self.carriers)
        matrix = response*self.phase
        known_time = (taps*self.known_time[None]).sum(dim=-1)
        obs = t.as_tensor(observations[:, [1, 2]], dtype=t.complex128, device="cuda")
        y = t.fft.ifft(t.fft.ifftshift(obs, dim=-1), dim=-1, norm="ortho")-known_time
        rhs = t.cat((matrix, y[..., None]), dim=-1)
        white = t.linalg.solve_triangular(self.chol, rhs, upper=False)
        h, y = white[..., :-1], white[..., -1:]
        gram = h.mH @ h
        gram.diagonal(dim1=-2, dim2=-1).add_(t.as_tensor(variances, dtype=t.float64, device="cuda")[:, None, None])
        return t.linalg.solve(gram, h.mH@y)[..., 0].cpu().numpy()


def _window_inputs(ds, wi):
    """Only publicly available observations and receiver noise enter prediction."""
    h = ds._h
    ids = np.flatnonzero(h["samples/window_index"][:] == wi)
    if not len(ids):
        raise ValueError("Physical window has no samples")
    clean, noise = h["samples/clean_observations"][ids], h["samples/standard_noise"][ids]
    variance = h["samples/rf_noise_variance"][:, ids]
    obs = clean[None] + np.sqrt(variance)[..., None, None]*noise[None]
    return ids, obs.reshape(-1, 4, ds._config["nfft"]), variance.ravel()


def _validate_split(ds):
    validate_plan(ds._config)
    _identity_check(ds, ds.split, ds.channel)
    expected = sample_identities(ds._config, ds.split)
    h = ds._h
    actual = list(zip(h["samples/window_index"][:].astype(int), h["samples/payload_repeat"][:].astype(int),
                      h["samples/noise_repeat"][:].astype(int)))
    if actual != expected or not np.all(h["samples/complete"][:]):
        raise ValueError("Sample identities differ from the accepted fixed split")


def _sync(backend):
    backend.synchronize()


def _timed(backend, fn):
    _sync(backend)
    start = time.perf_counter()
    value = fn()
    _sync(backend)
    return value, time.perf_counter()-start


def _omp_predict(omp, observations, effective_variance, eta, checkpoints, rows, whiteners, backend):
    y = whiten_observations(observations, rows, whiteners, backend)
    if backend.torch:
        ridge = backend.torch.as_tensor(eta*effective_variance, dtype=backend.torch.float32, device="cuda")
    else:
        ridge = eta*effective_variance
    output = omp.estimate(y, ridge, checkpoints)
    return {k: backend.numpy(value) for k, value in output.items()}


def _open_state(path, identity, split, nsnr, nframes, nwindows, etas, groups):
    if path.exists():
        state = h5py.File(path, "r+")
        if state.attrs.get("identity") != identity or state.attrs.get("split") != split:
            state.close()
            raise ValueError("Existing diagnostic results have a different identity")
        return state
    state = h5py.File(path, "w")
    state.attrs.update(identity=identity, split=split, complete=False)
    state.create_dataset("window_complete", (nwindows,), dtype="bool")
    state.create_dataset("lmmse_nmse", (nsnr, nframes, 2), dtype="float64", fillvalue=np.nan)
    state.create_dataset("estimator_seconds", (nwindows, 2), dtype="float64", fillvalue=np.nan)
    if split == "validation":
        state.create_dataset("omp_nmse", (len(etas), len(groups), nsnr, nframes, 2), dtype="float64", fillvalue=np.nan)
    else:
        state.create_dataset("omp_nmse", (nsnr, nframes, 2), dtype="float64", fillvalue=np.nan)
        state.create_dataset("bit_errors", (2, nsnr, nframes, 2), dtype="int64", fillvalue=-1)
        state.create_dataset("detector_seconds", (nwindows, 2), dtype="float64", fillvalue=np.nan)
    return state


def _evaluate_split(ds, output, identity, receiver, omp, rows, whiteners, config, backend, selected=None, suffix=""):
    split, nframes, nsnr = ds.split, ds.frame_count, len(ds.snr_db)
    nwindows = len(ds._h["windows/record"])
    etas, groups = config["ridge_multipliers"], config["group_counts"]
    path = output/f"{ds.channel.lower()}_{split}{suffix}.h5"
    detector = FullMatrixDetector(ds, backend) if split == "test" else None
    with _open_state(path, identity, split, nsnr, nframes, nwindows, etas, groups) as state:
        for wi in range(nwindows):
            if state["window_complete"][wi]:
                continue
            ids, obs, variance = _window_inputs(ds, wi)
            repeats = len(ids)
            batch_size = config.get("micro_batch_size", len(variance))
            # These predictors cannot receive reference CSI or labels.
            def lmmse():
                return np.concatenate([receiver.estimate_coefficients(obs[j:j+batch_size], ds._known, ds._masks,
                                       variance[j:j+batch_size]) for j in range(0, len(variance), batch_size)])
            def omp_one(eta, points):
                # Charge the same receiver-side noise-floor calculation to both
                # estimators; LMMSE performs it inside estimate_coefficients.
                batches = []
                for j in range(0, len(variance), batch_size):
                    y, v = obs[j:j+batch_size], variance[j:j+batch_size]
                    effective = receiver.effective_noise_variances(y, v)
                    batches.append(_omp_predict(omp, y, effective, eta, points, rows, whiteners, backend))
                return {k: np.concatenate([result[k] for result in batches]) for k in points}
            omp_results = []
            omp_time = 0.
            choices = etas if split == "validation" else [selected["ridge_multiplier"]]
            points = groups if split == "validation" else [selected["group_count"]]
            if split != "test" or wi % 2 == 0:
                lmmse_c, lmmse_time = _timed(backend, lmmse)
            for eta in choices:
                result, seconds = _timed(backend, lambda: omp_one(eta, points))
                omp_results.append(result)
                omp_time += seconds
            if split == "test" and wi % 2:
                lmmse_c, lmmse_time = _timed(backend, lmmse)
            detections, detector_times = [], []
            if split == "test":
                omp_c = omp_results[0][selected["group_count"]]
                for coefficients in (lmmse_c, omp_c):
                    symbols, seconds = _timed(backend, lambda: np.concatenate([
                        detector.detect(coefficients[j:j+batch_size], obs[j:j+batch_size], variance[j:j+batch_size])
                        for j in range(0, len(variance), batch_size)]))
                    detections.append(symbols)
                    detector_times.append(seconds)
            # Supervision is accessed only after all receiver outputs are computed.
            reference = ds._h["windows/reference_csi"][wi]
            scorer = ReferenceScore(ds._model, reference)
            state["lmmse_nmse"][:, ids, :] = scorer.nmse(lmmse_c).reshape(nsnr, repeats, 2)
            if split == "validation":
                for ei, result in enumerate(omp_results):
                    for ki, count in enumerate(groups):
                        state["omp_nmse"][ei, ki, :, ids, :] = scorer.nmse(result[count]).reshape(nsnr, repeats, 2)
            else:
                state["omp_nmse"][:, ids, :] = scorer.nmse(omp_c).reshape(nsnr, repeats, 2)
                truth = ds._h["samples/bits"][ids]
                for ai, symbols in enumerate(detections):
                    estimated_bits = decisions(symbols).reshape(nsnr, repeats, 2, len(ds._indices), 2)
                    errors = np.count_nonzero(estimated_bits != truth[None], axis=(-1, -2))
                    state["bit_errors"][ai, :, ids, :] = errors
                state["detector_seconds"][wi] = detector_times
            state["estimator_seconds"][wi] = (lmmse_time, omp_time)
            state.flush()
            state["window_complete"][wi] = True
            state.flush()
            if (wi+1) % 5 == 0 or wi+1 == nwindows:
                LOG.info("%s %s: %d/%d physical windows", ds.channel, split, wi+1, nwindows)
        if not np.all(state["window_complete"][:]):
            raise RuntimeError("Incomplete diagnostic")
        state.attrs["complete"] = True
        state.flush()
    return path


def select_validation(path, config):
    with h5py.File(path, "r") as state:
        if not state.attrs["complete"]:
            raise ValueError("Incomplete validation")
        values = state["omp_nmse"][:]
        score = values.mean(axis=(2, 3, 4))
        if not np.all(np.isfinite(score)):
            raise ValueError("Nonfinite validation scores")
        best = np.unravel_index(np.argmin(score), score.shape)
        return dict(group_count=config["group_counts"][best[1]], ridge_multiplier=config["ridge_multipliers"][best[0]],
                    objective="mean linear full-matrix NMSE over validation SNR, frames and D blocks",
                    score_linear=float(score[best]), candidate_mean_nmse_linear=score.tolist(),
                    grid_boundary_group_count=bool(best[1] == len(config["group_counts"])-1),
                    grid_boundary_ridge=bool(best[0] in (0, len(config["ridge_multipliers"])-1)))


def collect_test(path, channel, snr, selected, setup, ndata):
    with h5py.File(path, "r") as h:
        if not h.attrs["complete"]:
            raise ValueError("Incomplete test")
        errors = h["bit_errors"][:]
        if np.any(errors < 0):
            raise ValueError("Missing bit error counts")
        nframes = errors.shape[2]
        bits = nframes*2*ndata*2
        curves = {}
        for ai, name in enumerate(ALGORITHMS):
            nmse = h["lmmse_nmse" if ai == 0 else "omp_nmse"][:].mean(axis=(1, 2))
            count = errors[ai].sum(axis=(1, 2))
            seconds = h["estimator_seconds"][:, ai]
            detector = h["detector_seconds"][:, ai]
            frames = len(snr)*nframes
            curves[name] = dict(nmse_linear=nmse.tolist(), nmse_db=(10*np.log10(nmse)).tolist(),
                                bit_errors=count.tolist(), bits_per_snr=bits, ber=(count/bits).tolist(),
                                estimator_seconds=float(seconds.sum()), estimator_ms_per_frame=float(1000*seconds.sum()/frames),
                                detector_seconds=float(detector.sum()), detector_ms_per_frame=float(1000*detector.sum()/frames),
                                setup_seconds=setup[name])
        return dict(channel=channel, snr_db=snr.tolist(), frame_count_per_snr=nframes,
                    bit_count_per_snr=bits, selection=selected, curves=curves)


def run(datasets, output, config, backend):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Diagnostic output configuration mismatch")
    write_json(config_path, config)
    reports = {}
    for channel, folder in datasets.items():
        folder = Path(folder)
        identity = DataIdentity({x: folder/x for x in ("model.h5", "validation.h5", "test.h5")})
        source = dict(channel=channel, config=config, files=identity.files, environment=numerical_environment(backend),
                      manifest_sha256=file_sha256(folder/"manifest.json"),
                      code_sha256={x: file_sha256(x) for x in config["source_files"]})
        token = digest(source)
        with CompactDataset(folder/"validation.h5") as val, CompactDataset(folder/"test.h5") as test:
            if val.channel != channel or test.channel != channel or val.split != "validation" or test.split != "test":
                raise ValueError("Incorrect channel or split")
            if val._config != test._config or not np.array_equal(val.snr_db, test.snr_db):
                raise ValueError("Validation and test identities differ")
            _validate_split(val)
            _validate_split(test)
            start = time.perf_counter()
            receiver = restore_receiver_lmmse(val, backend)
            _sync(backend)
            setup_lmmse = time.perf_counter()-start
            start = time.perf_counter()
            dictionary, transforms, rows, whiteners = build_dictionary(
                val._model, val._known, val._masks, val._shared["unit_rf_noise_covariance"][:], val._blocks, backend)
            omp = GroupOMP(dictionary, transforms, backend)
            _sync(backend)
            setup_omp = time.perf_counter()-start
            if any(not np.array_equal(a, b) for a, b in zip(rows, receiver.rows)):
                raise ValueError("Group OMP and LMMSE use different observation rows")
            validation = _evaluate_split(val, output, token, receiver, omp, rows, whiteners, config, backend)
            coarse = select_validation(validation, config)
            extended = dict(config, ridge_multipliers=[coarse["ridge_multiplier"]],
                            group_counts=config.get("extension_group_counts", config["group_counts"]))
            extension_token = digest(dict(base=token, phase="extended_group_grid", controls=extended))
            validation_extended = _evaluate_split(val, output, extension_token, receiver, omp, rows, whiteners,
                                                  extended, backend, suffix="_extended")
            more = select_validation(validation_extended, extended)
            winner = more if more["score_linear"] < coarse["score_linear"] else coarse
            selected = dict(group_count=winner["group_count"], ridge_multiplier=winner["ridge_multiplier"],
                objective=winner["objective"], score_linear=winner["score_linear"],
                search="coarse ridge/group grid, then expanded group grid at coarse-selected ridge",
                coarse=coarse, extended=more,
                grid_boundary_group_count=bool(winner["group_count"] == max(extended["group_counts"])),
                grid_boundary_ridge=coarse["grid_boundary_ridge"])
            selected_path = output/f"{channel.lower()}_selection.json"
            if selected_path.exists() and json.loads(selected_path.read_text(encoding="utf-8")) != selected:
                raise ValueError("Frozen validation selection changed")
            write_json(selected_path, selected)
            LOG.info("%s validation selected K=%d eta=%g (group boundary=%s, ridge boundary=%s)",
                     channel, selected["group_count"], selected["ridge_multiplier"],
                     selected["grid_boundary_group_count"], selected["grid_boundary_ridge"])
            test_path = _evaluate_split(test, output, token, receiver, omp, rows, whiteners, config, backend, selected)
            reports[channel] = collect_test(test_path, channel, test.snr_db, selected,
                                            dict(joint_lmmse=setup_lmmse, group_omp=setup_omp), len(test._indices))
            reports[channel]["source"] = source
            identity.assert_unchanged()
        del omp, dictionary, transforms, receiver
        if backend.torch:
            backend.torch.cuda.empty_cache()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    final = dict(complete=True, commit=commit, diagnostic="receiver-only CSI; no reference-assisted compensation or legacy raw caches",
                 input="identical physical observations, public P1/P2 and D tracking positions, independent frozen receiver prior",
                 omp="full 34 DPSS coefficients per selected delay, group-score selection, marginal-prior scaling, ridge active refit",
                 noise="both estimators use receiver-derived effective noise; OMP whitens thermal noise but does not model data ICI covariance",
                 equalizer="same full-matrix LMMSE with colored thermal covariance for each estimate",
                 nmse="10 log10(mean over frames and D blocks of per-block full-matrix relative squared error)",
                 timing="synchronized estimator-only wall time including selection, noise-floor calculation, whitening and coefficient transfer; same micro-batches for both; test estimator order alternates by window; setup and equalization reported separately; no labels or reference scoring",
                 environment=numerical_environment(backend), channels=reports)
    write_json(output/"metrics.json", final)
    _publish(output, final)
    return final


def _publish(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (output/"curves.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["channel", "method", "snr_db", "nmse_db", "ber", "bit_errors", "bits"])
        for channel, r in report["channels"].items():
            for method, value in r["curves"].items():
                for row in zip(r["snr_db"], value["nmse_db"], value["ber"], value["bit_errors"]):
                    w.writerow([channel, method, *row, value["bits_per_snr"]])
    for metric in ("nmse", "ber"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        for ax, (channel, r) in zip(axes, report["channels"].items()):
            for name, values in r["curves"].items():
                ys = values["nmse_db"] if metric == "nmse" else values["ber"]
                ax.plot(r["snr_db"], ys, "o-", label=name.replace("_", " "))
            if metric == "ber":
                ax.set_yscale("log")
            ax.set(title=channel, xlabel="SNR (dB)", ylabel="CSI NMSE (dB)" if metric == "nmse" else "BER",
                   xticks=r["snr_db"])
            ax.grid(alpha=.25, which="both")
            ax.legend()
        fig.suptitle("Receiver-only DPSS channel estimates; held-out test")
        fig.savefig(output/f"{metric}_snr.png", dpi=170)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.3), layout="constrained")
    x = np.arange(len(report["channels"]))
    for offset, method in ((-.18, "joint_lmmse"), (.18, "group_omp")):
        ax.bar(x+offset, [r["curves"][method]["estimator_ms_per_frame"] for r in report["channels"].values()],
               width=.35, label=method.replace("_", " "))
    ax.set(xticks=x, xticklabels=list(report["channels"]), ylabel="Channel estimation time (ms/frame)")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.savefig(output/"estimator_runtime.png", dpi=170)
    plt.close(fig)
