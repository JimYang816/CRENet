"""Paired precision and runtime audit for the receiver-only DPSS comparison."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import numpy as np

from src.algorithms.dpss_bomp.estimator import GroupOMP, build_dictionary, whiten_observations
from src.common.experiment import file_sha256, write_json
from src.evaluation.dpss_channel_comparison import _omp_predict, _window_inputs, restore_receiver_lmmse
from src.evaluation.fingerprints import DataIdentity, numerical_environment
from src.data.compact import CompactDataset


class LMMSEFloat32:
    """The frozen receiver-LMMSE formula evaluated in complex64 on CUDA."""
    def __init__(self, receiver):
        t = receiver.backend.torch
        if t is None:
            raise ValueError("The precision timing audit requires CUDA")
        self.receiver, self.t = receiver, t
        self.projected = receiver.projected.to(t.complex64)
        self.gain = receiver.gain.to(t.complex64)
        self.values = receiver.values.to(t.float32)

    def estimate(self, observations, variance):
        t, e = self.t, self.receiver
        selected = np.concatenate([observations[:, s, row] for s, row in enumerate(e.rows)], axis=1)
        effective = e.effective_noise_variances(observations, variance)
        y = t.as_tensor(selected.T, dtype=t.complex64, device="cuda")
        spectrum = self.projected.mH @ y
        denom = self.values[:, None] + t.as_tensor(effective, dtype=t.float32, device="cuda")[None] + e.ridge
        coefficients = self.gain @ (spectrum/denom)
        model = e.prior.model
        return coefficients.T.reshape(len(observations), model.count, model.length).cpu().numpy()


def _timed(backend, function):
    backend.synchronize()
    start = time.perf_counter()
    value = function()
    backend.synchronize()
    return value, time.perf_counter()-start


def _audit_active_refit(omp, rows, whiteners, receiver, obs, variance, eta, k, backend):
    """Independently solve one selected active set in complex128, without truth."""
    t = backend.torch
    effective = receiver.effective_noise_variances(obs, variance)
    estimate = _omp_predict(omp, obs, effective, eta, [k], rows, whiteners, backend)[k]
    i = len(obs)-1
    groups = np.flatnonzero(np.any(estimate[i] != 0, axis=0))
    if len(groups) != k:
        raise ValueError("Unexpected number of active delay groups")
    index = t.as_tensor(groups, device="cuda")
    matrix = omp.a.index_select(1, index).reshape(omp.m, k*omp.q).to(t.complex128)
    y = whiten_observations(obs[i:i+1], rows, whiteners, backend)[0].to(t.complex128)
    normal = matrix.mH @ matrix
    normal.diagonal().add_(float(eta*effective[i]))
    z = t.linalg.solve(normal, matrix.mH@y).reshape(k, omp.q, 1)
    reference = (omp.transforms.index_select(0, index).to(t.complex128) @ z)[..., 0].cpu().numpy()
    actual = estimate[i, :, groups]
    relative = np.linalg.norm(actual-reference)/max(np.linalg.norm(reference), 1e-300)
    return dict(selected_groups=k, relative_coefficient_error_vs_complex128=float(relative),
                definition="same selected support and complex64-built dictionary, independently refit in complex128")


def run(datasets, source_output, destination, backend, windows=10, repeats=3):
    if not backend.torch or windows < 2 or windows % 2 or repeats < 1:
        raise ValueError("Audit needs CUDA, an even window count and positive repeats")
    source_output, destination = Path(source_output), Path(destination)
    original = json.loads((source_output/"metrics.json").read_text(encoding="utf-8"))
    if not original.get("complete"):
        raise ValueError("Primary comparison must be complete")
    channels = {}
    for channel, supplied in datasets.items():
        directory = Path(supplied)
        identity = DataIdentity({x: directory/x for x in ("model.h5", "validation.h5")})
        expected = original["channels"][channel]["source"]["files"]
        if any(identity.files[name] != expected[name] for name in identity.files):
            raise ValueError("Validation/model identity differs from the primary comparison")
        selection = original["channels"][channel]["selection"]
        k, eta = selection["group_count"], selection["ridge_multiplier"]
        with CompactDataset(directory/"validation.h5") as ds:
            receiver = restore_receiver_lmmse(ds, backend)
            single = LMMSEFloat32(receiver)
            dictionary, transforms, rows, whiteners = build_dictionary(
                ds._model, ds._known, ds._masks, ds._shared["unit_rf_noise_covariance"][:], ds._blocks, backend)
            omp = GroupOMP(dictionary, transforms, backend)
            selected_windows = np.linspace(0, len(ds._h["windows/record"])-1, windows, dtype=int)
            if len(set(selected_windows)) != windows:
                raise ValueError("Not enough distinct validation windows")
            batches = []
            for j in range(0, windows, 2):
                data = [_window_inputs(ds, int(i)) for i in selected_windows[j:j+2]]
                batches.append((np.concatenate([x[1] for x in data]), np.concatenate([x[2] for x in data])))

            def actions(obs, variance):
                def wide():
                    return receiver.estimate_coefficients(obs, ds._known, ds._masks, variance)
                def narrow():
                    return single.estimate(obs, variance)
                def pursuit():
                    eff = receiver.effective_noise_variances(obs, variance)
                    return _omp_predict(omp, obs, eff, eta, [k], rows, whiteners, backend)[k]
                return {"lmmse_complex128": wide, "lmmse_complex64": narrow, "omp_complex64": pursuit}

            # Warm each implementation on identical data before timing.
            for fn in actions(*batches[0]).values():
                _timed(backend, fn)
            names = ("lmmse_complex128", "lmmse_complex64", "omp_complex64")
            elapsed = {name: [] for name in names}
            coefficient_errors, channel_errors = [], []
            basis = ds._model.sampled_basis.reshape(4, ds._config["nfft"], ds._model.count)[[1, 2]]
            gram = np.einsum("dnq,dnr->qr", basis.conj(), basis, optimize=True)
            for repeat in range(repeats):
                total = {name: 0. for name in names}
                for batch_index, (obs, variance) in enumerate(batches):
                    fn = actions(obs, variance)
                    shift = (repeat+batch_index) % len(names)
                    order = names[shift:]+names[:shift]
                    result = {}
                    for name in order:
                        result[name], seconds = _timed(backend, fn[name])
                        total[name] += seconds
                    a = result["lmmse_complex128"]
                    b = result["lmmse_complex64"].astype(np.complex128)
                    delta = b-a
                    coefficient_errors.extend((np.linalg.norm(delta.reshape(len(a), -1), axis=1)
                                               / np.maximum(np.linalg.norm(a.reshape(len(a), -1), axis=1), 1e-300)).tolist())
                    def energy(c):
                        mapped = np.einsum("qr,brl->bql", gram, c, optimize=True)
                        return np.maximum(np.sum((c.conj()*mapped).real, axis=(1, 2)), 0)
                    channel_errors.extend(np.sqrt(energy(delta)/np.maximum(energy(a), 1e-300)).tolist())
                frame_count = sum(len(x[1]) for x in batches)
                for name in names:
                    elapsed[name].append(1000*total[name]/frame_count)
            active_refit = _audit_active_refit(omp, rows, whiteners, receiver, *batches[0], eta, k, backend)
            channels[channel] = dict(group_count=k, ridge_multiplier=eta, validation_window_indices=selected_windows.tolist(),
                timed_frames_per_repeat=frame_count, repeats=repeats,
                milliseconds_per_frame={name: dict(median=float(statistics.median(values)), repeats=values)
                                        for name, values in elapsed.items()},
                lmmse32_relative_coefficient_error=dict(mean=float(np.mean(coefficient_errors)), max=float(np.max(coefficient_errors))),
                lmmse32_relative_dpss_channel_error=dict(mean=float(np.mean(channel_errors)), max=float(np.max(channel_errors))),
                omp_active_refit_audit=active_refit,
                input_files=identity.files)
            identity.assert_unchanged()
        del receiver, single, dictionary, transforms, omp
        backend.torch.cuda.empty_cache()
    report = dict(complete=True, source_metrics_sha256=file_sha256(source_output/"metrics.json"),
                  definition="warm synchronized estimation time on identical receiver-only validation observations; "
                             "same group settings and batch sizes; alternating execution order; setup, detector and scoring excluded",
                  note="LMMSE complex64 is an additional precision/timing audit, not the primary test curve",
                  environment=numerical_environment(backend), channels=channels)
    write_json(destination, report)
    return report
