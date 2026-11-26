"""Independent pilot-aided LMMSE estimates in one OFDM FFT window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.signal.windows import dpss

from data_generation.src.receiver.dpss import SINGLE_SYMBOL_DOMAIN
from data_generation.src.receiver.estimation import input_windows
from data_generation.src.receiver.pdd import (
    observation_spectrum, selected_noise_whitener, selected_time_covariance, stationary_interference,
)
from src.common.numerics import hermitian


def symbol_basis(n, sample_rate, bandwidth_hz, order):
    """Orthonormal DPSS on useful samples, with a constant zero-Doppler case."""
    if (type(order) is not int or not 1 <= order <= n or not np.isfinite(bandwidth_hz)
            or not 0 <= bandwidth_hz < sample_rate/2 or (bandwidth_hz == 0 and order != 1)):
        raise ValueError("Invalid single-symbol DPSS bandwidth or order")
    if bandwidth_hz == 0:
        return np.full((n, 1), n**-.5)
    return dpss(n, n*bandwidth_hz/sample_rate, Kmax=order, sym=True, norm=2).T.copy()


@dataclass(frozen=True)
class SymbolDPSSModel:
    c: dict
    basis: np.ndarray
    delays: np.ndarray

    @property
    def count(self):
        return self.basis.shape[1]

    @property
    def length(self):
        return len(self.delays)


class SingleSymbolEstimator:
    """A prepared receiver for one public pilot pattern and one observation."""

    def __init__(self, model, rows, noise_gain, covariance, cross, whitener, regularizer, backend):
        self.model, self.rows, self.noise_gain, self.backend = model, rows, noise_gain, backend
        self.values, self.projected, self.gain, self.ridge, self.diagnostics = observation_spectrum(
            covariance, cross, whitener, regularizer, backend)

    def estimate_coefficients(self, observations, noise_variances, model_error_ratio=0):
        """Return batch × Q × K coefficients from batch × nfft observations."""
        y = np.atleast_2d(np.asarray(observations, np.complex128))
        if y.ndim != 2 or y.shape[1] != self.model.c["nfft"] or not np.all(np.isfinite(y)):
            raise ValueError("Expected finite observations from one FFT window per batch row")
        variance = np.broadcast_to(np.asarray(noise_variances, float), (len(y),))
        if np.any(variance < 0) or not np.all(np.isfinite(variance)):
            raise ValueError("Invalid RF noise variance")
        if not 0 <= model_error_ratio <= 1:
            raise ValueError("Invalid model-error ratio")
        if model_error_ratio:
            power = np.mean(abs(y)**2, axis=1)/self.noise_gain
            variance = variance + model_error_ratio*np.maximum(power-variance, 0)
        b = self.backend
        spectral = self.projected.conj().T @ b.array(y[:, self.rows].T)
        denominator = self.values[:, None] + b.array(variance)[None] + self.ridge
        coefficients = self.gain @ (spectral/denominator)
        return b.numpy(coefficients).T.reshape(len(y), self.model.count, self.model.length)


class SymbolPilotDesign:
    """Cache responses to public pilot patterns for a grid of DPSS models."""

    def __init__(self, basis, delays, known, masks, noise, backend):
        known, noise = np.asarray(known, complex), np.asarray(noise, complex)
        n = len(basis)
        if (known.ndim != 2 or known.shape[1] != n or not np.all(np.isfinite(known))
                or np.any(known[:, ~masks["pilot"]] != 0) or not np.all(np.any(known != 0, axis=1))):
            raise ValueError("Expected public pilots with zero known values outside the pilot carriers")
        if (noise.shape != (n, n) or not np.all(np.isfinite(noise)) or not np.allclose(noise, noise.conj().T)):
            raise ValueError("Invalid unit RF noise covariance")
        self.basis, self.delays, self.backend = basis, delays, backend
        self.rows = np.flatnonzero(masks["tracking"])
        self.noise_gain = float(np.trace(noise).real/n)
        self.whitener = selected_noise_whitener(noise[np.ix_(self.rows, self.rows)], backend)
        self.data_correlation = np.fft.ifft(np.fft.ifftshift(masks["data"].astype(float)))
        self.difference = (np.arange(n)[:, None]-np.arange(n)) % n
        windows = input_windows(known, len(delays), delay_indices=delays)
        responses = []
        for window in windows:
            time = basis[:, :, None]*window[:, None, :]
            frequency = np.fft.fftshift(np.fft.fft(time, axis=0, norm="ortho"), axes=0)
            responses.append(frequency[self.rows])
        self.responses = backend.array(np.stack(responses))

    def build(self, prior):
        """Prepare separate marginal LMMSE receivers, one per public pilot pattern."""
        model, b = prior.model, self.backend
        columns = np.searchsorted(self.delays, model.delays)
        protocol = model.c.get("learning_dataset", {}).get("records", {})
        held_out = set(protocol.get("validation", [])) | set(protocol.get("test", []))
        if (prior.fit_domain != SINGLE_SYMBOL_DOMAIN or not prior.source_records
                or held_out.intersection(prior.source_records) or prior.empirical_weight != 0 or prior.scale != 1
                or prior.interference_scale != 1 or prior.feedback_iterations != 0):
            raise ValueError("Single-symbol estimation requires independent FFT-window moments and fixed priors")
        if (np.any(columns >= len(self.delays)) or not np.array_equal(self.delays[columns], model.delays)
                or not np.array_equal(self.basis[:, :model.count], model.basis)):
            raise ValueError("Prior basis or delays are outside the cached pilot design")
        patterns, size, _, _ = self.responses.shape
        q, k = model.count, model.length
        cov = b.array(np.zeros((patterns, size, size), complex))
        cross = b.array(np.zeros((patterns, size, q, k), complex))
        interference = np.zeros((len(model.basis), len(model.basis)), complex)
        for block, temporal, delay in prior.separable_blocks():
            indices = b.array(columns[block])
            response = self.responses[:, :, :q, indices]
            lt = b.array(linalg.cholesky(temporal, lower=True))
            ld = b.array(linalg.cholesky(delay, lower=True))
            factored = ((response @ ld).swapaxes(-1, -2) @ lt).swapaxes(-1, -2)
            z = factored.reshape(patterns, size, -1)
            cov += z @ z.conj().swapaxes(-1, -2)
            cross[..., block] = (lt @ factored.conj()) @ ld.T
            stationary = stationary_interference(delay, model.delays[block], self.data_correlation, self.difference)
            interference += (model.basis @ temporal @ model.basis.conj().T)*stationary
        cov += b.array(hermitian(selected_time_covariance(interference, self.rows)))
        return tuple(SingleSymbolEstimator(model, self.rows, self.noise_gain, cov[s],
                    cross[s].reshape(size, q*k).T, self.whitener, prior.regularizer, b) for s in range(patterns))
