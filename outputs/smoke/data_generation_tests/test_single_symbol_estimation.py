"""Numerical checks for a receiver restricted to one OFDM symbol."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from data_generation.src.receiver.dpss import SINGLE_SYMBOL_DOMAIN
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.single_symbol import SymbolDPSSModel, SymbolPilotDesign, symbol_basis
from data_generation.src.transmitter.ofdm import fft
from src.common.numerics import DenseBackend
from src.evaluation.four_data_single_symbol import estimate_frames, projection_nmse
from src.evaluation.ncs1_delay_support import full_nmse


def test_single_symbol_lmmse_matches_a_dense_selected_frequency_solve():
    rng = np.random.default_rng(71)
    n, q = 8, 2
    delays = np.array([0, 2, 5])
    basis = symbol_basis(n, 8, 1., q)
    model = SymbolDPSSModel(dict(nfft=n), basis, delays)
    prior_coeff = rng.standard_normal((12, q, len(delays)))+1j*rng.standard_normal((12, q, len(delays)))
    prior = FrozenPrior.fit_coefficients(model, prior_coeff, 0, 1, [51], fit_domain=SINGLE_SYMBOL_DOMAIN,
                                         delay_block_size=3)
    pilot, data = np.isin(np.arange(n), [0, 6]), np.isin(np.arange(n), [2, 4, 5])
    tracking = pilot | np.isin(np.arange(n), [1, 3])
    known = np.zeros((2, n), complex)
    known[:, pilot] = [[1, 1j], [-1, -1j]]
    factor = rng.standard_normal((n, n))+1j*rng.standard_normal((n, n))
    noise = np.eye(n)+factor @ factor.conj().T/n
    masks = dict(pilot=pilot, data=data, tracking=tracking)
    estimators = SymbolPilotDesign(basis, delays, known, masks, noise, DenseBackend()).build(prior)

    def observation_matrix(values):
        x = input_windows(values[None], len(delays), delay_indices=delays)[0]
        return fft((basis[:, :, None]*x[:, None, :]).reshape(n, -1).T).T[tracking]

    coefficient_covariance = np.zeros((q, len(delays), q, len(delays)), complex)
    for block, temporal, delay in prior.separable_blocks():
        coefficient_covariance[:, block, :, block] = np.einsum("qr,lk->qlrk", temporal, delay)
    coefficient_covariance = coefficient_covariance.reshape(q*len(delays), q*len(delays))
    interference = np.zeros((tracking.sum(), tracking.sum()), complex)
    for active_bin in np.flatnonzero(data):
        response = observation_matrix(np.eye(n)[active_bin])
        interference += response @ coefficient_covariance @ response.conj().T

    observations = rng.standard_normal((3, 2, n))+1j*rng.standard_normal((3, 2, n))
    variance, floor = np.array([0.09, 0.5, 1.2]), .03
    prepared = estimate_frames(estimators, observations[None], variance[None], floor)[0]
    for s, estimator in enumerate(estimators):
        response = observation_matrix(known[s])
        cross = coefficient_covariance @ response.conj().T
        covariance = response @ cross + interference
        y = observations[:, s]
        receive_power = np.mean(abs(y)**2, axis=1)/(np.trace(noise).real/n)
        effective = variance + floor*np.maximum(receive_power-variance, 0)
        expected = np.stack([cross @ linalg.solve(covariance+(v+estimator.ridge)*noise[np.ix_(tracking, tracking)],
                            sample[tracking], assume_a="pos") for v, sample in zip(effective, y, strict=True)])
        np.testing.assert_allclose(prepared[:, s].reshape(3, -1), expected, rtol=1e-12, atol=1e-12)
        with pytest.raises(ValueError, match="one FFT window"):
            estimator.estimate_coefficients(observations, variance)
    changed = observations.copy()
    changed[:, 1] *= -5j
    np.testing.assert_array_equal(estimate_frames(estimators, changed[None], variance[None], floor)[0, :, 0],
                                  prepared[:, 0])


def test_symbol_projection_nmse_includes_unselected_delays_and_time_modes():
    rng = np.random.default_rng(702)
    n = 12
    basis = symbol_basis(n, 12, 1.5, 3)
    delays = np.array([1, 3, 8])
    truth = rng.standard_normal((4, n, n))+1j*rng.standard_normal((4, n, n))
    coeff = rng.standard_normal((7, 4, 3, 3))+1j*rng.standard_normal((7, 4, 3, 3))
    projected = basis.T @ truth[..., delays]
    energy = np.sum(abs(truth)**2, axis=(-1, -2))
    np.testing.assert_allclose(projection_nmse(coeff, projected, energy), full_nmse(basis @ coeff, truth, delays),
                               rtol=1e-14, atol=1e-14)
