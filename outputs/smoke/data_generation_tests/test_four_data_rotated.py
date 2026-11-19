"""Independent RF-power and staggered-mask LMMSE checks for the new 4D link."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg
from threadpoolctl import threadpool_limits

from data_generation.src.config import load_config
from data_generation.src.receiver.chain import ReceiverChain
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.four_data import FourDataEstimator, fixed_four_data_model, make_four_data_frame
from data_generation.src.receiver.four_data_rotated import (
    TransmitPower, full_band_layout, make_full_band_frame, waveform,
)
from data_generation.src.transmitter.ofdm import carrier_masks, qpsk
from src.common.numerics import DenseBackend


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(2):
        yield


def test_rf_ensemble_power_matches_independent_tone_sum_and_frame_power():
    config = load_config()
    config.update(nfft=56, cp=16, fft_advance=2, guard_samples=12, null_per_edge=0,
                  data_count=32, pilot_count=8)
    chain = ReceiverChain(config)
    calibration = TransmitPower(chain)
    known, masks = full_band_layout(config)
    literal = 0.0
    for symbol in range(4):
        for carrier in np.flatnonzero(masks["data"][symbol]):
            tone = np.zeros_like(known)
            tone[symbol, carrier] = 1
            real = chain.transmit(waveform(tone, config))
            imag = chain.transmit(waveform(1j*tone, config))
            literal += .5*(np.sum(real**2)+np.sum(imag**2))
    np.testing.assert_allclose(calibration.data_energy(masks["data"]), literal, rtol=2e-13)
    power = calibration.match(config, known, masks["data"])
    np.testing.assert_allclose(power["new_expected_rf_power"], power["old_expected_rf_power"], rtol=1e-14)
    known *= np.sqrt(power["pilot_symbol_energy"])
    new = make_full_band_frame(config, 11, 2, known, masks)
    old = make_four_data_frame(config, 11, 2)
    np.testing.assert_array_equal(new.bits, old.bits[:, :14])
    np.testing.assert_array_equal(new.known[masks["data"]], 0)
    for symbol in range(4):
        np.testing.assert_array_equal(new.x[symbol, masks["data"][symbol]], qpsk(new.bits[symbol]))
    old_energy = np.sum(chain.transmit(old.baseband)**2)
    gain = np.sqrt(old_energy/np.sum(chain.transmit(new.baseband)**2))
    np.testing.assert_allclose(np.sum(chain.transmit(gain*new.baseband)**2), old_energy, rtol=2e-13)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_staggered_rows_match_explicit_dense_channel_lmmse(device):
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    config = load_config()
    config.update(nfft=14, cp=6, fft_advance=1, guard_samples=6, null_per_edge=0,
                  data_count=8, pilot_count=2)
    rng = np.random.default_rng(42)
    model = fixed_four_data_model(config, count=2, length=3)
    prior = FrozenPrior.fit_coefficients(model, rng.normal(size=(20, 2, 3))+1j*rng.normal(size=(20, 2, 3)),
                                         0, 1, (51,), delay_block_size=2)
    base = carrier_masks(config)
    masks = {name:np.stack([np.roll(base[name], s) for s in (0, 1, 3, 5)])
             for name in ("data", "pilot", "tracking")}
    masks["tracking"][1, np.flatnonzero(np.roll(base["protection"], 1))[0]] = False
    known = np.zeros((4, 14), complex)
    for s in range(4):
        known[s, masks["pilot"][s]] = qpsk(rng.integers(0, 2, (2, 2)))
    rows = [np.flatnonzero(mask) for mask in masks["tracking"]]
    offsets = np.cumsum([0]+list(map(len, rows)))
    basis = model.sampled_basis.reshape(4, 14, 2)
    covariance = np.zeros((6, 6), complex)
    for select, temporal, delay in prior.separable_blocks():
        indices = (np.arange(2)[:, None]*3+np.arange(3)[select]).ravel()
        covariance[np.ix_(indices, indices)] = np.kron(temporal, delay)
    def response(s, x):
        delayed = input_windows(x[None], 3)[0]
        time = (basis[s, :, :, None]*delayed[:, None, :]).reshape(14, 6)
        return np.fft.fftshift(np.fft.fft(time, axis=0, norm="ortho"), axes=0)[rows[s]]
    a = np.concatenate([response(s, x) for s, x in enumerate(known)])
    interference = np.zeros((offsets[-1], offsets[-1]), complex)
    for s in range(4):
        target = slice(offsets[s], offsets[s+1])
        for carrier in np.flatnonzero(masks["data"][s]):
            x = np.zeros(14, complex)
            x[carrier] = 1
            atom = response(s, x)
            interference[target, target] += atom @ covariance @ atom.conj().T
    z = rng.normal(size=(14, 14))+1j*rng.normal(size=(14, 14))
    noise = z @ z.conj().T/14+np.eye(14)
    selected_noise = linalg.block_diag(*[noise[np.ix_(r, r)] for r in rows])
    estimator = FourDataEstimator(prior, known, masks, noise, DenseBackend(device))
    y = rng.normal(size=(2, 4, 14))+1j*rng.normal(size=(2, 4, 14))
    variances, expected = np.array([.01, .4]), []
    for yi, variance in zip(y, variances, strict=True):
        selected = np.concatenate([yi[s, r] for s, r in enumerate(rows)])
        weights = linalg.solve(a @ covariance @ a.conj().T+interference
                               +(variance+estimator.ridge)*selected_noise, selected, assume_a="her")
        expected.append((covariance @ a.conj().T @ weights).reshape(2, 3))
    np.testing.assert_allclose(estimator.estimate_coefficients(y, variances), expected, rtol=2e-10, atol=2e-11)
    # Known transmit gain g is absorbed into CSI. Scaling P by g^2
    # is exactly equivalent to scaling only the RF variance by 1/g^2.
    gain = 1.13
    physical_variances = variances*gain**2
    actual = estimator.estimate_coefficients(gain*y, physical_variances/gain**2)
    scaled_expected = []
    for yi, variance in zip(gain*y, physical_variances, strict=True):
        selected = np.concatenate([yi[s, r] for s, r in enumerate(rows)])
        weights = linalg.solve(gain**2*(a @ covariance @ a.conj().T+interference)
                               +(variance+gain**2*estimator.ridge)*selected_noise, selected, assume_a="her")
        scaled_expected.append((gain**2*covariance @ a.conj().T @ weights).reshape(2, 3))
    np.testing.assert_allclose(actual, scaled_expected, rtol=2e-10, atol=2e-11)
