"""Meaningful checks for the separate four-data Q search."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg

from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.transmitter.ofdm import decisions
from src.algorithms.full_matrix_lmmse.equalizer import equalize_batch
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.numerics import DenseBackend
from src.data.contract import target_nmse_db
from src.evaluation.four_data_q_sweep import (
    SNR_DB, VALIDATION, TEST, PRIOR, basis, choose, compensation, config, four_data_estimator_layout,
    make_frame, masks, raw_nmse, score_frame, score_frame_cpu, time_noise_covariance,
)


def test_carriers_and_paired_frame_are_fixed():
    c = config("NOF1")
    m = masks(c)
    assert (m["data"].sum(), m["pilot"].sum(), m["protection"].sum(), m["edge"].sum()) == (512, 256, 128, 128)
    assert np.array_equal(np.diff(np.flatnonzero(m["pilot"]))[:6], [3, 4, 3, 4, 3, 4])
    a, b = make_frame(c, 13, 2, 1), make_frame(c, 21, 14, 1)
    np.testing.assert_array_equal(a.known, b.known)
    assert np.all(a.known[:, m["data"]] == 0)
    assert np.all(a.x[:, m["null"]] == 0)
    assert a.bits.shape == (4, 512, 2)
    assert not ({r for r, _ in PRIOR} & {r for r, _ in VALIDATION+TEST})
    assert not ({r for r, _ in VALIDATION} & {r for r, _ in TEST})


def test_dpss_q32_and_compensation_targets():
    c = config("NCS1")
    vectors, ratio = basis(c)
    assert vectors.shape == (5632, 32)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(32), atol=2e-13)
    assert np.all(np.diff(ratio) <= 1e-12)
    rng = np.random.default_rng(42)
    raw = rng.uniform(.2, .8, (5, len(SNR_DB), 4))
    alpha = compensation(raw, "NCS1")
    np.testing.assert_allclose(10*np.log10((raw*alpha[None, :, None]**2).mean(axis=(0, 2))),
                               target_nmse_db("NCS1", SNR_DB), atol=1e-12)


def test_raw_nmse_includes_all_reference_energy():
    rng = np.random.default_rng(33)
    n, q, length = 8, 2, 3
    c = dict(nfft=n, cp=2, fft_advance=0)
    full, _ = np.linalg.qr(rng.normal(size=(40, q)))
    model = DPSSModel(full, 12, 1e-5, length, c)
    coeff = rng.normal(size=(3, q, length))+1j*rng.normal(size=(3, q, length))
    truth = rng.normal(size=(4, n, n))+1j*rng.normal(size=(4, n, n))
    actual = raw_nmse(model, coeff, truth)
    for s in range(3):
        for d in range(4):
            a = taps_to_matrix(model.reconstruct(coeff[s], d))
            b = taps_to_matrix(truth[d])
            np.testing.assert_allclose(actual[s, d], np.sum(abs(a-b)**2)/np.sum(abs(b)**2), rtol=1e-13)


def test_time_coordinates_equal_full_frequency_lmmse():
    rng = np.random.default_rng(123)
    n = 12
    taps = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    known = np.zeros(n, complex)
    known[[2, 8]] = [1+1j, 1-1j]
    data = np.array([3, 4, 5, 6])
    y = rng.normal(size=n)+1j*rng.normal(size=n)
    a = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    noise = a@a.conj().T+np.eye(n)
    variance = 0.7
    h = taps_to_matrix(taps)
    backend = DenseBackend()
    whiten = make_whitener(noise, backend)
    batch = EqualizerBatch(h[None], y[None], (y-h@known)[None], np.array([variance]), data, noise)
    expected = equalize_batch(batch, backend, whitener=whiten).symbols[0]
    carriers = np.fft.fftshift(np.arange(n))[data]
    phase = np.exp(2j*np.pi/n*np.arange(n)[:, None]*carriers)/np.sqrt(n)
    matrix = np.fft.fft(taps, axis=-1)[:, carriers]*phase
    from data_generation.src.receiver.estimation import input_windows
    residual = np.fft.ifft(np.fft.ifftshift(y), norm="ortho")-np.sum(taps*input_windows(known[None], n)[0], axis=-1)
    factor = linalg.cholesky(time_noise_covariance(noise), lower=True)
    hw = linalg.solve_triangular(factor, matrix, lower=True)
    yw = linalg.solve_triangular(factor, residual, lower=True)
    result = linalg.solve(hw.conj().T@hw+variance*np.eye(len(data)), hw.conj().T@yw, assume_a="pos")
    np.testing.assert_allclose(result, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(decisions(result), decisions(expected))


def test_selection_uses_both_complete_validation_grids():
    from src.evaluation.four_data_q_sweep import ORDERS
    values = {"NOF1": [dict(q=q, mean_ber=abs(q-19)/1000+.1) for q in ORDERS],
              "NCS1": [dict(q=q, mean_ber=abs(q-24)/100+.2) for q in ORDERS]}
    result = choose(values)
    assert result["by_channel"] == {"NOF1": 19, "NCS1": 24}
    assert result["shared"] == 24


def test_cuda_scoring_matches_official_full_matrix():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(52)
    n, q, length = 12, 2, 4
    c = dict(nfft=n, cp=2, fft_advance=0)
    full, _ = np.linalg.qr(rng.normal(size=(4*(n+2), q)))
    model = DPSSModel(full, 12, 1e-5, length, c)
    complex_random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    coeff, truth = complex_random((2, q, length)), complex_random((4, n, n))
    y, known = complex_random((2, 4, n)), np.zeros((4, n), complex)
    known[:, 2] = 1+1j
    mask = np.isin(np.arange(n), [3, 4, 5])
    bits = rng.integers(0, 2, (4, 3, 2), dtype=np.uint8)
    variance, alpha = np.array([.4, 1.2]), np.array([.2, .7])
    a = complex_random((n, n))
    noise = a@a.conj().T+np.eye(n)
    expected = score_frame_cpu(model, coeff, truth, y, known, bits, variance, alpha, noise, mask)
    actual = score_frame(model, coeff, truth, y, known, bits, variance, alpha, noise, mask, DenseBackend("cuda"))
    np.testing.assert_array_equal(actual, expected)
