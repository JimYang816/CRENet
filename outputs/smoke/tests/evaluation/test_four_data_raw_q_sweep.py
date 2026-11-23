"""Verify raw CSI detection and the independent full-matrix implementation."""
from inspect import signature

import numpy as np
import pytest
from scipy import linalg

from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.dpss import DPSSModel
from src.common.numerics import DenseBackend
from src.evaluation.four_data_raw_q_sweep import detect_raw


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(85)
    n, order, length = 12, 3, 4
    basis, _ = np.linalg.qr(rng.normal(size=(4*(n+2), order)))
    model = DPSSModel(basis, 12, 1e-5, length, dict(nfft=n, cp=2, fft_advance=0))
    random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    coefficients = random((2, order, length))
    y = random((2, 4, n))
    known = np.zeros((4, n), complex)
    known[:, [2, 8]] = random((4, 2))
    data = np.isin(np.arange(n), [3, 4, 5])
    noise_factor = random((n, n))
    noise = noise_factor@noise_factor.conj().T+np.eye(n)
    return model, coefficients, y, known, np.array([.3, 1.1]), noise, data


def test_raw_detector_matches_direct_frequency_domain_full_matrix(synthetic):
    model, coefficients, y, known, variances, noise, data = synthetic
    result = detect_raw(*synthetic, DenseBackend("cpu"))
    expected = np.empty_like(result)
    for snr in range(len(variances)):
        for symbol in range(4):
            channel = taps_to_matrix(model.reconstruct(coefficients[snr], symbol))
            h = channel[:, data]
            residual = y[snr, symbol]-channel@known[symbol]
            whitened = linalg.solve(noise, np.column_stack((h, residual)), assume_a="pos")
            gram = h.conj().T@whitened[:, :-1]+variances[snr]*np.eye(data.sum())
            expected[snr, symbol] = linalg.solve(gram, h.conj().T@whitened[:, -1], assume_a="pos")
    np.testing.assert_allclose(result, expected, atol=2e-12, rtol=2e-12)
    assert not ({"truth", "reference", "alpha", "labels"} & set(signature(detect_raw).parameters))


def test_cuda_raw_detector_matches_cpu_full_matrix(synthetic):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cpu = detect_raw(*synthetic, DenseBackend("cpu"))
    gpu = detect_raw(*synthetic, DenseBackend("cuda"))
    np.testing.assert_allclose(gpu, cpu, atol=2e-12, rtol=2e-12)
