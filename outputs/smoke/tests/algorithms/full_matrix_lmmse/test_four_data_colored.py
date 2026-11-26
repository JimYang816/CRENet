"""Four-data scorer versus independent colored covariance normal equations."""
import numpy as np

from data_generation.src.receiver.chain import taps_to_matrix
from src.common.numerics import DenseBackend
from src.evaluation.four_data_lmmse import detect


def test_lmmse_matches_independent_colored_covariance_normal_equations():
    rng = np.random.default_rng(25)

    def complex_normal(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)

    taps = complex_normal((2, 4, 10, 5))
    observations = complex_normal((2, 4, 10))
    known = np.zeros((4, 10), dtype=complex)
    known[:, 1] = 1+1j
    data = np.arange(10) % 2 == 0
    factor = complex_normal((10, 10))
    covariance = factor@factor.conj().T + .4*np.eye(10)
    variance = np.array([.1, .001])

    _, actual = detect(taps, observations, known, variance, covariance, data, DenseBackend())
    for si, noise in enumerate(variance):
        for d in range(4):
            channel = taps_to_matrix(taps[si, d])
            data_channel = channel[:, data]
            cancelled = observations[si, d] - channel@known[d]
            gram = data_channel@data_channel.conj().T + noise*covariance
            reference = data_channel.conj().T@np.linalg.solve(gram, cancelled)
            np.testing.assert_allclose(actual[si, d], reference, rtol=2e-12, atol=2e-12)
