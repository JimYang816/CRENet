"""Independent receive-space checks of the physical-index B16 approximation."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.src.receiver.chain import taps_to_matrix
from src.algorithms.banded_lmmse.equalizer import band_mask, equalize_batch
from src.algorithms.interface import EqualizerBatch
from src.common.numerics import DenseBackend
from src.evaluation.four_data_lmmse import detect


def fixture():
    rng = np.random.default_rng(948)
    cn = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    h, y, raw = cn((2, 48, 48)), cn((2, 48)), cn((2, 48))
    factor = cn((48, 48))
    covariance = factor @ factor.conj().T + np.eye(48)
    return EqualizerBatch(
        h, raw, y, np.array([0.07, 0.5]), np.array([2, 7, 19, 29, 44]), covariance
    )


def test_band_uses_physical_fft_spacing_and_exactly_33_diagonals():
    indices = np.array([2, 20, 47, 78])
    mask = band_mask(80, indices)
    assert mask[36, 1] and not mask[37, 1]
    assert mask[31, 2] and not mask[30, 2]
    assert mask[:, 1].sum() == mask[:, 2].sum() == 33
    assert not mask[0, 2]  # Compressed data-column position 2 is not FFT index 47.


@pytest.mark.parametrize("bandwidth", [-1, True, 1.5])
def test_invalid_half_bandwidth_is_rejected(bandwidth):
    with pytest.raises(ValueError):
        band_mask(48, np.array([1, 4]), bandwidth)


def test_banded_solution_matches_independent_receive_covariance_with_colored_noise():
    batch = fixture()
    expected = []
    for h, y, v in zip(
        batch.channel, batch.cancelled_observations, batch.variances, strict=True
    ):
        truncated = np.zeros((48, len(batch.data_indices)), complex)
        for j, k in enumerate(batch.data_indices):
            for r in range(48):
                if abs(r - int(k)) <= 16:
                    truncated[r, j] = h[r, k]
        expected.append(
            truncated.conj().T
            @ np.linalg.solve(
                truncated @ truncated.conj().T + v * batch.noise_covariance, y
            )
        )
    with threadpool_limits(2):
        actual = equalize_batch(batch, DenseBackend()).symbols
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
    # Raw observations are distinct; the detector must consume pilot-cancelled y.
    assert not np.array_equal(batch.raw_observations, batch.cancelled_observations)


def test_four_symbol_detection_cancels_full_pilots_before_banding():
    rng = np.random.default_rng(52)
    taps = rng.normal(size=(2, 4, 40, 5)) + 1j * rng.normal(size=(2, 4, 40, 5))
    y = rng.normal(size=(2, 4, 40)) + 1j * rng.normal(size=(2, 4, 40))
    known = np.zeros((4, 40), complex)
    known[:, 3] = 1j
    data = np.zeros(40, bool)
    data[[1, 8, 22, 37]] = True
    covariance = 0.6 ** abs(np.arange(40)[:, None] - np.arange(40))
    variance = np.array([0.1, 0.8])
    with threadpool_limits(2):
        actual = detect(taps, y, known, variance, covariance, data, DenseBackend())
    for si in range(2):
        for d in range(4):
            h = taps_to_matrix(taps[si, d])
            hb = h[:, data].copy()
            for j, k in enumerate(np.flatnonzero(data)):
                hb[abs(np.arange(40) - k) > 16, j] = 0
            expected = hb.conj().T @ np.linalg.solve(
                hb @ hb.conj().T + variance[si] * covariance, y[si, d] - h @ known[d]
            )
            np.testing.assert_allclose(actual[si, d], expected, rtol=2e-11, atol=2e-11)


def test_cuda_matches_cpu_for_banded_colored_noise_solve():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable locally")
    batch = fixture()
    with threadpool_limits(2):
        cpu = equalize_batch(batch, DenseBackend()).symbols
        gpu = equalize_batch(batch, DenseBackend("cuda")).symbols
    np.testing.assert_allclose(gpu, cpu, rtol=2e-11, atol=2e-11)
