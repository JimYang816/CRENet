"""Check the oracle compensation against independently formed channel matrices."""
import numpy as np
import pytest

from data_generation.src.receiver.chain import taps_to_matrix
from src.common.numerics import DenseBackend
from src.evaluation.four_data_csi_compensation import ALPHA, compensate, tap_nmse
from src.evaluation.four_data_lmmse import detect


def test_compensation_shrinks_complete_matrix_error_by_five_db():
    rng = np.random.default_rng(2505)
    n, length = 8, 3
    shape = (2, 4, n, length)
    estimated = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    truth = rng.normal(size=(4, n, n)) + 1j*rng.normal(size=(4, n, n))
    original_estimated, original_truth = estimated.copy(), truth.copy()
    corrected = compensate(estimated, truth)
    padded = np.pad(estimated, ((0, 0), (0, 0), (0, 0), (0, n-length)))
    raw_nmse = np.empty((2, 4))
    for snr in range(2):
        for symbol in range(4):
            reference_matrix = taps_to_matrix(truth[symbol])
            raw_matrix = taps_to_matrix(padded[snr, symbol])
            compensated_matrix = taps_to_matrix(corrected[snr, symbol])
            np.testing.assert_allclose(compensated_matrix, reference_matrix + ALPHA*(raw_matrix-reference_matrix),
                                       rtol=1e-13, atol=1e-13)
            raw_nmse[snr, symbol] = np.linalg.norm(raw_matrix-reference_matrix)**2 / np.linalg.norm(reference_matrix)**2
    np.testing.assert_allclose(tap_nmse(corrected, truth), raw_nmse*10**(-.5), rtol=1e-13)
    np.testing.assert_array_equal(estimated, original_estimated)
    np.testing.assert_array_equal(truth, original_truth)
    np.testing.assert_allclose(corrected[..., length:],
                               np.broadcast_to((1-ALPHA)*truth[None, ..., length:], corrected[..., length:].shape))


@pytest.mark.parametrize("estimated,reference", [((2, 4, 8, 3), (3, 8, 8)),
                                                   ((2, 4, 8, 9), (4, 8, 8)),
                                                   ((4, 8, 3), (4, 8, 8))])
def test_compensation_rejects_incompatible_shapes(estimated, reference):
    with pytest.raises(ValueError, match="dimensions"):
        compensate(np.zeros(estimated), np.zeros(reference))


def test_nmse_rejects_zero_reference_energy():
    with pytest.raises(ValueError, match="energy"):
        tap_nmse(np.zeros((2, 4, 8, 8)), np.zeros((4, 8, 8)))


def test_both_detectors_consume_the_compensated_full_support():
    rng = np.random.default_rng(442)
    random = lambda shape: rng.normal(size=shape) + 1j*rng.normal(size=shape)
    n = 8
    corrected = compensate(random((2, 4, n, 3)), random((4, n, n)))
    y, known = random((2, 4, n)), np.zeros((4, n), dtype=complex)
    known[:, [0, 7]] = random((4, 2))
    mask = np.isin(np.arange(n), [1, 2, 3, 4])
    variance = np.array([.3, .7])
    single, full = detect(corrected, y, known, variance, np.eye(n), mask, DenseBackend("cpu"))
    for snr in range(2):
        for symbol in range(4):
            matrix = taps_to_matrix(corrected[snr, symbol])
            diagonal, h = np.diag(matrix)[mask], matrix[:, mask]
            expected_single = diagonal.conj()*y[snr, symbol, mask] / (abs(diagonal)**2 + variance[snr])
            expected_full = np.linalg.solve(h.conj().T@h + variance[snr]*np.eye(mask.sum()),
                                            h.conj().T@(y[snr, symbol] - matrix@known[symbol]))
            np.testing.assert_allclose(single[snr, symbol], expected_single, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(full[snr, symbol], expected_full, rtol=1e-12, atol=1e-12)
