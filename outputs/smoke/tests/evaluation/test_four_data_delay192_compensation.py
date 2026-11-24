"""Target curve, full-matrix calibration and CPU detector checks."""
import numpy as np
import pytest

from data_generation.src.receiver.chain import taps_to_matrix
from src.common.numerics import DenseBackend
from src.evaluation.four_data_csi_compensation import tap_nmse
from src.evaluation.four_data_delay192_compensation import (
    compensate, error_scales, load_target_config, target_curve,
)
from src.evaluation.four_data_lmmse import detect


@pytest.mark.parametrize("channel,endpoints", [("NOF1", (-10, -16)), ("NCS1", (-5, -10))])
def test_requested_targets_decrease_steeply_then_gently(channel, endpoints):
    options = load_target_config()
    target = target_curve(options["snr_db"], **options["targets"][channel], decay_db=options["decay_db"])
    assert (target[0], target[-1]) == endpoints
    changes = -np.diff(target)
    assert np.all(changes > 0) and np.all(np.diff(changes) < 0)


def test_full_matrix_mean_hits_targets_and_preserves_frame_error_ratios():
    rng = np.random.default_rng(321)
    truth = rng.normal(size=(2, 4, 8, 8))+1j*rng.normal(size=(2, 4, 8, 8))
    # The raw support omits four reference taps and differs across frames/symbols.
    raw = rng.normal(size=(2, 7, 4, 8, 4))+1j*rng.normal(size=(2, 7, 4, 8, 4))
    padded = np.pad(raw, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 4)))
    raw_nmse = np.stack([tap_nmse(padded[i], truth[i]) for i in range(2)])
    targets = target_curve(np.arange(0, 31, 5), -10, -16)
    scales = error_scales(raw_nmse.mean(axis=(0, 2)), targets)
    corrected = np.stack([compensate(raw[i], truth[i], scales) for i in range(2)])
    measured = np.stack([tap_nmse(corrected[i], truth[i]) for i in range(2)])
    np.testing.assert_allclose(10*np.log10(measured.mean(axis=(0, 2))), targets, atol=1e-12)
    np.testing.assert_allclose(measured/raw_nmse, np.broadcast_to(scales[None, :, None]**2, measured.shape), rtol=1e-12)
    # Validate the Frobenius identity against actual dense frequency matrices.
    si, frame, symbol = 3, 1, 2
    actual = taps_to_matrix(corrected[frame, si, symbol])
    reference = taps_to_matrix(truth[frame, symbol])
    dense_nmse = np.sum(abs(actual-reference)**2)/np.sum(abs(reference)**2)
    np.testing.assert_allclose(dense_nmse, measured[frame, si, symbol], rtol=1e-12)


def test_reject_targets_that_worsen_raw_csi():
    with pytest.raises(ValueError, match="improve"):
        error_scales([0.001], [-10])


@pytest.mark.parametrize("snr,start,end,decay", [([0, 30, 15], -10, -16, 10), ([0, 30], -16, -10, 10), ([0, 30], -10, -16, 0)])
def test_reject_invalid_curves(snr, start, end, decay):
    with pytest.raises(ValueError):
        target_curve(snr, start, end, decay)


def test_cpu_compensated_csi_drives_both_lmmse_detectors():
    n = 16
    reference = np.zeros((4, n, n), complex)
    reference[..., 0] = 1
    raw = np.full((2, 4, n, 1), 1+0.5j)
    scales = error_scales([.25, .25], [-10, -16])
    corrected = compensate(raw, reference, scales)
    symbols = np.full((4, n), (1+1j)/np.sqrt(2))
    observations = np.broadcast_to(symbols, (2, 4, n))
    single, full = detect(corrected, observations, np.zeros((4, n), complex),
        np.array([.1, .001]), np.eye(n), np.ones(n, dtype=bool), DenseBackend("cpu"))
    assert single.shape == full.shape == (2, 4, n)
    assert np.all(np.isfinite(single)) and np.all(np.isfinite(full))
    np.testing.assert_allclose(single, full, rtol=1e-12, atol=1e-12)
