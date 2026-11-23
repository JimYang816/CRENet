"""Exercise the fixed four-data setup and the actual detector contracts."""
from inspect import signature

import numpy as np
import pytest
from scipy import linalg

from data_generation.src.config import ROOT
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.pdd import JointEstimator, symbol_layout
from src.common.numerics import DenseBackend
from src.evaluation.four_data_lmmse import channel_config, detect, load_run_config, model_basis
from src.evaluation.four_data_q_sweep import make_frame, masks


def test_four_data_parameters_and_independent_test_records():
    options = load_run_config(ROOT/"configs/evaluation/four_data_lmmse.json")
    assert set(options["prior_records"]).isdisjoint(options["test_records"])
    for channel in options["channels"]:
        c = channel_config(channel)
        assert (c["frame_structure"], c["data_count"], c["pilot_count"]) == ("DDDD", 512, 256)
        assert (c["dpss_half_bandwidth_hz"], c["dpss_expected_count"], c["delay_length"]) == (12, 32, 176)
        assert c["csi_model_error_ratio"] == .03
        assert symbol_layout(c) == ((0, 1, 2, 3), ())
        carriers = masks(c)
        assert (carriers["data"].sum(), carriers["pilot"].sum(), carriers["protection"].sum(), carriers["edge"].sum()) == (512, 256, 128, 128)
        frame = make_frame(c, options["test_records"][0], options["start_seconds"], 2)
        assert frame.bits.shape == (4, 512, 2)
        assert np.all(frame.known[:, carriers["data"]] == 0)
        assert np.all(frame.x[:, carriers["null"]] == 0)
        basis, concentration = model_basis(c)
        assert basis.shape == (5632, 32)
        np.testing.assert_allclose(basis.T@basis, np.eye(32), atol=2e-13)
        assert np.all(np.diff(concentration) <= 1e-12)


def test_four_data_noise_floor_uses_data_frame_power():
    estimator = JointEstimator.__new__(JointEstimator)
    estimator.known = np.zeros((4, 8), complex)
    estimator.known[:, :2] = 1
    estimator.masks = {"data": np.array([False, False, True, True, True, True, False, False])}
    estimator.data_symbols, estimator.training_symbols = (0, 1, 2, 3), ()
    estimator.model_error_ratio, estimator.noise_gain = .03, 2.0
    y = np.full((2, 4, 8), 4+0j)
    variance = np.array([1., 3.])
    actual = estimator.effective_noise_variances(y, variance)
    np.testing.assert_allclose(actual, variance+.03*(8-variance))


@pytest.fixture
def problem():
    rng = np.random.default_rng(812)
    n = 12
    random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    taps = random((2, 4, n, 4))
    observations = random((2, 4, n))
    known = np.zeros((4, n), complex)
    known[:, [2, 8]] = random((4, 2))
    data = np.isin(np.arange(n), [3, 4, 5])
    a = random((n, n))
    noise = a@a.conj().T+np.eye(n)
    return taps, observations, known, np.array([.3, 1.1]), noise, data


def test_both_detectors_match_independent_frequency_domain_solutions(problem):
    taps, y, known, variances, noise, mask = problem
    single, full = detect(*problem, DenseBackend("cpu"))
    for si in range(2):
        for d in range(4):
            h = taps_to_matrix(taps[si, d])
            diag = np.diag(h)[mask]
            expected_single = diag.conj()*y[si, d, mask]/(abs(diag)**2+variances[si]*np.diag(noise).real[mask])
            hd = h[:, mask]
            residual = y[si, d]-h@known[d]
            whitened = linalg.solve(noise, np.column_stack((hd, residual)), assume_a="pos")
            expected_full = linalg.solve(hd.conj().T@whitened[:, :-1]+variances[si]*np.eye(mask.sum()),
                                         hd.conj().T@whitened[:, -1], assume_a="pos")
            np.testing.assert_allclose(single[si, d], expected_single, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(full[si, d], expected_full, atol=2e-12, rtol=2e-12)
    assert not ({"truth", "reference", "alpha", "labels", "bits"} & set(signature(detect).parameters))


@pytest.mark.parametrize("shared_reference", (False, True))
def test_cuda_matches_cpu_for_per_snr_and_shared_reference(problem, shared_reference):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    taps, *rest = problem
    args = ((taps[0] if shared_reference else taps), *rest)
    cpu = detect(*args, DenseBackend("cpu"))
    gpu = detect(*args, DenseBackend("cuda"))
    for a, b in zip(cpu, gpu, strict=True):
        np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-12)
