"""Physical cross-D columns and PSD joint-moment LMMSE independent checks."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest
from scipy import linalg
from threadpoolctl import threadpool_limits

from data_generation.src.channel.watermark.replay import Sounding
from data_generation.src.config import load_config
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.four_data import BlockedFourDataReceiver, fixed_four_data_model
from data_generation.src.receiver.four_data_covariance import (
    JointFourDataEstimator, JointMoments, collect_joint_moments, physical_data_columns,
    reference_taps_from_rows, selected_fft,
)
from data_generation.src.receiver.four_data_rotated import full_band_layout, waveform
from data_generation.src.transmitter.ofdm import Frame, qpsk, receive_symbols, window_indices
from src.common.numerics import DenseBackend


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(2):
        yield


@pytest.fixture
def setup():
    config = load_config()
    config.update(nfft=56, cp=32, fft_advance=2, guard_samples=32, null_per_edge=0,
                  data_count=32, pilot_count=8)
    receiver = BlockedFourDataReceiver(config, row_block=31)
    model = fixed_four_data_model(config, count=2, length=8)
    factor = linalg.cholesky(model.sampled_basis.conj().T @ model.sampled_basis)
    model.basis = model.basis @ linalg.inv(factor) @ (np.array([[1, 1j], [1j, 1]])/np.sqrt(2))
    known, masks = full_band_layout(config)
    time = np.arange(128)/250
    soundings = {}
    for record, amplitude in ((51, .8+.2j), (52, 1.1-.1j), (53, .7-.4j)):
        h = np.zeros((128, 128), complex)
        h[:, 0] = amplitude*np.exp(2j*np.pi*.7*time)
        h[:, 40] = .3j*np.exp(-2j*np.pi*.4*time)
        h[:, 96] = .1*np.exp(2j*np.pi*1.1*time)
        soundings[record] = Sounding(h, 250, 16000, 14000, 0, "synthetic")
    identities = [(r, .02) for r in soundings]
    sample_indices = np.concatenate([window_indices(config, s) for s in range(4)])
    coefficients = []
    for record, start in identities:
        direct, _ = receiver.reference_parts(soundings[record], start-config["guard_samples"]/6000,
                                               sample_indices, include_image=False)
        truth = np.stack([reference_taps_from_rows(direct, config, s) for s in range(4)])
        coefficients.append(model.sampled_basis.conj().T @ truth[..., :8].reshape(224, 8))
    prior = FrozenPrior.fit_coefficients(model, coefficients, 0, 1, tuple(soundings), delay_block_size=4)
    return config, receiver, prior, known, masks, soundings, identities, sample_indices


def test_full_physical_columns_include_cross_symbol_and_image_paths(setup):
    c, receiver, prior, known, masks, soundings, identities, indices = setup
    sounding, start = soundings[51], identities[0][1]
    origin = start-c["guard_samples"]/6000
    a, image = receiver.reference_parts(sounding, origin, indices)
    full_a, full_b = receiver.chain.operators(sounding, origin)
    np.testing.assert_allclose(a.toarray(), full_a[indices].toarray(), rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(image.toarray(), full_b[indices].toarray(), rtol=2e-12, atol=2e-14)
    for s in range(4):
        np.testing.assert_allclose(reference_taps_from_rows(a, c, s), receiver.reference_taps(sounding, start, s)[0], atol=2e-14)
    rows = [np.flatnonzero(m) for m in masks["tracking"]]
    rng = np.random.default_rng(201)
    x = known.copy()
    predicted = selected_fft(a @ waveform(known, c)+image @ waveform(known, c).conj(), c, rows)
    source_first = None
    for s in range(4):
        data = qpsk(rng.integers(0, 2, (14, 2)))
        x[s, masks["data"][s]] = data
        direct, conjugate = physical_data_columns(a, image, c, masks, rows, s)
        predicted += direct @ data+conjugate @ data.conj()
        if s == 0:
            source_first = direct
    frame = Frame(x, known, np.empty((4, 14, 2), np.uint8), waveform(x, c), ())
    replay = receiver.replay_frame(sounding, origin, frame)
    received = receive_symbols(receiver.chain.receive(replay), c)
    actual = np.concatenate([received[s, r] for s, r in enumerate(rows)])
    np.testing.assert_allclose(predicted, actual, rtol=2e-11, atol=2e-13)
    assert np.sum(abs(source_first[len(rows[0]):])**2) > 1e-6


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_joint_moments_match_full_unwhitened_dense_lmmse(setup, device, tmp_path):
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    c, receiver, prior, known, masks, soundings, identities, indices = setup
    archive = SimpleNamespace(load=lambda record:soundings[record], channel="synthetic")
    noise = receiver.chain.noise_covariance
    backend = DenseBackend(device)
    raw, metadata = collect_joint_moments(prior, known, masks, noise, receiver, archive, identities, backend)
    assert metadata["physical_data_off_diagonal_D_fraction"] > 1e-4
    assert abs(sum(f["data_in_out_cross_trace"] for f in metadata["frames"])) > 1e-7
    moments = JointMoments(prior, known, masks, raw, metadata, backend)
    alpha, rho = .4, .25
    estimator = JointFourDataEstimator(moments, alpha, rho)
    rows = estimator.rows
    basis = prior.model.sampled_basis.reshape(4, 56, 2)
    a = []
    for s in range(4):
        time = (basis[s, :, :, None]*input_windows(known, 8)[s, :, None, :]).reshape(56, 16)
        a.append(np.fft.fftshift(np.fft.fft(time, axis=0, norm="ortho"), axes=0)[rows[s]])
    a = np.concatenate(a)
    p = np.zeros((16, 16), complex)
    for select, temporal, delay in prior.separable_blocks():
        ix = (np.arange(2)[:, None]*8+np.arange(8)[select]).ravel()
        p[np.ix_(ix, ix)] = np.kron(temporal, delay)
    coeff, residual = raw["coefficients"], raw["pilot_residual"]
    p = (1-alpha)*p+alpha*coeff @ coeff.conj().T/3
    cr = np.sqrt(alpha*(1-rho))*coeff @ residual.conj().T/3
    rr = (1-rho)*residual @ residual.conj().T/3+rho*moments.residual_variance*raw["selected_noise"]
    received = a @ p @ a.conj().T+a @ cr+cr.conj().T @ a.conj().T+rr+raw["physical_data_covariance"]
    cross = p @ a.conj().T+cr
    joint = np.block([[p, cr], [cr.conj().T, rr]])
    assert linalg.eigvalsh(joint).min() > -1e-11
    rng = np.random.default_rng(21)
    y = rng.normal(size=(3, 4, 56))+1j*rng.normal(size=(3, 4, 56))
    variances = np.array([.0, .01, .2])
    expected = []
    for yi, variance in zip(y, variances, strict=True):
        selected = np.concatenate([yi[s, r] for s, r in enumerate(rows)])
        weights = linalg.solve(received+(variance+estimator.ridge)*raw["selected_noise"], selected, assume_a="her")
        expected.append((cross @ weights).reshape(2, 8))
    actual = estimator.estimate_coefficients(y, variances)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
    from src.evaluation.four_data_covariance import load_estimator, save_estimator
    path = tmp_path/"frozen_estimator.h5"
    save_estimator(path, estimator, "independent-unit-case", {"selected": {"alpha":alpha, "rho":rho}})
    restored, selection = load_estimator(path, prior, known, masks, backend, "independent-unit-case")
    np.testing.assert_allclose(restored.estimate_coefficients(y, variances), actual, rtol=2e-12, atol=2e-13)
    assert selection["selected"] == {"alpha":alpha, "rho":rho}


def test_calibration_rejects_test_records_before_loading(setup):
    c, receiver, prior, known, masks, soundings, identities, indices = setup
    archive = SimpleNamespace(load=lambda record:pytest.fail("No test record may be loaded"), channel="synthetic")
    with pytest.raises(ValueError, match="independent prior"):
        collect_joint_moments(prior, known, masks, receiver.chain.noise_covariance, receiver,
                              archive, [(11, .02)], DenseBackend())
