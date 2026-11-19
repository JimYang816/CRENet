"""Independent physical and dense-covariance checks for the isolated 4D study."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg
from threadpoolctl import threadpool_limits

from data_generation.src.channel.watermark.replay import Sounding
from data_generation.src.config import load_config
from data_generation.src.receiver.chain import full_frame_kernels, taps_to_matrix
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.four_data import (
    BlockedFourDataReceiver, FourDataEstimator, fixed_four_data_model, make_four_data_frame,
)
from data_generation.src.transmitter.ofdm import carrier_masks, fft, qpsk, receive_symbols
from src.algorithms.likelihood import make_whitener
from src.common.numerics import DenseBackend
from src.evaluation.four_data_mmse import score_window, window_ids


@pytest.fixture(autouse=True)
def limit_threads():
    with threadpool_limits(2):
        yield


def small_config():
    config = load_config()
    config.update(nfft=14, cp=6, fft_advance=1, guard_samples=6, null_per_edge=0,
                  data_count=8, pilot_count=2, delay_length=3)
    return config


def test_four_data_waveform_has_no_public_payload_or_training_symbol():
    config = load_config()
    masks = carrier_masks(config)
    first = make_four_data_frame(config, 11, 2)
    other = make_four_data_frame(config, 15, 14)
    assert first.bits.shape == (4, 512, 2)
    assert first.bits.size == 4096
    assert first.baseband.shape == (6232,)
    np.testing.assert_array_equal(first.known[:, masks["data"]], 0)
    np.testing.assert_array_equal(first.known, other.known)
    np.testing.assert_array_equal(first.x[:, masks["data"]], qpsk(first.bits))
    np.testing.assert_array_equal(first.x[:, masks["null"]], 0)
    time = first.baseband[300:-300].reshape(4, 1408)
    np.testing.assert_allclose(fft(time[:, 384:]), first.x, atol=2e-15)
    np.testing.assert_array_equal(time[:, :384], time[:, -384:])
    assert not np.array_equal(first.bits, make_four_data_frame(config, 11, 2, repeat=1).bits)
    model = fixed_four_data_model(config)
    assert model.basis.shape == (5632, 24)
    assert model.sampled_basis.shape == (4096, 24)
    assert model.bandwidth_hz == 12 and model.length == 176
    assert model.tail_ratio > 1e-5  # Q is fixed explicitly, not selected by the old tail rule.
    prior, test = window_ids(config, False)
    assert len(prior) == 10 and len(test) == 30
    assert {r for r, _ in test} == set(config["learning_dataset"]["records"]["test"])
    assert not {r for r, _ in test} & {r for r, _ in prior}


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_joint_four_data_estimate_matches_independent_dense_lmmse(device):
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    config = small_config()
    rng = np.random.default_rng(121)
    model = fixed_four_data_model(config, count=2, length=3)
    coefficients = rng.normal(size=(20, 2, 3))+1j*rng.normal(size=(20, 2, 3))
    prior = FrozenPrior.fit_coefficients(model, coefficients, 0, 1, (51,), delay_block_size=2)
    masks = carrier_masks(config)
    known = make_four_data_frame(config, 11, 2).known
    n, q, length = config["nfft"], model.count, model.length
    rows = np.flatnonzero(masks["tracking"])
    covariance = np.zeros((q*length, q*length), dtype=complex)
    for select, temporal, delay in prior.separable_blocks():
        indices = (np.arange(q)[:, None]*length+np.arange(length)[select]).ravel()
        covariance[np.ix_(indices, indices)] = np.kron(temporal, delay)
    basis = model.sampled_basis.reshape(4, n, q)

    def response(symbol, x):
        delayed = input_windows(x[None], length)[0]
        columns = (basis[symbol, :, :, None]*delayed[:, None, :]).reshape(n, q*length)
        return np.fft.fftshift(np.fft.fft(columns, axis=0, norm="ortho"), axes=0)[rows]

    public_response = np.concatenate([response(symbol, x) for symbol, x in enumerate(known)])
    data_covariance = np.zeros((4*len(rows), 4*len(rows)), dtype=complex)
    # Sum the response covariance of every unknown unit-energy carrier,
    # without using the estimator's stationary-kernel identity.
    for symbol in range(4):
        select = slice(symbol*len(rows), (symbol+1)*len(rows))
        for carrier in np.flatnonzero(masks["data"]):
            x = np.zeros(n, dtype=complex)
            x[carrier] = 1
            a = response(symbol, x)
            data_covariance[select, select] += a @ covariance @ a.conj().T
    factor = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    noise = factor @ factor.conj().T/n+np.eye(n)
    estimator = FourDataEstimator(prior, known, masks, noise, DenseBackend(device))
    y = rng.normal(size=(3, 4, n))+1j*rng.normal(size=(3, 4, n))
    variances = np.array([0.0, 0.1, 0.7])
    received_covariance = public_response @ covariance @ public_response.conj().T+data_covariance
    cross = covariance @ public_response.conj().T
    selected_noise = linalg.block_diag(*([noise[np.ix_(rows, rows)]]*4))
    expected = []
    for observation, variance in zip(y, variances, strict=True):
        weights = linalg.solve(received_covariance+(variance+estimator.ridge)*selected_noise,
                                observation[:, rows].ravel(), assume_a="her")
        expected.append((cross @ weights).reshape(q, length))
    actual = estimator.estimate_coefficients(y, variances)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
    hidden = y.copy()
    hidden[..., ~masks["tracking"]] += 1e6
    np.testing.assert_array_equal(estimator.estimate_coefficients(hidden, variances), actual)
    bad = known.copy()
    bad[0, np.flatnonzero(masks["data"])[0]] = 1
    with pytest.raises(ValueError, match="hide their unknown data"):
        FourDataEstimator(prior, bad, masks, noise, DenseBackend())


def test_blocked_physical_replay_reference_and_prior_match_full_operators():
    config = small_config()
    time = np.arange(128)/250
    h = np.zeros((128, 64), dtype=complex)
    h[:, 0] = np.exp(2j*np.pi*0.7*time)
    h[:, 40] = 0.2j*np.exp(-2j*np.pi*0.4*time)
    sounding = Sounding(h, 250, 16000, 14000, 0, "synthetic")
    receiver = BlockedFourDataReceiver(config, row_block=9, native_block=11)
    frame = make_four_data_frame(config, 11, .01)
    origin = .01-config["guard_samples"]/config["fs_baseband"]
    chain = receiver.chain
    direct, _ = chain.operators(sounding, origin)
    actual = receiver.replay_frame(sounding, origin, frame)
    expected = chain.replay(chain.transmit(frame.baseband), sounding, origin)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    for symbol in range(4):
        matrix, taps, leakage = chain.reference(direct, symbol)
        reference, blocked_leakage = receiver.reference_taps(sounding, .01, symbol)
        np.testing.assert_allclose(reference, taps, rtol=2e-13, atol=2e-14)
        assert blocked_leakage == pytest.approx(leakage, abs=1e-15)
        np.testing.assert_allclose(taps_to_matrix(reference), matrix, rtol=2e-13, atol=2e-14)
    model = fixed_four_data_model(config, count=2, length=3)
    expected = model.basis.conj().T @ full_frame_kernels(direct, config, model.length)
    np.testing.assert_allclose(receiver.prior_coefficients(sounding, .01, model), expected,
                                rtol=2e-13, atol=2e-14)
    assert receive_symbols(chain.receive(actual), config).shape == (4, 14)


def test_mmse_scoring_uses_all_four_symbols_and_full_reference_energy():
    config = small_config()
    model = fixed_four_data_model(config, count=2, length=3)
    frame = make_four_data_frame(config, 11, 2)
    fitted = np.zeros((2, model.count, model.length), dtype=complex)
    fitted[:, 0, 0] = 4
    truth = np.zeros((4, config["nfft"], config["nfft"]), dtype=complex)
    observations = np.zeros((2, 4, config["nfft"]), dtype=complex)
    for symbol in range(4):
        truth[symbol, :, :model.length] = model.reconstruct(fitted[0], symbol)
        truth[symbol, :, 8] = .5  # Outside L; must still enter the NMSE denominator and numerator.
        matrix = taps_to_matrix(model.reconstruct(fitted[0], symbol))
        observations[:, symbol] = matrix @ frame.x[symbol]
    noise = np.eye(config["nfft"], dtype=complex)
    backend = DenseBackend()
    nmse, errors = score_window(model, fitted, truth, observations, frame.known, frame.bits,
                                np.array([1e-4, 1e-3]), noise, backend, make_whitener(noise, backend),
                                verify_matrix_nmse=True)
    for symbol in range(4):
        expected = np.sum(abs(truth[symbol, :, model.length:])**2)/np.sum(abs(truth[symbol])**2)
        np.testing.assert_allclose(nmse[:, symbol], expected, rtol=1e-13)
    assert errors["full_matrix_mmse"].shape == (2, 4)
    np.testing.assert_array_equal(errors["full_matrix_mmse"], 0)
