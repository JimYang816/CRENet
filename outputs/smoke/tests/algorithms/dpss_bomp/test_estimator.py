import inspect
import os

import numpy as np
import pytest

from data_generation.src.receiver.chain import taps_to_matrix
from outputs.smoke.data_generation_tests.test_pdd import small_fixture
from src.algorithms.dpss_bomp.estimator import GroupOMP, build_dictionary, whiten_observations
from src.common.numerics import DenseBackend


def test_dictionary_matches_direct_pilot_transmission_after_whitening():
    prior, known, masks, noise, _ = small_fixture()
    model = prior.model
    backend = DenseBackend("cpu")
    blocks = [(s.start, s.stop, t, d) for s, t, d in prior.separable_blocks()]
    dictionary, transforms, rows, whiteners = build_dictionary(
        model, known, masks, noise, blocks, backend)
    rng = np.random.default_rng(712)
    z = rng.normal(size=(model.length, model.count))+1j*rng.normal(size=(model.length, model.count))
    coefficients = np.einsum("lqr,lr->ql", transforms, z)
    signals = np.stack([taps_to_matrix(model.reconstruct(coefficients, s)) @ known[s] for s in range(4)])
    expected = whiten_observations(signals[None], rows, whiteners, backend)[0]
    np.testing.assert_allclose(dictionary.reshape(len(expected), -1) @ z.ravel(), expected, atol=1e-12)
    contaminated = known.copy()
    contaminated[1, np.flatnonzero(masks["data"])[0]] = 1
    with pytest.raises(ValueError, match="Data tones"):
        build_dictionary(model, contaminated, masks, noise, blocks, backend)


def test_group_omp_recovers_two_groups_and_refits_all_coefficients():
    m, groups, q = 12, 4, 2
    dictionary = np.eye(m, groups*q, dtype=complex).reshape(m, groups, q)
    transforms = np.tile(np.eye(q), (groups, 1, 1)).astype(complex)
    truth = np.zeros((groups, q), complex)
    truth[1] = (3+2j, 1-1j)
    truth[3] = (-1, .5j)
    y = (dictionary.reshape(m, -1) @ truth.ravel())[None]
    estimator = GroupOMP(dictionary, transforms, DenseBackend("cpu"))
    outputs = estimator.estimate(y, np.array([1e-8]), (1, 2))
    assert np.count_nonzero(np.linalg.norm(outputs[1][0], axis=0) > 1e-4) == 1
    np.testing.assert_allclose(outputs[2][0].T, truth, atol=5e-8)
    assert not any(name in inspect.signature(estimator.estimate).parameters
                   for name in ("reference", "truth", "bits", "labels"))


def test_ridge_refit_matches_direct_selected_normal_equations():
    rng = np.random.default_rng(17)
    m, groups, q = 12, 5, 2
    dictionary = rng.normal(size=(m, groups, q))+1j*rng.normal(size=(m, groups, q))
    transforms = np.tile(np.eye(q), (groups, 1, 1)).astype(complex)
    y = rng.normal(size=(1, m))+1j*rng.normal(size=(1, m))
    regularizer = .3
    output = GroupOMP(dictionary, transforms, DenseBackend("cpu")).estimate(y, np.array([regularizer]), (3,))[3]
    active = np.flatnonzero(np.linalg.norm(output[0], axis=0) > 1e-12)
    matrix = dictionary[:, active, :].reshape(m, -1)
    expected = np.linalg.solve(matrix.conj().T@matrix+regularizer*np.eye(len(active)*q), matrix.conj().T@y[0])
    np.testing.assert_allclose(output[0, :, active].ravel(), expected, atol=1e-12)


@pytest.mark.skipif(os.environ.get("CRENET_TEST_CUDA") != "1", reason="explicit server GPU verification")
def test_cuda_incremental_refit_agrees_with_cpu():
    import torch
    rng = np.random.default_rng(21)
    m, groups, q = 24, 7, 3
    a = rng.normal(size=(m, groups, q))+1j*rng.normal(size=(m, groups, q))
    transform = np.tile(np.eye(q), (groups, 1, 1)).astype(complex)
    y = rng.normal(size=(3, m))+1j*rng.normal(size=(3, m))
    ridge = np.array([.2, .4, .7])
    cpu = GroupOMP(a, transform, DenseBackend()).estimate(y, ridge, (1, 2, 3))
    gpu_backend = DenseBackend("cuda")
    gpu = GroupOMP(torch.as_tensor(a, device="cuda", dtype=torch.complex64),
                   torch.as_tensor(transform, device="cuda", dtype=torch.complex64), gpu_backend)
    got = gpu.estimate(torch.as_tensor(y, device="cuda", dtype=torch.complex64),
                       torch.as_tensor(ridge, device="cuda", dtype=torch.float32), (1, 2, 3))
    for k in cpu:
        np.testing.assert_allclose(got[k].cpu().numpy(), cpu[k], rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(os.environ.get("CRENET_TEST_CUDA") != "1", reason="explicit server GPU verification")
def test_cuda_dictionary_matches_cpu():
    prior, known, masks, noise, _ = small_fixture()
    blocks = [(s.start, s.stop, t, d) for s, t, d in prior.separable_blocks()]
    cpu, transforms, rows, whiteners = build_dictionary(prior.model, known, masks, noise, blocks, DenseBackend())
    gpu_backend = DenseBackend("cuda")
    gpu, gt, grows, gw = build_dictionary(prior.model, known, masks, noise, blocks, gpu_backend)
    np.testing.assert_allclose(gpu.cpu().numpy(), cpu, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(gt.cpu().numpy(), transforms, rtol=2e-6, atol=1e-7)
    for a, b in zip(rows, grows):
        np.testing.assert_array_equal(a, b)
