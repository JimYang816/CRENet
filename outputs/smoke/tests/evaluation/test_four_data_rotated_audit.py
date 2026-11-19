"""Independent checks of the CSI-norm and finite RF covariance audit."""
import numpy as np
from threadpoolctl import threadpool_limits

from data_generation.src.config import load_config
from data_generation.src.receiver.four_data import BlockedFourDataReceiver
from src.evaluation.four_data_rotated_audit import coefficient_nmse, exact_noise_audit


def test_coefficient_nmse_includes_complex_basis_and_omitted_taps():
    rng = np.random.default_rng(71)
    random = lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    basis, coefficients, truth = random((4, 7, 2)), random((3, 2, 3)), random((4, 7, 6))
    energy = np.sum(abs(truth)**2, axis=(1, 2))
    cross = np.einsum("dnq,dnl->dql", basis.conj(), truth[..., :3])
    reconstructed = np.zeros((3, 4, 7, 6), complex)
    reconstructed[..., :3] = np.einsum("dnq,bql->bdnl", basis, coefficients)
    exact = np.sum(abs(reconstructed-truth)**2, axis=(2, 3))/energy
    np.testing.assert_allclose(coefficient_nmse(basis, coefficients, cross, energy), exact, rtol=2e-14)


def test_finite_rf_thermal_covariance_matches_analytical_chain():
    config = load_config()
    config.update(nfft=56, cp=32, fft_advance=2, guard_samples=12)
    with threadpool_limits(2):
        receiver = BlockedFourDataReceiver(config)
        result = exact_noise_audit(receiver, config, receiver.chain.noise_covariance)
    assert result["covariance_relative_frobenius"] < 2e-14
    assert result["cross_D_noise_covariance_norm"] == 0
