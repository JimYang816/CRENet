"""Independent checks for design-matrix atoms and representation error accounting."""
import numpy as np

from data_generation.src.config import load_config
from data_generation.src.receiver.estimation import input_windows
from data_generation.src.transmitter.ofdm import fft
from src.evaluation.four_data_design import measurement_matrix, pilot_layout, projection_error


def test_public_pilot_matrix_matches_direct_channel_action():
    rng = np.random.default_rng(33)
    config = load_config()
    config.update(nfft=14, null_per_edge=0)
    basis = rng.normal(size=(4, 14, 3))+1j*rng.normal(size=(4, 14, 3))
    delay_basis, _ = np.linalg.qr(rng.normal(size=(5, 2))+1j*rng.normal(size=(5, 2)))
    latent = rng.normal(size=(3, 2))+1j*rng.normal(size=(3, 2))
    coefficients = latent @ delay_basis.conj().T
    taps = basis @ coefficients
    for layout in ("current", "more_pilots_same_data", "more_pilots_more_observations"):
        known, rows, counts = pilot_layout(config, layout)
        assert all(sum(c[key] for key in ("pilots", "protection", "data")) == 14 for c in counts)
        matrix = measurement_matrix(basis, delay_basis, known, rows)
        y = fft(np.sum(taps*input_windows(known, 5), axis=-1))
        selected = np.concatenate([y[s, r] for s, r in enumerate(rows)])
        np.testing.assert_allclose(matrix @ latent.ravel(), selected, rtol=2e-13, atol=2e-13)
    cross = np.einsum("dnq,dnl->dql", basis.conj(), taps)
    energy = np.sum(abs(taps)**2, axis=(1, 2))
    np.testing.assert_allclose(projection_error(basis, coefficients, cross, energy, 5, delay_basis),
                                0, rtol=0, atol=2e-15)
    fitted = basis @ coefficients[:, :2]
    expected = (np.sum(abs(fitted-taps[..., :2])**2, axis=(1, 2))
                +np.sum(abs(taps[..., 2:])**2, axis=(1, 2)))/energy
    np.testing.assert_allclose(projection_error(basis, coefficients, cross, energy, 2), expected,
                                rtol=2e-13, atol=2e-13)
