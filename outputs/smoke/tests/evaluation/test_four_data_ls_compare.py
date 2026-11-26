"""Check the underdetermined pilot solve and independent symbol layout."""
from types import SimpleNamespace

import numpy as np

from src.evaluation.four_data_ls_compare import least_squares, minimum_norm_inverses


def test_minimum_norm_ls_matches_each_symbols_direct_solution():
    rng = np.random.default_rng(27)
    responses = rng.normal(size=(4, 3, 2, 4))+1j*rng.normal(size=(4, 3, 2, 4))
    matrices, inverse, ranks = minimum_norm_inverses(responses)
    design = SimpleNamespace(basis=np.zeros((6, 2)), delays=np.arange(4), rows=[1, 3, 5])
    y = rng.normal(size=(2, 3, 4, 6))+1j*rng.normal(size=(2, 3, 4, 6))
    actual, residual = least_squares(design, y, matrices, inverse)
    assert ranks == [3]*4 and residual < 1e-24
    for s, a in enumerate(matrices):
        inverse_direct = a.conj().T @ np.linalg.solve(a @ a.conj().T, np.eye(3))
        expected = y[:, :, s, design.rows] @ inverse_direct.T
        np.testing.assert_allclose(actual[:, :, s].reshape(2, 3, 8), expected, atol=1e-13, rtol=1e-12)
        null_move = np.eye(8)-inverse_direct @ a
        np.testing.assert_allclose(actual[:, :, s].reshape(2, 3, 8) @ null_move.T, 0, atol=1e-13)
