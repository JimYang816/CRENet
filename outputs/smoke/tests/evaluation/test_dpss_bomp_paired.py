import numpy as np

from src.evaluation.dpss_bomp_paired import paired


def test_paired_record_bootstrap_preserves_constant_method_difference():
    records = np.array([1, 1, 2, 2, 3, 3])
    errors = np.zeros((2, 2, 6, 2), dtype=int)
    errors[1] = 1
    nmse = np.ones_like(errors, dtype=float)
    nmse[1] = 2
    out = paired(records, errors, nmse, ndata=2, draws=100, seed=9)
    np.testing.assert_allclose(out["ber_difference_omp_minus_lmmse"], [.25, .25])
    np.testing.assert_allclose(out["ber_difference_95pct_record_bootstrap"], [[.25, .25], [.25, .25]])
    np.testing.assert_allclose(out["nmse_db_difference_omp_minus_lmmse"], 10*np.log10([2, 2]))
    np.testing.assert_allclose(out["nmse_db_difference_95pct_record_bootstrap"],
                               np.tile(10*np.log10(2), (2, 2)))
