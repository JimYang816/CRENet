"""Checks for the independent Q33..40 raw-CSI continuation."""
from copy import deepcopy

import numpy as np
import pytest

from src.evaluation.four_data_q_sweep import basis, config
from src.evaluation.four_data_raw_q_extension import ORDERS, extended_basis, select, verify_baseline


def test_new_dpss_columns_preserve_original_and_are_orthonormal():
    c = config("NCS1")
    original, _ = basis(c)
    extended, concentration, diag = extended_basis(c, original)
    np.testing.assert_array_equal(extended[:, :32], original)
    np.testing.assert_allclose(extended.T@extended, np.eye(40), atol=5e-10, rtol=0)
    assert diag["first_32_max_absolute_discrepancy"] < 1e-9
    assert concentration.shape == (40,)
    assert concentration[32] < 1e-7
    wrong = original.copy()
    wrong[:, 0] = -wrong[:, 0]
    with pytest.raises(ValueError, match="do not reproduce"):
        extended_basis(c, wrong)


def test_select_only_from_complete_raw_validation_and_tie_lower_q():
    rows = [dict(q=q, input_view="raw", compensation=False, split="validation",
                 mean_ber=abs(q-35)*.01+.1) for q in ORDERS]
    selected = select(rows)
    assert selected["selected_q"] == 35
    assert selected["runner_up_q"] == 34
    assert selected["best_extension_q"] == 35
    with pytest.raises(ValueError, match="incomplete"):
        select(rows[:-1])
    changed = deepcopy(rows)
    changed[1]["compensation"] = True
    with pytest.raises(ValueError, match="raw validation"):
        select(changed)


def test_q32_reproduction_checks_per_frame_errors_and_reference_nmse():
    row = dict(q=32, input_view="raw", split="validation", frames=[
        dict(record=13, start_seconds=2, errors=[[1, 2], [3, 4]], nmse_linear=[[.1, .2], [.3, .4]])])
    assert verify_baseline(row, deepcopy(row))["bit_errors_identical"]
    bad = deepcopy(row)
    bad["frames"][0]["errors"][0][0] += 1
    with pytest.raises(AssertionError):
        verify_baseline(row, bad)
