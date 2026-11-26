"""The holdout must freeze selection and compare matching test symbols."""
import copy

import numpy as np
import pytest

from src.evaluation.four_data_crenet import ALGORITHMS
from src.evaluation.four_data_detector_holdout import paired_comparison, validate_selection


def metrics(ber):
    return dict(ber=ber, corrected_nmse=.5, initial_nmse=1.,
                per_snr=[dict(initial_nmse=1., corrected_nmse=.5)])


def selected_candidate():
    validation = metrics(.09)
    change = dict(scalars=dict(changed_tensors=1), neural=dict(changed_tensors=1))
    diagnostic = dict(complete=True, test_used=False, source_checkpoint_stage="B",
        validation_error_count_difference=0, original_identity="identity", variants={
            "scaled":dict(best_interval=1, best_validation=validation, beats_baseline=True,
                          intervals=[dict(changes=change)])})
    candidate = dict(variant="scaled", interval=1, source_identity="identity", validation=validation)
    original = dict(identity="identity", validation=metrics(.1))
    return diagnostic, candidate, original


def test_selection_requires_both_trained_groups_and_validation_improvement():
    diagnostic, candidate, original = selected_candidate()
    assert validate_selection(diagnostic, candidate, original, "scaled")["best_interval"] == 1
    diagnostic["variants"]["scaled"]["intervals"][0]["changes"]["neural"]["changed_tensors"] = 0
    with pytest.raises(ValueError, match="physical and neural"):
        validate_selection(diagnostic, candidate, original, "scaled")
    diagnostic, candidate, original = selected_candidate()
    original["validation"] = metrics(.08)
    with pytest.raises(ValueError, match="beat the original"):
        validate_selection(diagnostic, candidate, original, "scaled")
    diagnostic, candidate, original = selected_candidate()
    diagnostic["test_used"] = True
    with pytest.raises(ValueError, match="validation-only"):
        validate_selection(diagnostic, candidate, original, "scaled")


def test_paired_comparison_preserves_record_and_symbol_alignment():
    report = dict(channel="NOF1", input_view="reference_assisted", snr_db=[0, 5],
        frames=[dict(record=1), dict(record=2)], bits_per_snr=16,
        curves=dict(crenet=dict(ber=[.1, .05]), full_matrix_lmmse=dict(ber=[.2, .1])),
        corrected_nmse_db=[-10., -12.], initial_nmse_db=[-9., -11.])
    candidate = copy.deepcopy(report)
    old = {name:np.ones((2, 2, 4), dtype=np.int64) for name in ALGORITHMS}
    new = {name:value.copy() for name, value in old.items()}
    new["crenet"][0, 0, 0] = 0
    new["crenet"][1, 1, 0] = 2
    result = paired_comparison(report, candidate, old, new)
    assert result["error_change_by_snr"] == [-1, 1]
    assert result["error_change_by_record"] == {"1":-1, "2":1}
    assert result["blocks_improved"] == result["blocks_worsened"] == 1
    new["full_matrix_lmmse"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="baseline error counts changed"):
        paired_comparison(report, candidate, old, new)
