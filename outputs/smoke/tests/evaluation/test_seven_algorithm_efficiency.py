"""Checks for provenance isolation and the reported latency statistic."""
import numpy as np
import pytest

from src.evaluation.seven_algorithm_efficiency import export_source, sample_keys, summarize_times


def test_samples_cover_snr_frame_extent_and_all_four_symbol_positions():
    keys = sample_keys(7, 600, 4)
    assert len(keys) == len(set(keys)) == 28
    for si in range(7):
        selected = [(fi, di) for s, fi, di in keys if s == si]
        assert {di for _, di in selected} == {0, 1, 2, 3}
        assert min(fi for fi, _ in selected) == 0
        assert max(fi for fi, _ in selected) == 599


def test_reported_latency_is_median_of_complete_round_means():
    result = summarize_times([1., 100., 2., 2., 3., 3.], sample_count=2)
    assert result["measured_blocks"] == 6
    assert result["median_round_ms"] == 3.
    np.testing.assert_allclose(result["round_mean_ms"], [50.5, 2., 3.])


def test_historical_export_rejects_a_modified_algorithm(tmp_path):
    source = export_source(tmp_path, "fe873ac5bb3a57d0e6d8e6ed042e97de655e19f4")
    implementation = source / "src/algorithms/MMSE/full_matrix.py"
    implementation.write_text("raise RuntimeError('modified source')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Historical source changed"):
        export_source(tmp_path, "fe873ac5bb3a57d0e6d8e6ed042e97de655e19f4")
