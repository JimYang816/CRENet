"""Verify reduced-order estimation and paired comparison safeguards."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.signal.windows import dpss
from threadpoolctl import threadpool_limits

from outputs.smoke.tests.evaluation.test_four_data_q24_compare import synthetic_source
from src.common.numerics import DenseBackend
from src.evaluation.four_data_lmmse import channel_config, evaluate, model_basis
from src.evaluation.four_data_q16_compare import verify_order, verify_q24_source
from src.evaluation.four_data_q24_compare import verify_reference


def test_q16_is_the_first_16_vectors_and_records_the_actual_tail():
    c = channel_config("NOF1")
    basis, concentration = model_basis(c)
    nw = len(basis)*c["dpss_half_bandwidth_hz"]/c["fs_baseband"]
    reduced = dpss(len(basis), nw, Kmax=16, sym=True, norm=2).T
    np.testing.assert_allclose(basis[:, :16], reduced, rtol=0, atol=2e-12)
    tail = 1-np.sum(concentration[:16])/(2*nw)
    assert tail == pytest.approx(.289772792300458, rel=1e-10)
    assert tail > c["dpss_tail_ratio"]


def test_three_orders_reestimate_the_same_frame_and_preserve_reference(tmp_path):
    source = tmp_path/"source"
    metadata = synthetic_source(source)
    with threadpool_limits(2):
        rows = {q: evaluate("NOF1", tmp_path/f"q{q}", metadata, DenseBackend(), physical_root=source, order=q)
                for q in (1, 2, 3)}
    assert [rows[q]["q"] for q in (1, 2, 3)] == [1, 2, 3]
    assert all(rows[q]["bits_per_snr"] == 64 for q in rows)
    verify_reference(rows[1], rows[3])
    verify_reference(rows[2], rows[3])
    assert not np.allclose(rows[1]["mean_raw_nmse_linear"], rows[3]["mean_raw_nmse_linear"])


def test_reproduction_rejects_changed_q24_results():
    frame = dict(record=11, start_seconds=2., raw_single_errors=[[1, 2, 3, 4]],
                 raw_full_errors=[[4, 3, 2, 1]], reference_single_errors=[[1, 1, 1, 1]],
                 reference_full_errors=[[0, 0, 0, 0]], raw_nmse_linear=[[.1, .2, .3, .4]])
    old = dict(channel="NOF1", q=24, snr_db=[0], bits_per_snr=64, frames=[frame])
    assert verify_order(old, deepcopy(old), 24)["bit_errors_identical"]
    altered = deepcopy(old)
    altered["frames"][0]["raw_full_errors"][0][0] += 1
    with pytest.raises(ValueError, match="bit errors"):
        verify_order(old, altered, 24)
    altered = deepcopy(old)
    altered["frames"][0]["raw_nmse_linear"][0][0] += .1
    with pytest.raises(ValueError, match="NMSE"):
        verify_order(old, altered, 24)


def test_q24_source_rejects_changed_input_or_physical_identity(tmp_path):
    metadata = dict(options={}, configs={}, physical={}, baseline_metrics_sha256="a",
                    baseline_manifest_sha256="b", code_sha256={})
    prior = dict(schema="four-data-paired-q24-q32-v1", input_view="raw", csi_compensation=False, **metadata)
    channels = {ch: {q: dict(complete=True) for q in ("q24", "q32")} for ch in ("NOF1", "NCS1")}
    report = dict(complete=True, metadata=prior, channels=channels)
    baseline = tmp_path/"previous"
    baseline.mkdir()
    (baseline/"manifest.json").write_text(json.dumps(prior), encoding="utf-8")
    (baseline/"metrics.json").write_text(json.dumps(report), encoding="utf-8")
    verify_q24_source(tmp_path, baseline, metadata)
    different = dict(metadata, physical={"NOF1": {"sha256": "changed"}})
    with pytest.raises(ValueError, match="physical data"):
        verify_q24_source(tmp_path, baseline, different)
