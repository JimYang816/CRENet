"""Raw-view receiver inputs are independent of audit and supervision fields."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from outputs.smoke.data_generation_tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from data_generation.src.dataset.compact_math import reconstruct
from src.algorithms.single_tap_mmse.equalizer import equalize_batch
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.inputs import prepare_batch


@pytest.fixture
def raw_data(tmp_path):
    with threadpool_limits(2):
        c, prior, chain, archive = fixture()
        c["snr_db"] = [0, 10]
        generation = tmp_path/"generation"
        generation.mkdir()
        generate(tmp_path/"data", generation, c, archive, chain, prior, DenseBackend(), {"commit": "raw-test"})
        yield tmp_path/"data"


def test_raw_reader_and_batches_do_not_access_forbidden_fields(raw_data, monkeypatch):
    original = h5py.Group.__getitem__
    forbidden = ("reference", "compensation", "actual_snr", "bits", "classes", "labels", "model_residual",
                 "effective_rf_variance", "training_calibration/")
    def guard(group, name):
        if any(key in str(name) for key in forbidden):
            raise AssertionError("Forbidden input read: "+str(name))
        return original(group, name)
    monkeypatch.setattr(h5py.Group, "__getitem__", guard)
    with CompactDataset(raw_data/"test.h5", input_view="raw") as ds:
        result = ds.model_input(0, 0)
        assert set(result["noise"]) == {"rf_variance", "unit_rx_covariance"}
        assert "uncertainty" not in result
        batch = prepare_batch(ds, 0, [0], 0, DenseBackend())
        np.testing.assert_allclose(batch.channel[0], result["csi"][0], atol=2e-15)


def test_changing_test_truth_and_diagnostics_preserves_inputs_and_predictions(raw_data):
    def capture():
        with CompactDataset(raw_data/"test.h5", input_view="raw") as ds:
            inp = ds.raw_model_input(1, 0)
            batch = prepare_batch(ds, 1, [0], 0, DenseBackend())
            pred = equalize_batch(batch, DenseBackend()).symbols
            return inp, batch, pred
    before, batch_before, prediction_before = capture()
    with h5py.File(raw_data/"test.h5", "r+") as h:
        h["windows/reference_csi"][:] = 123+87j
        h["windows/reference_projection"][:] = -321-75j
        h["samples/bits"][:] ^= np.uint8(1)
        h["samples/actual_snr_db"][:] = -99
    after, batch_after, prediction_after = capture()
    for name in ("csi", "observations", "known", "data_indices", "carrier_positions"):
        np.testing.assert_array_equal(before[name], after[name])
    np.testing.assert_array_equal(batch_before.channel, batch_after.channel)
    np.testing.assert_array_equal(prediction_before, prediction_after)


def test_raw_is_unmixed_reconstruction_and_other_views_are_rejected(raw_data):
    with CompactDataset(raw_data/"test.h5", input_view="raw") as raw:
        expected = reconstruct(raw._model, raw._h["samples/raw_coefficients"][0, 0].astype(complex))
        np.testing.assert_array_equal(raw.model_input(0, 0)["csi"], expected)
        assert raw.model_input(0, 0)["input_view"] == "raw"
    with pytest.raises(ValueError, match="unmodified"):
        CompactDataset(raw_data/"test.h5", input_view="compensated")
