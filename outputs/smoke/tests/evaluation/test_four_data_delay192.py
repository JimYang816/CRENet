"""Physical-index selection and full-support scoring for the dual-channel study."""
import json

import h5py
import numpy as np
import pytest

from src.evaluation.four_data_delay192 import (
    channel_study, freeze_support, load_study_config, select_prior_delays,
)
from src.evaluation.ncs1_delay_support import full_nmse


def test_current_channels_use_192_physical_taps():
    options = load_study_config()
    assert channel_study("NOF1", options)["frozen_support"] is None
    ncs1 = channel_study("NCS1", options)
    assert ncs1["frozen_support"]["indices"] == list(range(32, 224))


def test_selection_preserves_noncontiguous_indices_and_stable_ties():
    powers = np.ones((2, 1024))
    strongest = np.arange(100, 484, 2)
    powers[:, strongest] = 10
    np.testing.assert_array_equal(select_prior_delays(powers), strongest)
    np.testing.assert_array_equal(select_prior_delays(np.ones((2, 1024))), np.arange(192))


@pytest.mark.parametrize("powers", [np.zeros((2, 1024)), np.full((2, 1024), np.nan), -np.ones((2, 1024))])
def test_reject_invalid_prior_powers(powers):
    with pytest.raises(ValueError):
        select_prior_delays(powers)


def test_support_freezes_without_reading_test_csi(tmp_path):
    path = tmp_path/"physical.h5"
    powers = np.ones((2, 1024))
    powers[:, 40:232] = 10
    with h5py.File(path, "w") as h:
        h["prior_power"] = powers
        # No test group is present or needed to select physical delays.
    meta = dict(study=dict(channel="NOF1", frozen_support=None, tap_count=192),
                frame_config_sha256="frame", git_head="code", archive_sha256="archive",
                prior_windows=[[51, 2], [52, 2]], smoke=True)
    supports, _ = freeze_support(path, tmp_path, meta)
    np.testing.assert_array_equal(supports["selected192"], np.arange(40, 232))
    frozen = json.loads((tmp_path/"frozen_support192.json").read_text(encoding="utf-8"))
    assert frozen["prior_windows"] == [[51, 2], [52, 2]]
    assert frozen["source_archive_sha256"] == "archive"
    assert meta["study"]["frozen_support"] is None
    meta["study"]["frozen_support"] = frozen
    meta["smoke"] = False
    freeze_support(path, tmp_path, meta)
    frozen["indices"][0] = 0
    with pytest.raises(ValueError, match="does not reproduce"):
        freeze_support(path, tmp_path, meta)


def test_nmse_scores_omitted_physical_taps():
    truth = np.zeros((4, 8, 10), dtype=complex)
    truth[..., 2] = 1
    truth[..., 9] = 2
    selected = truth[None, ..., [2]].copy()
    np.testing.assert_allclose(full_nmse(selected, truth, np.array([2])), 4/5)
