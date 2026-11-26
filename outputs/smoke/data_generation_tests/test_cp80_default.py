"""Verify current CP80 defaults and reject stale CP256 training/prior artifacts."""
import json

import h5py
import numpy as np
import pytest

from data_generation.src.config import ROOT, digest, frozen_support_matches_config, load_config
from data_generation.src.dataset.four_data_export import load_prior
from data_generation.src.dataset.four_data_protocol import channel_config, counts, load_options
from scripts.run.four_data_experiment import preflight


@pytest.mark.parametrize("channel", ["NOF1", "NCS1"])
def test_cp80_is_the_formal_default_with_independent_existing_plan(channel):
    base = load_config(ROOT/f"configs/data/{channel.lower()}/dddd.json")
    c = channel_config(channel)
    assert base["cp"] == c["cp"] == 480
    assert c["cp"]/c["fs_baseband"] == .08
    assert 4*(c["nfft"]+c["cp"]) == 6016
    assert c["delay_length"] == 192 and c["dpss_expected_count"] == 24
    assert [counts(c)[s]["frames_all_snr"] for s in ("train", "validation", "test")] == [3675, 525, 4200]
    support = json.loads((ROOT/f"configs/data/{channel.lower()}/delay_support192.json").read_text(encoding="utf-8"))
    assert frozen_support_matches_config(support, base)
    changed = dict(base, fft_advance=base["fft_advance"]+1)
    assert not frozen_support_matches_config(support, changed)
    assert load_options()["source_study"] == "outputs/smoke/four_data_cp80_20261125"
    assert load_options()["default_input_view"] == "reference_assisted"


def test_training_rejects_superseded_cp256_acceptance_before_loading_data(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps(dict(complete=True, smoke=False,
        channel="NOF1", cp=256, cp_ms=256/6, files={})), encoding="utf-8")
    with pytest.raises(ValueError, match="CP80"):
        preflight(tmp_path, "NOF1")


def test_formal_prior_rejects_a_cp256_time_basis(tmp_path):
    c = channel_config("NOF1")
    directory = tmp_path/"nof1"
    directory.mkdir()
    support = json.loads((ROOT/"configs/data/nof1/delay_support192.json").read_text(encoding="utf-8"))
    (directory/"frozen_support192.json").write_text(json.dumps(support), encoding="utf-8")
    meta = dict(config=c)
    (directory/"manifest.json").write_text(json.dumps(meta), encoding="utf-8")
    with h5py.File(directory/"physical.h5", "w") as h:
        h.attrs.update(complete=True, identity=digest(meta))
        h.create_dataset("basis", shape=(5120, 24), dtype=np.float64)
    with pytest.raises(ValueError, match="CP80 physical frame"):
        load_prior(c, tmp_path, None, None)
