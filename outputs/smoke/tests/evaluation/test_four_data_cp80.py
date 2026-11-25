"""Check CP duration, actual waveform framing and independent frozen support."""
import copy
import json

import numpy as np
import pytest

from data_generation.src.config import ROOT, load_config
from data_generation.src.transmitter.ofdm import ifft, window_indices
from src.evaluation.four_data_cp80 import channel_settings, load_options
from src.evaluation.four_data_lmmse import model_basis
from src.evaluation.four_data_q_sweep import make_frame


ARCHIVE_SHA256 = "3cf9e3a9b116b525f9b52c098018383665ca24f3a9ab05bc53b7716b1b3c5cc1"


@pytest.mark.parametrize("channel", ["NOF1", "NCS1"])
def test_cp80_changes_physical_frame_and_preserves_accepted_delays(channel):
    options = load_options()
    c, study, priors, tests = channel_settings(channel, options, ARCHIVE_SHA256)
    assert c["cp"] == 480 and c["cp"]/c["fs_baseband"] == .08
    assert c["delay_length"] == 192
    assert c["dpss_expected_count"] == 24
    assert load_config(ROOT/f"configs/data/{channel.lower()}/dddd.json")["cp"] == 480
    assert len(priors) == 25 and len(tests) == 5
    assert not {r for r, _ in priors} & {r for r, _ in tests}
    original = json.loads((ROOT/options["support_files"][channel]).read_text(encoding="utf-8"))
    assert study["frozen_support"] == original
    if channel == "NOF1":
        assert max(original["indices"]) == 501
        assert len(set(np.diff(original["indices"]))) > 1
    frame = make_frame(c, *tests[0], 2)
    assert len(frame.baseband) == 4*(1024+480)+2*c["guard_samples"]
    blocks = frame.baseband[c["guard_samples"]:-c["guard_samples"]].reshape(4, 1504)
    np.testing.assert_allclose(blocks[:, 480:], ifft(frame.x), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(blocks[:, :480], blocks[:, -480:])
    for symbol in range(4):
        assert window_indices(c, symbol)[0] == c["guard_samples"]+symbol*1504+448


def test_cp80_channels_share_regenerated_full_frame_basis():
    bases = []
    for channel in ("NOF1", "NCS1"):
        c, _, priors, tests = channel_settings(channel, load_options(), ARCHIVE_SHA256, smoke=True)
        assert len(priors) == 2 and len(tests) == 1 and c["snr_db"] == [0, 15, 30]
        basis, _ = model_basis(c)
        assert basis.shape == (6016, 24)
        bases.append(basis)
    np.testing.assert_array_equal(*bases)
    np.testing.assert_allclose(bases[0].T@bases[0], np.eye(24), rtol=0, atol=1e-12)


def test_cp80_rejects_wrong_archive_and_record_leakage(tmp_path):
    options = load_options()
    with pytest.raises(ValueError, match="accepted independent-prior source"):
        channel_settings("NOF1", options, "wrong-archive")
    changed = copy.deepcopy(options)
    changed["test_records"] = [51]
    path = tmp_path/"overlap.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        load_options(path)
    changed["test_records"] = [1]
    with pytest.raises(ValueError, match="existing independent prior/test split"):
        channel_settings("NOF1", changed, ARCHIVE_SHA256)
