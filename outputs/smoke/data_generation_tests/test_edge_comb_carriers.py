"""The three-carrier DDDD comb preserves frame labels and moves all nulls to the edges."""

import numpy as np
import pytest

from data_generation.src.dataset.four_data_protocol import channel_config
from data_generation.src.transmitter.ofdm import carrier_masks, make_frame


@pytest.mark.parametrize("channel", ["NOF1", "NCS1"])
def test_edge_comb_layout_and_paired_payload(channel):
    base = channel_config(channel, legacy_comb=True)
    previous = carrier_masks(base)
    cfg = dict(base, null_per_edge=128)
    masks = carrier_masks(cfg)
    fft_bins = np.arange(1024)
    np.testing.assert_array_equal(np.flatnonzero(masks["edge"]), fft_bins[np.r_[0:128, 896:1024]])
    np.testing.assert_array_equal(np.flatnonzero(masks["pilot"]), np.arange(128, 896, 3))
    assert masks["data"].sum() == 512 and not masks["protection"].any()
    np.testing.assert_array_equal(masks["tracking"], masks["pilot"])
    assert previous["protection"].sum() == 128 and previous["edge"].sum() == 128

    old = make_frame(base, 11, 2, 4, payload_repeat=1)
    new = make_frame(cfg, 11, 2, 4, payload_repeat=1)
    assert new.seed == old.seed and not new.x[:, masks["edge"]].any()
    np.testing.assert_array_equal(new.bits, old.bits)
    np.testing.assert_array_equal(new.known[:, masks["pilot"]], old.known[:, previous["pilot"]])
    np.testing.assert_array_equal(new.x[:, masks["data"]], old.x[:, previous["data"]])
    assert not np.array_equal(new.baseband, old.baseband)
