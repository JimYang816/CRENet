"""Verify the fixed data budget and two-sided guards of every candidate."""

import numpy as np
import pytest

from data_generation.src.dataset.four_data_protocol import channel_config
from data_generation.src.transmitter.ofdm import carrier_masks, make_frame
from src.evaluation.four_data_protected_pilots import cluster_config, load_options


@pytest.mark.parametrize("channel", ["NOF1", "NCS1"])
def test_protected_clusters_fill_the_fft_and_preserve_payload(channel):
    base = channel_config(channel)
    previous = make_frame(base, 13, 2, 3, payload_repeat=0)
    for spec in load_options()["candidates"]:
        c = cluster_config(base, spec)
        masks = carrier_masks(c)
        pilots = np.flatnonzero(masks["pilot"])
        clusters = np.split(pilots, np.flatnonzero(np.diff(pilots) != 1)+1)
        assert len(clusters) == spec["clusters"]
        assert masks["data"].sum() == 512
        assert masks["edge"].sum() == 2*spec["null_per_edge"]
        guard = spec["guard_nulls_per_side"]
        assert masks["protection"].sum() == 2*guard*len(clusters)
        np.testing.assert_array_equal(masks["tracking"], masks["pilot"] | masks["protection"])
        for cluster in clusters:
            assert len(cluster) == spec["pilots_per_cluster"]
            assert masks["protection"][cluster[0]-guard:cluster[0]].all()
            assert masks["protection"][cluster[-1]+1:cluster[-1]+1+guard].all()
        frame = make_frame(c, 13, 2, 3, payload_repeat=0)
        np.testing.assert_array_equal(frame.bits, previous.bits)
        assert not frame.x[:, masks["null"]].any()


def test_cluster_config_rejects_a_layout_without_two_sided_guards():
    spec = dict(clusters=64, pilots_per_cluster=4, guard_nulls_per_side=0, null_per_edge=64)
    with pytest.raises(ValueError, match="valid integers"):
        cluster_config(channel_config("NOF1"), spec)
