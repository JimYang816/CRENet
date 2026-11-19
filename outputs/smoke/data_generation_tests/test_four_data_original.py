"""User's original carrier positions, powers and complete payload are fixed."""
import sys

import numpy as np
import pytest

from data_generation.src.config import load_config
from data_generation.src.receiver.four_data import make_four_data_frame
from data_generation.src.receiver.four_data_original import (
    assert_original_protocol, make_original_frame, original_protocol,
)
from src.evaluation.four_data_original import restored_parameters


def test_original_masks_match_literal_seven_bin_pattern_and_edges():
    config = load_config()
    known,masks = original_protocol(config)
    layout = np.pad(np.tile([1,1,0,2,0,1,1],128),(64,64),constant_values=-1)
    for name,label,count in (("data",1,512),("pilot",2,128),("protection",0,256)):
        expected = np.broadcast_to(layout==label,(4,1024))
        np.testing.assert_array_equal(masks[name],expected)
        np.testing.assert_array_equal(masks[name].sum(axis=-1),[count]*4)
    np.testing.assert_array_equal(masks["tracking"],np.broadcast_to((layout==0)|(layout==2),(4,1024)))
    np.testing.assert_array_equal(known[~masks["pilot"]],0)
    np.testing.assert_allclose(abs(known[masks["pilot"]])**2,1,rtol=2e-15)
    assert not np.any(known[:,:64]) and not np.any(known[:,-64:])


def test_original_frame_retains_all_bits_and_exact_unscaled_waveform():
    config = load_config()
    known,masks = original_protocol(config)
    for record,start in ((11,2),(15,14)):
        original = make_four_data_frame(config,record,start)
        restored = make_original_frame(config,record,start,known,masks)
        for name in ("x","known","bits","baseband"):
            np.testing.assert_array_equal(getattr(restored,name),getattr(original,name))
        assert restored.bits.size==4096 and restored.seed==original.seed


def test_original_contract_rejects_relocation_or_pilot_power_change():
    config = load_config()
    known,masks = original_protocol(config)
    shifted = {k:v.copy() for k,v in masks.items()}
    shifted["data"][1] = np.roll(shifted["data"][1],1)
    with pytest.raises(ValueError,match="mask changed"):
        assert_original_protocol(config,known,shifted)
    with pytest.raises(ValueError,match="pilot.*power"):
        assert_original_protocol(config,known*np.sqrt(.809016852263548),masks)
    changed = dict(config,data_count=256,null_per_edge=0)
    with pytest.raises(ValueError,match="original carrier contract"):
        original_protocol(changed)


def test_restored_unknown_count_is_reported_as_underdetermined():
    for channel,unknowns in (("NOF1",2560),("NCS1",2688)):
        p = restored_parameters(channel)
        assert p["observations"]==1536 and p["unknown_coefficients"]==unknowns
        assert p["pilot_system_underdetermined"]
        assert p["pilots_per_d"]+p["protection_per_d"]+p["data_per_d"]+p["empty_edges_per_d"]==1024


def test_archived_full_band_entry_cannot_start_another_waveform_run(monkeypatch):
    from scripts.evaluate.four_data_rotated import main
    monkeypatch.setattr(sys,"argv",["four_data_rotated"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code==2
