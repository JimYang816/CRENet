"""Fixed original 4D carrier contract; no pilot relocation or power rescaling."""
from __future__ import annotations

import numpy as np

from data_generation.src.receiver.four_data import make_four_data_frame
from data_generation.src.transmitter.ofdm import carrier_masks


def original_protocol(config):
    expected = dict(nfft=1024, null_per_edge=64, data_count=512, pilot_count=128)
    if any(config[name] != value for name,value in expected.items()):
        raise ValueError("The original carrier contract requires FFT1024, 64 empty bins per edge, 512 data and 128 pilots")
    one = carrier_masks(config)
    literal = np.pad(np.tile([1,1,0,2,0,1,1],128),(64,64),constant_values=-1)
    required = dict(data=literal==1,pilot=literal==2,protection=literal==0,
                    tracking=(literal==0)|(literal==2),edge=literal==-1)
    if any(not np.array_equal(one[name],value) for name,value in required.items()):
        raise ValueError("Original carrier positions changed from the fixed DDZPZDD pattern")
    masks = {name:np.broadcast_to(value, (4, 1024)).copy() for name,value in one.items()}
    known = make_four_data_frame(config, 0, 0).known
    return known, masks


def assert_original_protocol(config, known, masks):
    expected_known, expected_masks = original_protocol(config)
    for name in ("data", "pilot", "protection", "tracking"):
        if name not in masks or not np.array_equal(np.asarray(masks[name]), expected_masks[name]):
            raise ValueError(f"Original carrier mask changed: {name}")
    if not np.array_equal(known, expected_known):
        raise ValueError("Original public pilot values or pilot power changed")


def make_original_frame(config, record, start, known, masks):
    assert_original_protocol(config, known, masks)
    frame = make_four_data_frame(config, record, start)
    np.testing.assert_array_equal(frame.known, known)
    if frame.bits.shape != (4, 512, 2):
        raise ValueError("The original payload must retain all 512 QPSK symbols per D")
    return frame
