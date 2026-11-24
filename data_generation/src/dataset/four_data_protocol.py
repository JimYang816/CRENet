"""Versioned four-data-symbol dataset identities, independent of historical PDDP."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_generation.src.config import ROOT, load_config

SCHEMA = "watermark-dddd-delay192-compact-v1"
MODEL_SCHEMA = SCHEMA+"-model"
SPLITS = ("train", "validation", "test")
SPLIT_CODES = dict(train=2, validation=3, test=4)
TEST_RECORDS = list(range(11, 48, 4))
VALIDATION_RECORDS = list(range(13, 46, 8))
TRAIN_RECORDS = sorted(set(range(1, 51))-set(TEST_RECORDS)-set(VALIDATION_RECORDS))


def load_options(path=None):
    options = json.loads(Path(path or ROOT/"configs/data/four_data_dataset.json").read_text(encoding="utf-8"))
    if (options.get("schema") != SCHEMA or options.get("channels") != ["NOF1", "NCS1"]
            or options.get("compensation_calibration") not in ("per_split", "train_only", "none")
            or options.get("default_input_view") not in ("raw", "reference_assisted")
            or options.get("reference_dtype") != "complex64"
            or type(options.get("ber_diagnostic_frames_per_split")) is not int
            or not 1 <= options["ber_diagnostic_frames_per_split"] <= 64
            or (options.get("compensation_calibration") == "none" and options.get("default_input_view") != "raw")):
        raise ValueError("Invalid four-data dataset options")
    return options


def channel_config(channel, smoke=False):
    c = load_config(ROOT/f"configs/data/{channel.lower()}/dddd.json")
    if (c["channel"] != channel or c["frame_structure"] != "DDDD"
            or (c["nfft"], c["cp"], c["data_count"], c["pilot_count"], c["dpss_expected_count"]) != (1024, 256, 512, 256, 24)
            or c["dpss_half_bandwidth_hz"] != 12 or c["feedback_iterations"] != 0):
        raise ValueError("Formal four-data generation requires the current DDDD / CP256 / Q24 frame")
    c["delay_length"] = 192
    if smoke:
        c["smoke"] = True
        c["snr_db"] = [0, 15, 30]
        c["learning_dataset"] = dict(protocol="watermark-dddd-v1", records=dict(train=[1], validation=[2], test=[3]),
            starts_seconds=[2], repeats=dict(train=1, validation=1, test=2))
    validate_plan(c)
    return c


def windows(c, split):
    p = c["learning_dataset"]
    return [(record, float(start)) for record in p["records"][split] for start in p["starts_seconds"]]


def counts(c):
    return {split: dict(records=c["learning_dataset"]["records"][split], physical_windows=len(windows(c, split)),
        repeats=c["learning_dataset"]["repeats"][split],
        frames_per_snr=len(windows(c, split))*c["learning_dataset"]["repeats"][split],
        frames_all_snr=len(windows(c, split))*c["learning_dataset"]["repeats"][split]*len(c["snr_db"]),
        data_symbols_per_frame=4,
        bits_per_snr=len(windows(c, split))*c["learning_dataset"]["repeats"][split]*4*c["data_count"]*2)
        for split in SPLITS}


def validate_plan(c):
    p = c["learning_dataset"]
    groups = [c["prior_records"], c["validation_records"]]+[p["records"][split] for split in SPLITS]
    if (p.get("protocol") != "watermark-dddd-v1" or any(not group or len(group) != len(set(group)) for group in groups)
            or any(set(group)&set(other) for i, group in enumerate(groups) for other in groups[:i])):
        raise ValueError("Independent prior, CSI validation and train/validation/test records must be disjoint")
    starts = p["starts_seconds"]
    duration = (4*(c["nfft"]+c["cp"])+2*c["guard_samples"])/c["fs_baseband"]
    if (not starts or any(not np.isfinite(s) or s < c["guard_samples"]/c["fs_baseband"] for s in starts)
            or any(b-a < duration for a, b in zip(starts, starts[1:]))
            or any(type(p["repeats"][split]) is not int or p["repeats"][split] < 1 for split in SPLITS)):
        raise ValueError("Invalid or overlapping physical windows/repeats")
    if not c.get("smoke"):
        expected = dict(train=TRAIN_RECORDS, validation=VALIDATION_RECORDS, test=TEST_RECORDS)
        if (p["records"] != expected or starts != list(range(2, 31, 2))
                or p["repeats"] != dict(train=1, validation=1, test=4)
                or c["prior_records"] != list(range(51, 56)) or c["validation_records"] != list(range(56, 61))
                or c["snr_db"] != list(range(0, 31, 5))):
            raise ValueError("Formal generation must preserve the configured 35/5/10-record plan")
    return counts(c)


def estimated_max_bytes(c):
    """Conservative uncompressed bound; reference matrices are stored once per window."""
    n, q, k, ns = c["nfft"], c["dpss_expected_count"], c["delay_length"], len(c["snr_db"])
    result = 0
    for item in counts(c).values():
        nw, nf = item["physical_windows"], item["frames_per_snr"]
        result += nw*(4*n*n*8+4*q*k*16)+nf*(ns*q*k*8+2*4*n*16+4*c["data_count"]*2)
    return result+512*1024**2
