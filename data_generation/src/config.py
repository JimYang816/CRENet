"""Fixed experiment parameters and stable provenance identifiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    c = json.loads(Path(path or ROOT / "configs/data/ncs1/dddd.json").read_text(encoding="utf-8"))
    frame = c.get("frame_structure")
    if frame not in ("DDDD", "PDDP"):
        raise ValueError("Unknown frame configuration; expected DDDD or PDDP")
    expected_cp = 256 if frame == "DDDD" else 384
    if (c["nfft"], c["cp"], c["fs_baseband"], c["fs_passband"], c["fft_advance"]) != (1024, expected_cp, 6000, 96000, 32):
        raise ValueError(f"{frame} requires FFT1024/CP{expected_cp}, 6/96-kHz and advance-32")
    sets = [set(c[k]) for k in ("prior_records", "validation_records", "test_records")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Prior, validation and test records must be disjoint")
    return c


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
