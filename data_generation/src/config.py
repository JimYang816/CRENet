"""Fixed experiment parameters and stable provenance identifiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN_SUPPORT_DIR = ROOT / "configs/backups/dddd_cp256_20261125"
FROZEN_SUPPORT_CONFIGS = tuple(f"configs/backups/dddd_cp256_20261125/{ch}.json" for ch in ("nof1", "ncs1"))
DDDD_CARRIER_COUNTS = dict(data_count=512, pilot_count=224, protection_count=224, null_per_edge=32)


def load_config(path=None):
    c = json.loads(Path(path or ROOT / "configs/data/ncs1/dddd.json").read_text(encoding="utf-8"))
    return _validate_config(c)


def _validate_config(c):
    frame = c.get("frame_structure")
    if frame not in ("DDDD", "PDDP"):
        raise ValueError("Unknown frame configuration; expected DDDD or PDDP")
    expected_cp = 480 if frame == "DDDD" else 384
    if (c["nfft"], c["cp"], c["fs_baseband"], c["fs_passband"], c["fft_advance"]) != (1024, expected_cp, 6000, 96000, 32):
        raise ValueError(f"{frame} requires FFT1024/CP{expected_cp}, 6/96-kHz and advance-32")
    sets = [set(c[k]) for k in ("prior_records", "validation_records", "test_records")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Prior, validation and test records must be disjoint")
    return c


def load_legacy_dddd_config(channel):
    """Load the frame used by the accepted CP80 study with 256 pilots."""
    path = FROZEN_SUPPORT_DIR / f"{channel.lower()}.json"
    return _validate_config(dict(json.loads(path.read_text(encoding="utf-8")), cp=480))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def carrier_independent_config(config):
    return {key: value for key, value in config.items() if key not in DDDD_CARRIER_COUNTS}


def frozen_support_matches_config(support, config):
    """The physical delay profile is independent of the carrier allocation."""
    if support.get("frame_config_sha256") == digest(config):
        return True
    channel = config["channel"].lower()
    path = FROZEN_SUPPORT_DIR / f"{channel}.json"
    historical = json.loads(path.read_text(encoding="utf-8"))
    if historical.get("cp") != 256 or support.get("frame_config_sha256") != digest(historical):
        return False
    historical["cp"] = config["cp"]
    return carrier_independent_config(historical) == carrier_independent_config(config)
