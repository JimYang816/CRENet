"""Fixed experiment parameters and stable provenance identifiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN_SUPPORT_CONFIGS = tuple(f"configs/backups/dddd_cp256_20261125/{ch}.json" for ch in ("nof1", "ncs1"))


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


def load_config(path=None):
    c = json.loads(Path(path or ROOT / "configs/data/ncs1/dddd.json").read_text(encoding="utf-8"))
    return _validate_config(c)


def load_dddd_config(channel, *, legacy_comb=False):
    name = channel.lower()
    if not legacy_comb:
        return load_config(ROOT/f"configs/data/{name}/dddd.json")
    # The accepted CP80 comb data used this frozen frame with CP changed to 480.
    path = ROOT/f"configs/backups/dddd_cp256_20261125/{name}.json"
    historical = json.loads(path.read_text(encoding="utf-8"))
    if historical.get("cp") != 256 or historical.get("pilot_cluster") is not None:
        raise ValueError("Expected the historical four-symbol comb frame")
    return _validate_config(dict(historical, cp=480))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def frozen_support_matches_config(support, config):
    """Keep independently frozen delays traceable across the approved CP change."""
    if support.get("frame_config_sha256") == digest(config):
        return True
    channel = config["channel"].lower()
    path = ROOT/f"configs/backups/dddd_cp256_20261125/{channel}.json"
    historical = json.loads(path.read_text(encoding="utf-8"))
    if historical.get("cp") != 256 or support.get("frame_config_sha256") != digest(historical):
        return False
    historical["cp"] = config["cp"]
    return historical == config
