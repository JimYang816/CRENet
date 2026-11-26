"""Validate the accepted physical CP80 source at its recorded Git revision.

Historical detector names below describe that source record only.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import h5py
from data_generation.src.config import (
    ROOT,
    FROZEN_SUPPORT_CONFIGS,
    digest,
    load_legacy_dddd_config,
)
from data_generation.src.dataset.four_data_protocol import channel_config
from src.common.experiment import file_sha256

CHANNELS = ("NOF1", "NCS1")
CODE = (
    "data_generation/src/dataset/four_data_source.py",
    "data_generation/src/dataset/four_data_targets.py",
    "data_generation/src/receiver/blocked.py",
    "data_generation/src/receiver/full_frame.py",
    "data_generation/src/receiver/chain.py",
    "data_generation/src/receiver/dpss.py",
    "data_generation/src/receiver/delay_support.py",
    "data_generation/src/receiver/estimation.py",
    "data_generation/src/receiver/pdd.py",
    "data_generation/src/transmitter/ofdm.py",
    "data_generation/src/channel/watermark/replay.py",
    "data_generation/src/config.py",
    "data_generation/src/pdd.py",
    "src/algorithms/banded_lmmse/equalizer.py",
    "src/algorithms/interface.py",
    "src/algorithms/likelihood.py",
    "src/evaluation/four_data_lmmse.py",
    "src/common/numerics.py",
    "src/common/experiment.py",
    "configs/data/nof1/delay_support192.json",
    "configs/data/ncs1/delay_support192.json",
    "configs/evaluation/four_data_delay192_compensation.json",
    *FROZEN_SUPPORT_CONFIGS,
)


def validate_protocol(original, reports):
    if (
        original.get("schema") != "four-data-cp80-lmmse-v1"
        or original.get("smoke") is not False
        or original.get("cp_ms") != 80
        or original.get("input_view") != "raw"
        or original.get("csi_compensation") is not False
        or set(reports) != set(CHANNELS)
    ):
        raise ValueError(
            "Source must be the completed raw CP80 dual-channel experiment"
        )
    expected_options = dict(
        channels=list(CHANNELS),
        cp_ms=80,
        tap_count=192,
        support_files={
            ch: f"configs/data/{ch.lower()}/delay_support192.json" for ch in CHANNELS
        },
        prior_records=list(range(51, 56)),
        prior_starts_seconds=[2, 8, 14, 20, 26],
        test_records=[11, 15, 19, 23, 27],
        start_seconds=2,
        input_view="raw",
        algorithms=["single_tap_mmse", "full_matrix_lmmse"],
    )
    if original["options"] != expected_options:
        raise ValueError("Recorded independent-prior protocol changed")
    priors = [
        [r, s]
        for r in expected_options["prior_records"]
        for s in expected_options["prior_starts_seconds"]
    ]
    tests = [[r, 2] for r in expected_options["test_records"]]
    for channel in CHANNELS:
        meta, report = original["channel_manifests"][channel], reports[channel]
        legacy = "protection_count" not in original["configs"][channel]
        c = (
            dict(load_legacy_dddd_config(channel), delay_length=192)
            if legacy
            else channel_config(channel)
        )
        support_file = expected_options["support_files"][channel]
        support = json.loads((ROOT / support_file).read_text(encoding="utf-8"))
        study = dict(
            channel=channel,
            tap_count=192,
            support_file=support_file,
            frozen_support=support,
        )
        if (
            original["configs"][channel] != c
            or meta["config"] != c
            or meta["study"] != study
            or meta.get("smoke") is not False
            or meta.get("raw_csi") is not True
            or meta.get("csi_compensation") is not False
            or meta["prior_windows"] != priors
            or meta["test_windows"] != tests
            or report.get("complete") is not True
            or report["bits_per_snr"] != 20480
            or report["delay_count"] != 192
            or report["coefficient_count"] != 4608
            or report["snr_db"] != list(range(0, 31, 5))
            or [[f["record"], f["start_seconds"]] for f in report["frames"]] != tests
        ):
            raise ValueError(
                f"{channel}: CP80 settings, fixed delays or held-out windows changed"
            )


def verify_source(source):
    source = Path(source)
    saved = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((source / "verification.json").read_text(encoding="utf-8"))
    if (
        saved.get("complete") is not True
        or saved.get("metadata") != original
        or acceptance.get("complete") is not True
        or acceptance.get("shared_dpss_basis") is not True
    ):
        raise ValueError(
            "Expected accepted, matching CP80 source manifests and reports"
        )
    validate_protocol(original, saved["channels"])
    accepted_files = {
        ch: acceptance["channels"][ch].get("physical_file") for ch in CHANNELS
    }
    if not all(accepted_files.values()):
        transfer = json.loads(
            (source / "transfer_manifest.json").read_text(encoding="utf-8")
        )
        if transfer.get("complete") is not True:
            raise ValueError("Expected a complete CP80 transfer manifest")
        accepted_files = {
            ch: transfer["physical_data_remain_on_server"][ch.lower()]
            for ch in CHANNELS
        }
    physical = {}
    for channel in CHANNELS:
        directory = source / channel.lower()
        meta = original["channel_manifests"][channel]
        detail = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        frozen = json.loads(
            (directory / "frozen_support192.json").read_text(encoding="utf-8")
        )
        if (
            json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            != meta
            or detail.get("metadata") != meta
            or detail.get("complete") is not True
            or frozen != meta["study"]["frozen_support"]
            or frozen["indices"]
            != detail["selection"]["candidates"]["selected192"]["indices"]
        ):
            raise ValueError(
                f"{channel}: saved channel provenance or fixed support changed"
            )
        for name, expected in meta["code_sha256"].items():
            payload = subprocess.check_output(
                ["git", "show", f"{meta['git_head']}:{name}"], cwd=ROOT
            )
            if expected not in (
                hashlib.sha256(payload).hexdigest(),
                hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest(),
            ):
                raise ValueError(
                    f"Recorded CP80 source does not match its committed revision: {name}"
                )
        path = directory / "physical.h5"
        identity, checksum = digest(meta), file_sha256(path)
        expected = accepted_files[channel]
        if (
            checksum != expected["sha256"]
            or path.stat().st_size != expected["bytes"]
            or identity != expected["identity"]
        ):
            raise ValueError(
                f"{channel}: physical source differs from its accepted hash"
            )
        with h5py.File(path, "r") as h:
            if (
                not h.attrs.get("complete")
                or h.attrs.get("identity") != identity
                or h["basis"].shape != (6016, 24)
                or h["prior"].shape != (25, 24, 1024)
                or [
                    [int(g.attrs["record"]), float(g.attrs["start_seconds"])]
                    for g in h["test"].values()
                ]
                != meta["test_windows"]
            ):
                raise ValueError(
                    f"{channel}: CP80 shapes, identity or test windows changed"
                )
        physical[channel] = dict(
            path=str(path),
            sha256=checksum,
            identity=identity,
            manifest_sha256=file_sha256(directory / "manifest.json"),
            frozen_support=frozen,
            frozen_support_sha256=file_sha256(directory / "frozen_support192.json"),
        )
    return original, saved["channels"], physical
