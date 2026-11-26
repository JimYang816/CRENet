"""Recalibrate only reference-assisted metadata on accepted physical datasets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest
from data_generation.src.dataset.four_data_acceptance import accept
from data_generation.src.dataset.four_data_export import calibrate
from data_generation.src.dataset.four_data_protocol import (
    SPLITS,
    channel_config,
    counts,
)
from data_generation.src.dataset.four_data_targets import channel_targets, error_scales
from src.common.experiment import file_sha256, write_json


def _plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def recalibrate_metadata(directory, c, options, targets, run_id, source_files):
    """Keep raw generation identity; explicitly identify the new CSI calibration."""
    directory = Path(directory)
    calibrate(directory, c, options, targets)
    identities = {}
    for split in SPLITS:
        path = directory / f"{split}.h5"
        with h5py.File(path, "r") as h:
            identity = digest(
                dict(
                    source_generation_identity=h.attrs["model_identity"],
                    source_file_sha256=source_files[path.name]["sha256"],
                    split=split,
                    target_nmse_db=h["compensation/target_nmse_db"][:].tolist(),
                    error_scales=h["compensation/error_scales"][:].tolist(),
                )
            )
            root_attrs = dict(calibration_identity=identity, calibration_run_id=run_id)
            group_attrs = dict(
                calibration_identity=identity,
                run_id=run_id,
                source_file_sha256=source_files[path.name]["sha256"],
                source_generation_identity=h.attrs["model_identity"],
            )
            unchanged = all(h.attrs.get(k) == v for k, v in root_attrs.items()) and all(
                h["compensation"].attrs.get(k) == v for k, v in group_attrs.items()
            )
        if not unchanged:
            with h5py.File(path, "r+") as h:
                h.attrs.update(root_attrs)
                h["compensation"].attrs.update(group_attrs)
                h.flush()
        identities[split] = identity
    return identities


def recalibrate_datasets(dataset_root, run, options, targets, backend):
    dataset_root, run = Path(dataset_root).resolve(), Path(run).resolve()
    if (ROOT / "data/processed").resolve() not in dataset_root.parents:
        raise ValueError(
            "Recalibration must target a dataset directory inside data/processed"
        )
    if (
        options["default_input_view"] != "reference_assisted"
        or options["compensation_calibration"] != "per_split"
    ):
        raise ValueError(
            "Use the explicit reference-assisted view and independent per-split calibration"
        )
    directories = {
        ch: dataset_root / (ch.lower() + "_dddd") for ch in options["channels"]
    }
    before_path = run / "calibration_before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        if before["targets"] != targets or before["directories"] != {
            ch: str(p) for ch, p in directories.items()
        }:
            raise ValueError("This run belongs to another dataset or set of targets")
    else:
        before = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            targets=targets,
            directories={ch: str(p) for ch, p in directories.items()},
            channels={},
        )
        # Preflight all six splits before the first metadata mutation.
        for channel, directory in directories.items():
            manifest = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
            if not manifest.get("complete") or manifest["channel"] != channel:
                raise ValueError("Expected a fully accepted source dataset")
            files = {}
            for name, entry in manifest["files"].items():
                path = directory / name
                checksum = file_sha256(path)
                if checksum != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                    raise ValueError(f"Accepted source file changed: {path}")
                files[name] = dict(bytes=path.stat().st_size, sha256=checksum)
            splits = {}
            curve = channel_targets(targets, channel)
            for split in SPLITS:
                with h5py.File(directory / f"{split}.h5", "r") as h:
                    if not h.attrs.get("complete") or not h.attrs.get("raw_complete"):
                        raise ValueError("Source split is incomplete")
                    mean = h["samples/raw_nmse_linear"][:].mean(axis=(1, 2))
                    error_scales(mean, curve)
                    splits[split] = dict(
                        root_attrs={k: _plain(v) for k, v in h.attrs.items()},
                        compensation_attrs={
                            k: _plain(v) for k, v in h["compensation"].attrs.items()
                        },
                        compensation={
                            k: h["compensation/" + k][:].tolist()
                            for k in h["compensation"]
                        },
                    )
            before["channels"][channel] = dict(
                files=files, manifest=manifest, splits=splits
            )
        write_json(before_path, before)
    reports = {}
    for channel, directory in directories.items():
        channel_run = run / channel.lower()
        channel_run.mkdir(parents=True, exist_ok=True)
        write_json(
            run / "progress.json",
            dict(stage="calibration_and_acceptance", channel=channel),
        )
        c = channel_config(channel)
        identities = recalibrate_metadata(
            directory,
            c,
            options,
            targets,
            run.name,
            before["channels"][channel]["files"],
        )
        print(
            f"{channel}: target CSI metadata calibrated; accepting every physical window",
            flush=True,
        )
        reports[channel] = accept(
            directory, channel_run, c, options, backend, diagnostic_ber=False
        )
        for split in SPLITS:
            achieved = reports[channel]["splits"][split]["mean_assisted_nmse_db"]
            np.testing.assert_allclose(
                achieved, channel_targets(targets, channel), rtol=0, atol=1e-8
            )
        if (
            reports[channel]["files"]["model.h5"]["sha256"]
            != before["channels"][channel]["files"]["model.h5"]["sha256"]
        ):
            raise ValueError("The frozen generation model changed during recalibration")
        reports[channel]["calibration_run_id"] = run.name
        reports[channel]["calibration_identities"] = identities
        # Append the calibration provenance without rewriting generation provenance.
        write_json(channel_run / "acceptance.json", reports[channel])
        write_json(directory / "manifest.json", reports[channel])
    if len({r["basis_sha256"] for r in reports.values()}) != 1:
        raise ValueError("Channels do not share the same actual DPSS basis")
    manifest = dict(
        complete=True,
        reference_assisted=True,
        compensation_calibration="per_split",
        targets=targets,
        reports=reports,
        source_before=str(before_path),
        raw_generation_identity_preserved=True,
        counts={ch: counts(channel_config(ch)) for ch in directories},
    )
    write_json(run / "dataset_manifest.json", manifest)
    return reports
