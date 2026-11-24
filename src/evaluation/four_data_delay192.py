"""Two-channel DDDD study with 192 independent-prior physical delay taps."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest
from src.common.experiment import write_json
from src.evaluation import ncs1_delay_support as receiver_study
from src.evaluation.four_data_lmmse import finish as finish_combined


CODE = (*receiver_study.CODE, "src/evaluation/four_data_delay192.py",
        "scripts/evaluate/four_data_delay192.py", "configs/evaluation/four_data_delay192.json")


def load_study_config(path=None):
    path = Path(path or ROOT/"configs/evaluation/four_data_delay192.json")
    options = json.loads(path.read_text(encoding="utf-8"))
    if (options.get("channels") != ["NOF1", "NCS1"]
            or type(options.get("tap_count")) is not int or options["tap_count"] != 192
            or options.get("input_view") != "raw"
            or options.get("algorithms") != ["single_tap_mmse", "full_matrix_lmmse"]
            or set(options.get("support_files", {})) != set(options["channels"])):
        raise ValueError("Both channels require 192 physical taps, raw CSI and the two LMMSE baselines")
    return options


def channel_study(channel, options):
    support_file = options["support_files"][channel]
    support = None
    if support_file:
        path = Path(support_file)
        if not path.is_absolute():
            path = ROOT/path
        support = json.loads(path.read_text(encoding="utf-8"))
        indices = np.asarray(support.get("indices", []))
        if (support.get("schema") != f"{channel.lower()}-frozen-delay-support-v1"
                or support.get("channel") != channel or support.get("tap_count") != 192
                or indices.shape != (192,) or not np.issubdtype(indices.dtype, np.integer)
                or np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= 1024
                or not support.get("prior_windows")):
            raise ValueError("Frozen support must contain 192 sorted unique physical delays")
    return dict(channel=channel, tap_count=192, support_file=support_file, frozen_support=support)


def select_prior_delays(powers, count=192):
    """Select only from independent physical PDPs; stable ties prefer lower delays."""
    powers = np.asarray(powers)
    if (powers.ndim != 2 or not len(powers) or not 0 < count <= powers.shape[1]
            or not np.all(np.isfinite(powers)) or np.any(powers < 0)
            or np.any(powers.sum(axis=1) <= 0)):
        raise ValueError("Independent prior powers must be finite, nonnegative and nonzero")
    ranking = np.argsort(-powers.mean(axis=0), kind="stable")
    return np.sort(ranking[:count])


def freeze_support(path, output, meta):
    # Deliberately read no test data when selecting or verifying delay positions.
    with h5py.File(path, "r") as h:
        powers = h["prior_power"][:]
    power = powers.mean(axis=0)
    expected = select_prior_delays(powers)
    study = dict(meta["study"])
    support = study["frozen_support"]
    if support is None:
        support = dict(schema=f"{study['channel'].lower()}-frozen-delay-support-v1",
            channel=study["channel"], tap_count=192, indices=expected.tolist(),
            frame_config_sha256=meta["frame_config_sha256"],
            selection_rule="largest mean physical tap powers over independent full-frame prior windows; stable ties prefer lower delays",
            source_git_head=meta["git_head"], source_archive_sha256=meta["archive_sha256"],
            source_profile_sha256=digest(power.tolist()), prior_windows=meta["prior_windows"],
            prior_aggregate_energy_coverage=float(power[expected].sum()/power.sum()),
            reference_use="independent prior CSI only; no test CSI used for support selection")
        study["frozen_support"] = support
    elif not meta["smoke"]:
        if support["prior_windows"] != meta["prior_windows"] or support["indices"] != expected.tolist():
            raise ValueError("Frozen delay selection does not reproduce from the full independent priors")
    write_json(output/"frozen_support192.json", support)
    return receiver_study.freeze_supports(path, output, study)


def combined_report(results, reference):
    result = results["selected192"]
    curves = {}
    for algorithm in ("single", "full"):
        for view, values in (("raw", result["curves"][algorithm]), ("reference", reference[algorithm])):
            curves[f"{view}_{algorithm}"] = dict(bit_errors=values["errors"], ber=values["ber"])
    return dict(q=24, snr_db=result["snr_db"], bits_per_snr=result["bits_per_snr"], curves=curves,
                mean_raw_nmse_db=result["mean_nmse_db"], mean_raw_nmse_linear=result["mean_nmse_linear"],
                delay_count=result["delay_count"], coefficient_count=result["coefficient_count"],
                support_span=result["support_span"], test_mean_coverage=result["test_mean_coverage"],
                test_min_coverage=result["test_min_coverage"], frames=result["frames"])


def run(output, archive_path, backend, smoke=False, *, options=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    options = load_study_config() if options is None else options
    reports, manifests = {}, {}
    code_files = tuple(dict.fromkeys((*CODE, *(path for path in options["support_files"].values() if path))))
    for channel in options["channels"]:
        directory = output/channel.lower()
        directory.mkdir(parents=True, exist_ok=True)
        study = channel_study(channel, options)
        meta = receiver_study.metadata(archive_path, backend, smoke, study, code_files=code_files)
        meta["selection"] = "192 physical delays from independent prior windows; freeze before estimating test CSI"
        meta["support_frozen"] = study["frozen_support"] is not None
        manifest_path = directory/"manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != meta:
            raise ValueError("Output directory belongs to a different 192-tap experiment")
        write_json(manifest_path, meta)
        print(f"{channel}: preparing independent priors and held-out test frames", flush=True)
        path = receiver_study.prepare(directory, archive_path, meta)
        supports, selection = freeze_support(path, directory, meta)
        print(f"{channel}: 192 physical delays, indices {supports['selected192'][0]}..{supports['selected192'][-1]}", flush=True)
        results, reference = receiver_study.evaluate(path, directory, meta, supports, backend)
        receiver_study.finish(directory, meta, selection, results, reference, path)
        reports[channel] = combined_report(results, reference)
        manifests[channel] = meta
    metadata = dict(schema="four-data-discrete192-lmmse-v1", options=options, smoke=smoke,
                    configs={channel: meta["config"] for channel, meta in manifests.items()},
                    channel_manifests=manifests)
    finish_combined(output, metadata, reports)
    return output
