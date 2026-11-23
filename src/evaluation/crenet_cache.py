"""Reuse complete CRENet reports when their scientific identities match."""
from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path

from src.algorithms.crenet.checkpoint import INFERENCE_CODE, load_model, validate_dataset
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.evaluation import crenet
from src.evaluation.fingerprints import DataIdentity, digest, numerical_environment, source_fingerprint
from src.evaluation.result_cache import DEFAULT_CACHE, FileLock

LOG = logging.getLogger(__name__)
SCHEMA = "crenet-complete-report-cache-v1"


def _matches(report, expected, counts, bootstrap_samples):
    """Check scientific identity and full coverage before accepting an old report."""
    try:
        if (report["schema"] != crenet.SCHEMA or not report["complete"]
                or any(report.get(k) != v for k, v in expected.items())):
            return False
        if len(report["per_snr"]) != len(expected["snr_db"]):
            return False
        curve = report["curves"]["CRENet"]
        if any(len(curve[key]) != len(expected["snr_db"]) for key in ("ber", "errors", "total_bits")):
            return False
        for i, row in enumerate(report["per_snr"]):
            if (row["snr_db"] != expected["snr_db"][i] or row["total_bits"] != counts["bits"]
                    or row["blocks"] != counts["blocks"] or curve["total_bits"][i] != counts["bits"]
                    or curve["errors"][i] != row["bit_errors"][-1]
                    or curve["ber"][i] != row["bit_errors"][-1]/counts["bits"]
                    or sum(r["bits"] for r in row["by_record"]) != counts["bits"]
                    or sum(r["bit_errors"][-1] for r in row["by_record"]) != row["bit_errors"][-1]):
                return False
            for interval in row["ber_intervals"]+row["nmse_intervals"]:
                if interval["records"] > 1 and interval.get("resamples") != bootstrap_samples:
                    return False
        return report["performance"]["blocks"] == counts["blocks"]*len(expected["snr_db"])
    except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
        return False


def _previous_reports(checkpoint, run):
    yield run/"metrics.json"
    # Current training branches keep reports under their original experiment.
    # No directory-wide search and no dependence on another algorithm's report.
    if checkpoint.parent.parent.name in ("joint", "joint_training", "fixed", "fixed_final"):
        experiment, channel = checkpoint.parents[2], checkpoint.parent.name
        joint = checkpoint.parent.parent.name in ("joint", "joint_training")
        base = experiment/("comparison" if joint else "fixed_evaluation")/channel
        yield base/("crenet/metrics.json" if joint else "metrics.json")
        if joint:
            yield base/"icr_mmse/metrics.json"  # Historical directory; identity is still verified.


def evaluate(dataset_path, checkpoint, run, backend, *, batch_size=8, cache_dir=None,
             bootstrap_samples=2000, force=False):
    """Cache complete test metrics; retain the source timing as a historical measurement."""
    if batch_size < 1 or bootstrap_samples < 1:
        raise ValueError("Batch size and bootstrap samples must be positive")
    dataset_path, checkpoint, run = Path(dataset_path), Path(checkpoint).resolve(), Path(run)
    run.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = file_sha256(checkpoint)
    model, payload = load_model(checkpoint, "cpu")
    effective_config = dict(model.config)
    del model
    config = payload["model_config"]
    references = INFERENCE_CODE+("src.evaluation.crenet",)
    if config["detector_mode"] == "budgeted":
        references += ("src.algorithms.crenet.inference",)
    code = source_fingerprint(references)
    with CompactDataset(dataset_path, input_view="raw") as dataset:
        validate_dataset(payload, dataset)
        if dataset.split != "test":
            raise ValueError("Cached CRENet comparison requires the test split")
        guard = DataIdentity({name: dataset.path.parent/name for name in
                              ("test.h5", "model.h5", "train.h5", "validation.h5")})
        expected = dict(channel=dataset.channel, split="test", input_view="raw",
            smoke=bool(payload.get("smoke")) or bool(dataset._config.get("smoke")),
            checkpoint_sha256=checkpoint_hash, model_config=config, effective_model_config=effective_config,
            data_identity=guard.files, code=code, snr_db=dataset.snr_db.tolist(), operator_backend="dense",
            numerical_environment=dict(numerical_environment(backend), precision=config["precision"]+"/"+
                ("float32" if config["precision"] == "complex64" else "float64")))
        counts = dict(blocks=dataset.frame_count*2, bits=dataset.frame_count*4*len(dataset._indices))
    identity = dict(schema=SCHEMA, expected=expected, counts=counts, bootstrap_samples=bootstrap_samples)
    entry = (Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE)/"crenet_reports"/digest(identity)
    entry.mkdir(parents=True, exist_ok=True)
    cached = entry/"metrics.json"

    def unchanged():
        guard.assert_unchanged()
        if file_sha256(checkpoint) != checkpoint_hash or source_fingerprint(references) != code:
            raise RuntimeError("CRENet checkpoint or computation code changed during cache access")

    with FileLock(entry/"entry.lock"):
        candidates = []
        if not force:
            try:
                if file_sha256(cached) == json.loads((entry/"checksum.json").read_text())["sha256"]:
                    candidates.append(cached)
            except (OSError, ValueError, KeyError, TypeError):
                pass
            candidates.extend(_previous_reports(checkpoint, run))
        report, source = None, None
        for path in candidates:
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if _matches(candidate, expected, counts, bootstrap_samples):
                report, source = deepcopy(candidate), path.resolve()
                break
        if report is None:
            report = crenet.evaluate(dataset_path, checkpoint, run, backend, batch_size=batch_size,
                cache_dir=cache_dir, bootstrap_samples=bootstrap_samples)
            if not _matches(report, expected, counts, bootstrap_samples):
                raise RuntimeError("New CRENet evaluation failed its cache identity/coverage check")
        unchanged()
        # Cache only CRENet; changes to optional comparators must not affect it.
        report["curves"] = {"CRENet": report["curves"]["CRENet"]}
        for key in ("baselines", "comparison_precision", "udnet_checkpoint", "evaluation_cache"):
            report.pop(key, None)
        report.setdefault("measurement_source_report", str(source) if source is not None else str((run/"metrics.json").resolve()))
        report["checkpoint"] = str(checkpoint)
        report["dataset"] = str(dataset_path.resolve())
        if source != cached.resolve():
            write_json(cached, report)
            write_json(entry/"checksum.json", dict(sha256=file_sha256(cached)))
        report["evaluation_cache"] = dict(reused=source is not None, cache_file=str(cached.resolve()),
            source_report=str(source) if source is not None else str((run/"metrics.json").resolve()),
            measurement_source_report=report["measurement_source_report"],
            requested_batch_size=batch_size,
            timing="Source evaluation measurement; cache reuse does not remeasure inference")
        with FileLock(run/".crenet_evaluation.lock"):
            if source is not None:
                crenet.plot_report(report, run)
                LOG.info("%s CRENet: reuse complete report from %s", expected["channel"], source)
            write_json(run/"metrics.json", report)
            write_json(run/"progress.json", dict(phase="finished", complete=True,
                processed_blocks=counts["blocks"]*len(expected["snr_db"]), cache_reused=source is not None))
        unchanged()
        return report
