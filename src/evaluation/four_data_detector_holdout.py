"""Freeze one validation-selected detector candidate and score the paired test set."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from data_generation.src.config import ROOT, digest
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_crenet import ALGORITHMS, evaluate
from src.training.four_data_crenet import save_checkpoint, selection_key


SOURCE_FILES = (
    "scripts/evaluate/four_data_detector_holdout.py", "src/evaluation/four_data_detector_holdout.py",
    "src/evaluation/four_data_crenet.py", "src/training/four_data_crenet.py",
    "src/data/four_data_training.py", "src/data/four_data.py", "src/algorithms/crenet/model.py",
    "src/algorithms/crenet/operators.py", "src/algorithms/crenet/lmmse.py",
    "src/algorithms/crenet/dpss.py", "src/algorithms/crenet/metrics.py",
)


def validate_selection(diagnostic, candidate, original, variant):
    """Selection is frozen from validation, before the test dataset is opened."""
    if (not diagnostic["complete"] or diagnostic["test_used"]
            or diagnostic["source_checkpoint_stage"] != "B"
            or diagnostic["validation_error_count_difference"] != 0):
        raise ValueError("Expected a completed, reproduced validation-only stage-B diagnosis")
    record = diagnostic["variants"][variant]
    if (candidate["variant"] != variant or candidate["interval"] != record["best_interval"]
            or candidate["source_identity"] != diagnostic["original_identity"]
            or candidate["validation"] != record["best_validation"]):
        raise ValueError("Candidate checkpoint does not match the frozen validation choice")
    if (original["identity"] != diagnostic["original_identity"] or not record["beats_baseline"]
            or selection_key("C", candidate["validation"]) >= selection_key("C", original["validation"])):
        raise ValueError("Candidate must beat the original final model under the validation selection rule")
    changes = record["intervals"][record["best_interval"]-1]["changes"]
    if not all(changes[name]["changed_tensors"] > 0 for name in ("scalars", "neural")):
        raise ValueError("This holdout requires trained physical and neural detector parameters")
    return record


def paired_comparison(original, candidate, old_errors, new_errors):
    """Summarize paired error-count changes without treating correlated bits as independent."""
    if (original["channel"] != candidate["channel"] or original["input_view"] != candidate["input_view"]
            or original["snr_db"] != candidate["snr_db"] or original["frames"] != candidate["frames"]
            or original["bits_per_snr"] != candidate["bits_per_snr"]):
        raise ValueError("Original and candidate test identities do not match")
    shape = (len(original["snr_db"]), len(original["frames"]), 4)
    for name in ALGORITHMS:
        if old_errors[name].shape != shape or new_errors[name].shape != shape:
            raise ValueError("Unexpected paired test error-count dimensions")
        if name != "crenet" and not np.array_equal(old_errors[name], new_errors[name]):
            raise ValueError(f"Paired baseline error counts changed: {name}")
    delta = new_errors["crenet"]-old_errors["crenet"]
    records = sorted({frame["record"] for frame in original["frames"]})
    record_delta = {str(record): int(delta[:, [i for i, frame in enumerate(original["frames"])
                                                    if frame["record"] == record], :].sum())
                    for record in records}
    return dict(snr_db=original["snr_db"], bits_per_snr=original["bits_per_snr"],
        original_ber=original["curves"]["crenet"]["ber"], candidate_ber=candidate["curves"]["crenet"]["ber"],
        error_change_by_snr=delta.sum(axis=(1, 2)).tolist(), error_change_total=int(delta.sum()),
        blocks_improved=int((delta < 0).sum()), blocks_unchanged=int((delta == 0).sum()),
        blocks_worsened=int((delta > 0).sum()), error_change_by_record=record_delta,
        original_corrected_nmse_db=original["corrected_nmse_db"],
        candidate_corrected_nmse_db=candidate["corrected_nmse_db"],
        initial_nmse_db=candidate["initial_nmse_db"], full_lmmse_ber=candidate["curves"]["full_matrix_lmmse"]["ber"])


def run(directory, previous, diagnosis, output, *, variant, device):
    directory, previous, diagnosis, output = map(Path, (directory, previous, diagnosis, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty holdout directory: {output}")
    diagnostic = json.loads((diagnosis/"summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((previous/"provenance.json").read_text(encoding="utf-8"))
    original = torch.load(previous/"model.pt", map_location="cpu", weights_only=False)
    candidate_path = diagnosis/f"{variant}.pt"
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    record = validate_selection(diagnostic, candidate, original, variant)
    if (file_sha256(previous/"best_B.pt") != diagnostic["original_checkpoint_sha256"]
            or file_sha256(directory/"manifest.json") != diagnostic["manifest_sha256"]
            or provenance["identity"] != diagnostic["original_identity"]):
        raise ValueError("Diagnosis provenance or accepted dataset manifest changed")
    config = diagnostic["config"]
    if config["model"] != original["config"] or config["input_view"] != original["input_view"]:
        raise ValueError("Candidate and original model/input configuration differ")
    sources = {name: file_sha256(ROOT/name) for name in SOURCE_FILES}
    frozen = dict(schema="four-data-detector-holdout-v1", channel=diagnostic["channel"],
        variant=variant, updates=record["best_interval"]*diagnostic["steps_per_interval"],
        candidate_sha256=file_sha256(candidate_path), diagnosis_sha256=file_sha256(diagnosis/"summary.json"),
        original_identity=provenance["identity"], original_model_sha256=file_sha256(previous/"model.pt"),
        manifest_sha256=diagnostic["manifest_sha256"], candidate_validation=record["best_validation"],
        original_validation=original["validation"], selection_uses_test=False,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256=sources)
    identity = digest(frozen)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"frozen_selection.json", dict(frozen, identity=identity))
    # Access the test file only after the candidate and source identity are frozen.
    for name in ("model.h5", "test.h5"):
        expected = provenance["data"]["files"][name]
        path = directory/name
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Test dataset file differs from the original experiment: {name}")
    save_checkpoint(output/"model.pt", dict(identity=identity, model=candidate["model"],
        config=config["model"], input_view=config["input_view"], stage="validation-selected-detector",
        epoch=frozen["updates"], protocol="DDDD", validation=candidate["validation"]))
    report = evaluate(config, directory, output, identity, device=device)
    old_report = json.loads((previous/"metrics.json").read_text(encoding="utf-8"))
    if old_report["identity"] != provenance["identity"]:
        raise ValueError("Original test report identity does not match the accepted experiment")
    with np.load(previous/"frame_metrics.npz") as old, np.load(output/"frame_metrics.npz") as new:
        if not old["complete"].all() or not new["complete"].all():
            raise ValueError("Paired test has incomplete symbol blocks")
        comparison = paired_comparison(old_report, report, old, new)
    comparison.update(complete=True, identity=identity, frozen_selection=frozen,
                      criterion="candidate minus original; negative error counts indicate improvement")
    write_json(output/"comparison.json", comparison)
    plot(comparison, output)
    return comparison


def plot(comparison, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snr = comparison["snr_db"]
    for kind in ("ber", "nmse"):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        if kind == "ber":
            curves = (("Original CRENet", comparison["original_ber"]),
                      ("Fine-tuned CRENet", comparison["candidate_ber"]),
                      ("Full-matrix LMMSE", comparison["full_lmmse_ber"]))
            for label, values in curves:
                axis.semilogy(snr, np.maximum(values, .5/comparison["bits_per_snr"]), marker="o", label=label)
        else:
            curves = (("Input CSI", comparison["initial_nmse_db"]),
                      ("Original CRENet", comparison["original_corrected_nmse_db"]),
                      ("Fine-tuned CRENet", comparison["candidate_corrected_nmse_db"]))
            for label, values in curves:
                axis.plot(snr, values, marker="o", label=label)
        axis.set(xlabel="SNR (dB)", ylabel="BER" if kind == "ber" else "Mean full-matrix NMSE (dB)",
                 title=f"{comparison['frozen_selection']['channel']} paired held-out test")
        axis.grid(True, which="both", alpha=.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(Path(output)/f"{kind}_snr.{suffix}", dpi=180)
        plt.close(fig)
