"""Regression checks for the three retained equalizers and their public workflows."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.numerics import DenseBackend
from src.evaluation.comparison_plots import replot
from src.evaluation.registry import (
    AVAILABLE_ALGORITHMS, IDENTIFIERS, SUPPORTED_CURVES, resolve_force, resolve_parameters,
)
from src.evaluation.unified import _preflight


RETIRED = ("mmse_dfe", "dd_ce_lmmse", "udnet", "oamp_net2")


def test_public_registry_contains_only_two_lmmse_baselines_and_crenet():
    assert IDENTIFIERS == ("single_tap_mmse", "full_matrix_lmmse")
    assert SUPPORTED_CURVES == ("Single-tap LMMSE", "Full-matrix LMMSE", "CRENet")
    assert resolve_parameters() == dict.fromkeys(IDENTIFIERS, {})
    assert resolve_force("all") == set(IDENTIFIERS)


@pytest.mark.parametrize("identifier", RETIRED)
def test_removed_algorithms_cannot_be_imported_or_requested(identifier):
    assert importlib.util.find_spec(f"src.algorithms.{identifier}") is None
    with pytest.raises(ValueError, match="Unknown algorithms"):
        resolve_parameters({identifier: {}})
    with pytest.raises(ValueError, match="Unknown force algorithms"):
        resolve_force(identifier)
    with pytest.raises(ValueError, match="Unknown learned comparator"):
        _preflight("not_opened.h5", {identifier: "not_opened.pt"}, DenseBackend())


def test_registered_lmmse_methods_match_independent_receive_space_reference():
    rng = np.random.default_rng(25)
    complex_normal = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    channel = complex_normal((2, 8, 8))
    observations = complex_normal((2, 8))
    cancelled = complex_normal((2, 8))
    factor = complex_normal((8, 8))
    covariance = factor@factor.conj().T+np.eye(8)
    variance = np.array([.3, 1.2])
    indices = np.array([1, 4, 6])
    batch = EqualizerBatch(channel, observations, cancelled, variance, indices, covariance)
    backend = DenseBackend()
    diagonal = np.diagonal(channel, axis1=-2, axis2=-1)[:, indices]
    expected_single = diagonal.conj()*observations[:, indices]/(
        abs(diagonal)**2+variance[:, None]*np.diag(covariance).real[indices])
    expected_full = []
    for h, y, noise in zip(channel[..., indices], cancelled, variance):
        expected_full.append(h.conj().T@np.linalg.solve(h@h.conj().T+noise*covariance, y))
    for algorithm, expected in zip(AVAILABLE_ALGORITHMS, (expected_single, expected_full)):
        extra = {} if algorithm.identifier == "single_tap_mmse" else {"whitener": make_whitener(covariance, backend)}
        actual = algorithm.equalize(batch, backend, **extra)
        np.testing.assert_allclose(actual.symbols, expected, rtol=2e-12, atol=2e-12)


def test_replot_keeps_three_curves_and_original_counts_without_editing_source(tmp_path):
    source, destination = tmp_path/"source", tmp_path/"retained"
    source.mkdir()
    identifiers = (*IDENTIFIERS, "crenet", *RETIRED)
    labels = (*SUPPORTED_CURVES, "MMSE-DFE", "DD-DPSS", "UDNet", "OAMP-Net2")
    curves = {label: dict(identifier=identifier, errors=[8, 1], total_bits=[100, 100], ber=[.08, .01])
              for label, identifier in zip(labels, identifiers)}
    report = dict(complete=True, input_view="raw", channel="NOF1", snr_db=[0, 30],
        frame_count=1, bits_per_snr=100, curves=curves, comparison_algorithms=list(curves),
        parameters={key: {} for key in (*IDENTIFIERS, *RETIRED)},
        checkpoint_paths={key: "historical.pt" for key in ("crenet", "udnet", "oamp_net2")},
        nmse=dict(split="test", mean_nmse_db=[-5., -10.], corrected_mean_nmse_db=[-6., -11.],
                  dd_mean_nmse_db=[-5.5, -10.5], provenance=dict(cache_key="existing")))
    path = source/"metrics.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    original = path.read_bytes()
    result = replot(source, output=destination, required_algorithms=IDENTIFIERS)
    assert tuple(result["curves"]) == SUPPORTED_CURVES
    assert result["parameters"] == dict.fromkeys(IDENTIFIERS, {})
    assert result["checkpoint_paths"] == {"crenet": "historical.pt"}
    assert "dd_mean_nmse_db" not in result["nmse"]
    for label in SUPPORTED_CURVES:
        assert result["curves"][label] == report["curves"][label]
    assert path.read_bytes() == original
    assert all((destination/name).is_file() for name in ("metrics.json", "ber_snr.png", "nmse_snr.png"))


def test_full_experiment_binds_only_new_crenet_models(tmp_path):
    from scripts.run.crenet_experiment import comparison_config
    training = json.loads(Path("configs/training/crenet.json").read_text(encoding="utf-8"))
    comparison = json.loads(Path("configs/evaluation/ber_comparison.json").read_text(encoding="utf-8"))
    result = comparison_config(training, comparison, tmp_path)
    for channel in training["datasets"]:
        assert result["checkpoints"][channel] == {
            "crenet": str(tmp_path/"joint"/channel.lower()/"crenet.pt")}
    invalid = deepcopy(comparison)
    invalid["checkpoints"] = {"NOF1": {"udnet": "retired.pt"}}
    with pytest.raises(ValueError, match="Only CRENet"):
        comparison_config(training, invalid, tmp_path)


def test_efficiency_runs_three_workers_with_only_one_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.evaluation import crenet_efficiency as efficiency
    for name in ("test.h5", "model.h5", "crenet.pt"):
        (tmp_path/name).write_bytes(b"identity-only smoke input")
    report = tmp_path/"comparison.json"
    report.write_text(json.dumps(dict(complete=True, input_view="raw")), encoding="utf-8")
    seen = []
    def worker(arguments, **kwargs):
        options = json.loads(kwargs["input"])
        seen.append(options)
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(algorithm=options["algorithm"])))
    monkeypatch.setattr(efficiency.subprocess, "run", worker)
    result = efficiency.benchmark(tmp_path/"test.h5", report, tmp_path/"crenet.pt")
    assert [row["algorithm"] for row in result["algorithms"]] == [*IDENTIFIERS, "crenet"]
    assert list(result["checkpoints"]) == ["crenet"]
    assert all({key for key in options if key.endswith("_checkpoint")} == {"crenet_checkpoint"}
               for options in seen)


@pytest.mark.parametrize("module", (
    "scripts.evaluate.compare", "scripts.evaluate.crenet", "scripts.evaluate.crenet_efficiency",
    "scripts.run.crenet_experiment",
))
def test_retained_cli_help_has_no_removed_algorithm_options(module):
    completed = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    for name in RETIRED:
        assert name.replace("_", "-") not in completed.stdout.lower()
