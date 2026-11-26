"""Regression checks for the LMMSE baselines and their public workflows."""
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.numerics import DenseBackend
from src.evaluation.comparison_plots import replot
from src.evaluation.registry import (
    AVAILABLE_ALGORITHMS, CURVES, IDENTIFIERS, resolve_force, resolve_parameters,
)


RETIRED = ("crenet", "mmse_dfe", "dd_ce_lmmse", "udnet", "oamp_net2")


def test_public_registry_contains_both_lmmse_baselines():
    assert IDENTIFIERS == ("single_tap_mmse", "full_matrix_lmmse")
    assert CURVES == ("Single-tap LMMSE", "Full-matrix LMMSE")
    assert resolve_parameters() == dict.fromkeys(IDENTIFIERS, {})
    assert resolve_force("all") == set(IDENTIFIERS)


@pytest.mark.parametrize("identifier", RETIRED)
def test_removed_algorithms_cannot_be_imported_or_requested(identifier):
    assert importlib.util.find_spec(f"src.algorithms.{identifier}") is None
    with pytest.raises(ValueError, match="Unknown algorithms"):
        resolve_parameters({identifier: {}})
    with pytest.raises(ValueError, match="Unknown force algorithms"):
        resolve_force(identifier)


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


def test_replot_keeps_baseline_counts_and_original_report(tmp_path):
    source, destination = tmp_path/"source", tmp_path/"retained"
    source.mkdir()
    identifiers = (*IDENTIFIERS, *RETIRED)
    labels = (*CURVES, "ICR-MMSE-Net", "MMSE-DFE", "DD-DPSS", "UDNet", "OAMP-Net2")
    curves = {label: dict(identifier=identifier, errors=[8, 1], total_bits=[100, 100], ber=[.08, .01])
              for label, identifier in zip(labels, identifiers)}
    report = dict(schema="watermark-unified-comparison-v1", complete=True, input_view="raw", channel="NOF1", snr_db=[0, 30],
        frame_count=1, bits_per_snr=100, curves=curves, comparison_algorithms=list(curves),
        parameters={key: {} for key in (*IDENTIFIERS, *RETIRED)},
        checkpoint_paths={key: "historical.pt" for key in ("crenet", "udnet", "oamp_net2")},
        crenet=dict(checkpoint="historical.pt"),
        nmse=dict(split="test", mean_nmse_db=[-5., -10.], corrected_mean_nmse_db=[-6., -11.],
                  dd_mean_nmse_db=[-5.5, -10.5], provenance=dict(cache_key="existing")))
    path = source/"metrics.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    original = path.read_bytes()
    result = replot(source, output=destination, required_algorithms=IDENTIFIERS)
    assert tuple(result["curves"]) == CURVES
    assert result["parameters"] == dict.fromkeys(IDENTIFIERS, {})
    assert "checkpoint_paths" not in result and "crenet" not in result
    assert "dd_mean_nmse_db" not in result["nmse"] and "corrected_mean_nmse_db" not in result["nmse"]
    for label in CURVES:
        assert result["curves"][label] == report["curves"][label]
    assert path.read_bytes() == original
    assert all((destination/name).is_file() for name in ("metrics.json", "ber_snr.png", "nmse_snr.png"))
    html = (destination/"report.html").read_text(encoding="utf-8")
    assert all(label in html for label in CURVES)
    assert "CRENet" not in html


@pytest.mark.parametrize("module", (
    "scripts.evaluate.compare", "scripts.evaluate.four_data_cp80",
))
def test_retained_cli_help_has_no_removed_algorithm_options(module):
    completed = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    for name in RETIRED:
        assert name.replace("_", "-") not in completed.stdout.lower()
