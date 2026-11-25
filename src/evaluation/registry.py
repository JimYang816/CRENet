"""Stable detector identifiers and explicit computation dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import importlib


@dataclass(frozen=True)
class Algorithm:
    identifier: str
    label: str
    module: str
    helpers: tuple[str, ...] = ()

    def equalize(self, *args, **kwargs):
        return importlib.import_module(self.module).equalize_batch(*args, **kwargs)


MATRIX = ("src.algorithms.likelihood:make_whitener", "src.algorithms.likelihood:likelihood_batch",
          "src.algorithms.likelihood:batch_likelihood")
ALGORITHMS = (
    Algorithm("single_tap_mmse", "Single-tap LMMSE", "src.algorithms.single_tap_mmse.equalizer"),
    Algorithm("full_matrix_lmmse", "Full-matrix LMMSE", "src.algorithms.full_matrix_lmmse.equalizer", MATRIX),
)
AVAILABLE_ALGORITHMS = ALGORITHMS
CURVES = tuple(a.label for a in ALGORITHMS)
IDENTIFIERS = tuple(a.identifier for a in AVAILABLE_ALGORITHMS)
SUPPORTED_CURVES = CURVES+("CRENet",)


def normalize_report_labels(report):
    """Translate historical display names without changing numerical provenance."""
    aliases = {"Single-tap MMSE": ALGORITHMS[0].label,
               "ICR-MMSE-Net": "CRENet", "ICR-MMSE": "CRENet", "CRU-Net": "CRENet"}
    curves = {}
    for name, curve in report.get("curves", {}).items():
        label = aliases.get(name, name)
        if label in curves:
            raise ValueError(f"Duplicate report curve after label normalization: {label}")
        # Historical reports retain their original numerical provenance. Only
        # normalize the display label and public detector identifier on read.
        if curve.get("identifier") == "icr_mmse":
            curve = dict(curve, identifier="crenet")
        curves[label] = curve
    normalized = dict(report, curves=curves)
    if "comparison_algorithms" in report:
        normalized["comparison_algorithms"] = [aliases.get(name, name) for name in report["comparison_algorithms"]]
    return normalized


def resolve_parameters(algorithm_parameters=None):
    supplied = algorithm_parameters or {}
    if set(supplied)-set(IDENTIFIERS):
        raise ValueError(f"Unknown algorithms: {sorted(set(supplied)-set(IDENTIFIERS))}")
    result = {}
    for algorithm in AVAILABLE_ALGORITHMS:
        override = supplied.get(algorithm.identifier, {})
        if not isinstance(override, dict) or override:
            raise ValueError(f"Invalid parameters for {algorithm.identifier}")
        result[algorithm.identifier] = {}
    return result


def selected_algorithms(parameters):
    return tuple(a for a in AVAILABLE_ALGORITHMS if a.identifier in parameters)


def resolve_force(force):
    selected = {force} if isinstance(force, str) else set(force)
    if selected-set(IDENTIFIERS)-{"all"}:
        raise ValueError(f"Unknown force algorithms: {sorted(selected-set(IDENTIFIERS)-{'all'})}")
    return set(IDENTIFIERS) if "all" in selected else selected
