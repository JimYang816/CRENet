"""Exact full-support Frobenius statistics for selected physical-delay DPSS CSI."""
from __future__ import annotations

import hashlib

import numpy as np


def array_sha256(*arrays):
    result = hashlib.sha256()
    for value in arrays:
        result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def sampled_grams(sampled):
    return np.einsum("dnq,dnr->dqr", sampled.conj(), sampled, optimize=True)


def reference_statistics(sampled, reference, delays):
    reference = np.asarray(reference, np.complex128)
    cross = np.einsum("dnq,dnl->dql", sampled.conj(), reference[..., delays], optimize=True)
    energy = np.sum(abs(reference)**2, axis=(1, 2))
    if np.any(energy <= 0) or not np.all(np.isfinite(energy)):
        raise ValueError("Reference CSI must have positive finite energy in every data symbol")
    return cross, energy


def coefficient_nmse(coefficients, grams, cross, reference_energy):
    """Include every omitted physical tap; no reference projection or band truncation."""
    coefficients = np.asarray(coefficients, np.complex128)
    raw = np.einsum("...ql,dqr,...rl->...d", coefficients.conj(), grams, coefficients, optimize=True).real
    overlap = np.einsum("...ql,dql->...d", coefficients.conj(), cross, optimize=True).real
    return np.maximum((raw+reference_energy-2*overlap)/reference_energy, 0)


def reconstruct_taps(sampled, coefficients, delays, *, full_length=None):
    selected = np.einsum("dnq,...ql->...dnl", sampled, np.asarray(coefficients, np.complex128), optimize=True)
    length = int(delays[-1])+1 if full_length is None else full_length
    if length <= delays[-1]:
        raise ValueError("Output support does not include every selected physical delay")
    result = np.zeros(selected.shape[:-1]+(length,), np.complex128)
    result[..., delays] = selected
    return result


def assisted_taps(raw, reference, alpha):
    raw, reference = np.asarray(raw), np.asarray(reference)
    if raw.shape[-3:-1] != reference.shape[-3:-1] or raw.shape[-1] > reference.shape[-1]:
        raise ValueError("Raw and reference physical-delay supports do not match")
    alpha = np.asarray(alpha)
    factor = alpha.reshape(alpha.shape+(1, 1, 1))
    result = np.asarray((1-factor)*reference, np.complex128)
    result[..., :raw.shape[-1]] += factor*raw
    return result
