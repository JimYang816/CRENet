"""Reconstruct the explicitly selected raw or reference-assisted CSI view."""

import numpy as np
from data_generation.src.dataset.four_data_math import assisted_taps, reconstruct_taps
from src.algorithms.interface import EqualizerBatch
from src.evaluation.four_data_lmmse import channel_matrices


def prepare_batch(dataset, si, frames, d, backend):
    h = dataset._h
    variance = h["samples/rf_noise_variance"][si, frames]
    observed = (
        h["samples/clean_observations"][frames, d]
        + np.sqrt(variance[:, None]) * h["samples/standard_noise"][frames, d]
    )
    taps = reconstruct_taps(
        dataset.sampled_basis[d : d + 1],
        h["samples/raw_coefficients"][si, frames],
        dataset.delays,
    )[:, 0]
    if dataset.input_view == "reference_assisted":
        windows = h["samples/window_index"][frames]
        unique, inverse = np.unique(windows, return_inverse=True)
        reference = np.stack(
            [h["windows/reference_taps"][int(wi), d] for wi in unique]
        ).astype(np.complex128)
        taps = assisted_taps(
            taps, reference[inverse], h["compensation/error_scales"][si]
        )
    channel = channel_matrices(taps, backend)
    observed = backend.array(observed)
    cancelled = observed - channel @ backend.array(dataset.known[d])
    return EqualizerBatch(
        channel,
        observed,
        cancelled,
        variance,
        np.flatnonzero(dataset.masks["data"]),
        dataset._model["unit_rf_noise_covariance"][:],
    )
