"""LMMSE on a band-limited data channel with the actual thermal covariance.

Known pilots are cancelled with the supplied, untruncated CSI before detection.
The data channel is truncated before noise whitening, in physical FFT indices.
"""

from __future__ import annotations

import numpy as np

from src.algorithms.interface import EqualizerResult
from src.algorithms.likelihood import likelihood_batch, make_whitener


def band_mask(nfft, data_indices, half_bandwidth=16):
    if (
        isinstance(half_bandwidth, bool)
        or not isinstance(half_bandwidth, (int, np.integer))
        or half_bandwidth < 0
    ):
        raise ValueError("Half-bandwidth must be a nonnegative integer")
    indices = np.asarray(data_indices)
    if (
        indices.ndim != 1
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= nfft)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "Data indices must be distinct increasing physical FFT indices"
        )
    return np.abs(np.arange(nfft)[:, None] - indices[None]) <= half_bandwidth


def equalize_batch(batch, backend, *, whitener=None, half_bandwidth=16):
    """Solve (H_B^H R^-1 H_B / variance + I) x = H_B^H R^-1 y / variance."""
    b = backend
    n = batch.noise_covariance.shape[0]
    mask = band_mask(n, batch.data_indices, half_bandwidth)
    h = batch.channel
    if not (b.torch and isinstance(h, b.torch.Tensor)):
        h = b.array(h)
    if h.ndim != 3 or h.shape[-2:] != (n, n):
        raise ValueError("Expected a batch of square FFT-domain channel matrices")
    variances = np.asarray(batch.variances)
    if (
        variances.shape != (len(h),)
        or np.any(variances <= 0)
        or not np.isfinite(variances).all()
    ):
        raise ValueError("RF noise variances must be finite and positive")
    channel = h[..., batch.data_indices] * b.array(mask)
    if whitener is None:
        whitener = make_whitener(batch.noise_covariance, b)
    gram, matched = likelihood_batch(
        channel, batch.cancelled_observations, variances, whitener, b
    )
    identity = b.array(np.eye(len(batch.data_indices), dtype=np.complex128))
    solve = b.torch.linalg.solve if b.torch else np.linalg.solve
    symbols = solve(gram + identity, matched[..., None])[..., 0]
    return EqualizerResult(b.numpy(symbols), np.zeros(len(h), dtype=np.int64))
