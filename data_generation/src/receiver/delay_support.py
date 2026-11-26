"""Independent physical-delay profiling and selected-delay BEM coefficients."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from data_generation.src.receiver.dpss import DPSSModel


@dataclass
class SelectedDelayBEM(DPSSModel):
    """Store K coefficients while retaining each tap's actual circular delay."""

    tail_ratio: float | None
    delay_indices: np.ndarray

    def __post_init__(self):
        indices = np.asarray(self.delay_indices)
        if (indices.shape != (self.length,) or not np.issubdtype(indices.dtype, np.integer)
                or self.length < 1 or np.any(np.diff(indices) <= 0)
                or indices[0] < 0 or indices[-1] >= self.c["nfft"]):
            raise ValueError("Selected delays must be sorted, unique integers inside the FFT support")
        self.delay_indices = indices.copy()

    @property
    def delays(self):
        return self.delay_indices

    def coefficients(self, taps):
        g = np.asarray(taps)[..., self.delays].reshape(-1, 4*self.c["nfft"], self.length)
        return np.einsum("qn,mnl->mql", linalg.pinv(self.sampled_basis), g, optimize=True)

    def coefficients_full_frame(self, taps):
        g = np.asarray(taps)
        if (g.ndim < 2 or g.shape[-2] != len(self.basis)
                or g.shape[-1] <= self.delays[-1] or not np.all(np.isfinite(g))):
            raise ValueError("Full-frame taps must include all selected physical delays")
        g = g[..., self.delays].reshape(-1, len(self.basis), self.length)
        return np.einsum("qn,mnl->mql", linalg.pinv(self.basis), g, optimize=True)

    def expand_taps(self, taps):
        """Restore physical positions before an FFT, equalization, or scoring."""
        taps = np.asarray(taps)
        if taps.shape[-1] != self.length:
            raise ValueError("Selected taps do not match the delay support")
        dense = np.zeros(taps.shape[:-1]+(int(self.delays[-1])+1,), taps.dtype)
        dense[..., self.delays] = taps
        return dense

    def reconstruct(self, coefficients, symbol=1):
        return self.expand_taps(super().reconstruct(coefficients, symbol))


SelectedDelayDPSSModel = SelectedDelayBEM


def profile_full_frame(receiver, sounding, start, basis):
    """Fit every circular-delay column and profile unprojected physical energy.

    Only independent prior recordings should enter this function when selecting
    a receiver support. The receiver retains its actual RF and FIR operators.
    """
    return project_full_frame(receiver, sounding, start, linalg.pinv(basis))


def project_full_frame(receiver, sounding, start, projection):
    """Apply one or more stacked basis pseudoinverses to the same physical kernel."""
    c = receiver.chain.c
    n, total = c["nfft"], projection.shape[1]
    origin = start-c["guard_samples"]/c["fs_baseband"]
    coefficients, power = np.zeros((projection.shape[0], n), complex), np.zeros(n)
    for begin in range(0, total, receiver.block):
        end = min(begin+receiver.block, total)
        indices = c["guard_samples"]+np.arange(begin, end)
        operator = receiver.reference_rows(sounding, origin, indices)
        cols = indices[:, None]+c["fft_advance"]-np.arange(n)
        rows = np.broadcast_to(np.arange(end-begin)[:, None], cols.shape)
        valid = (cols >= 0) & (cols < operator.shape[1])
        taps = np.zeros(cols.shape, complex)
        taps[valid] = np.asarray(operator[rows[valid], cols[valid]]).ravel()
        coefficients += projection[:, begin:end] @ taps
        power += np.sum(abs(taps)**2, axis=0)
    return coefficients, power/total
