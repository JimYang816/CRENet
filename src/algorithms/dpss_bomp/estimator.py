"""Noise-whitened delay-group OMP with regularized active-set refitting.

Each group contains all Q coefficients for one delay.  The group coordinate is
scaled by the independently fitted receiver-prior marginal covariance; the
same complete DPSS basis and delay support are retained.  No reference CSI,
data symbols, or data-derived decision feedback enter this estimator.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from data_generation.src.receiver.estimation import input_windows


def coefficient_transforms(blocks, q, length):
    """Marginal covariance square root for each delay (not cross-delay prior)."""
    result = np.zeros((length, q, q), dtype=np.complex128)
    covered = np.zeros(length, dtype=bool)
    for start, stop, temporal, delay in blocks:
        if not 0 <= start < stop <= length or np.any(covered[start:stop]):
            raise ValueError("Prior delay blocks overlap or are outside support")
        temporal, delay = np.asarray(temporal), np.asarray(delay)
        if temporal.shape != (q, q) or delay.shape != (stop-start, stop-start):
            raise ValueError("Invalid receiver prior block dimensions")
        lt = linalg.cholesky((temporal+temporal.conj().T)/2, lower=True)
        diagonal = np.diag(delay).real
        if np.min(diagonal) < -1e-10*max(float(np.max(abs(diagonal))), 1e-30):
            raise ValueError("Negative prior delay variance")
        result[start:stop] = np.sqrt(np.maximum(diagonal, 0))[:, None, None]*lt
        covered[start:stop] = True
    return result


def selection_rows(masks, n):
    tracking = np.flatnonzero(masks["tracking"])
    return (np.arange(n), tracking, tracking, np.arange(n))


def selected_observations(observations, rows):
    y = np.asarray(observations)
    if y.ndim != 3 or y.shape[1] != 4:
        raise ValueError("Expected batch x four symbols x frequency observations")
    return np.concatenate([y[:, s, row] for s, row in enumerate(rows)], axis=1)


def build_dictionary(model, known, masks, noise, blocks, backend):
    """Whiten public-pilot response and return delay-major normalized groups."""
    n, q, length = model.c["nfft"], model.count, model.length
    known = np.asarray(known, dtype=np.complex128)
    if known.shape != (4, n) or np.any(known[[1, 2]][:, masks["data"]] != 0):
        raise ValueError("Data tones in public pilots must be zero")
    noise = np.asarray(noise, dtype=np.complex128)
    if noise.shape != (n, n):
        raise ValueError("Unexpected receiver noise covariance")
    rows = selection_rows(masks, n)
    basis = model.sampled_basis.reshape(4, n, q)
    windows = input_windows(known, length)
    transforms = coefficient_transforms(blocks, q, length)
    whiteners = []
    for row in rows:
        cov = noise[np.ix_(row, row)]
        chol = linalg.cholesky((cov+cov.conj().T)/2, lower=True)
        whiteners.append(linalg.solve_triangular(chol, np.eye(len(row)), lower=True))
    if backend.torch is None:
        parts = []
        for s, row in enumerate(rows):
            time = windows[s, :, :, None]*basis[s, :, None, :]
            response = np.fft.fftshift(np.fft.fft(time, axis=0, norm="ortho"), axes=0)[row]
            whitened = (whiteners[s] @ response.reshape(len(row), -1)).reshape(len(row), length, q)
            parts.append(np.einsum("mlq,lqr->mlr", whitened, transforms, optimize=True))
        return np.concatenate(parts), transforms, rows, whiteners
    t = backend.torch
    parts, device_whiteners = [], []
    trans = t.as_tensor(transforms, dtype=t.complex64, device="cuda")
    for s, row in enumerate(rows):
        w = t.as_tensor(windows[s], dtype=t.complex64, device="cuda")
        b = t.as_tensor(basis[s], dtype=t.complex64, device="cuda")
        time = w[:, :, None]*b[:, None, :]
        response = t.fft.fftshift(t.fft.fft(time, dim=0, norm="ortho"), dim=0)
        response = response.index_select(0, t.as_tensor(row, device="cuda"))
        whitener = t.as_tensor(whiteners[s], dtype=t.complex64, device="cuda")
        device_whiteners.append(whitener)
        whitened = (whitener @ response.reshape(len(row), -1)).reshape(len(row), length, q)
        parts.append(t.einsum("mlq,lqr->mlr", whitened, trans))
    return t.cat(parts, dim=0).contiguous(), trans, rows, device_whiteners


def whiten_observations(observations, rows, whiteners, backend):
    selected = selected_observations(observations, rows)
    if backend.torch is None:
        result, start = [], 0
        for row, whitener in zip(rows, whiteners):
            stop = start+len(row)
            result.append(selected[:, start:stop] @ whitener.T)
            start = stop
        return np.concatenate(result, axis=1)
    t = backend.torch
    y = t.as_tensor(selected, dtype=t.complex64, device="cuda")
    result, start = [], 0
    for row, whitener in zip(rows, whiteners):
        stop = start+len(row)
        w = t.as_tensor(whitener, dtype=t.complex64, device="cuda")
        result.append(y[:, start:stop] @ w.T)
        start = stop
    return t.cat(result, dim=1)


class GroupOMP:
    """Greedy delay selection with a ridge solve over every active coefficient."""

    def __init__(self, dictionary, transforms, backend):
        self.a, self.transforms, self.backend = dictionary, transforms, backend
        if len(dictionary.shape) != 3:
            raise ValueError("Expected observations x delay x basis dictionary")
        self.m, self.length, self.q = dictionary.shape
        if tuple(transforms.shape) != (self.length, self.q, self.q):
            raise ValueError("Invalid group transforms")
        if backend.torch:
            t = backend.torch
            self.norm = t.sum(t.abs(dictionary).square(), dim=(0, 2))
            self.active = self.norm > self.norm.max()*1e-14
            self.flat = dictionary.reshape(self.m, -1)
        else:
            self.norm = np.sum(abs(dictionary)**2, axis=(0, 2))
            self.active = self.norm > self.norm.max()*1e-14
            self.flat = dictionary.reshape(self.m, -1)
        if not bool(self.active.any()):
            raise ValueError("All delay groups have zero observation response")

    def estimate(self, y, ridge, checkpoints):
        """Yield coefficients at each selected group count; y is already whitened."""
        checkpoints = tuple(sorted(set(int(k) for k in checkpoints)))
        if (not checkpoints or checkpoints[0] < 1 or checkpoints[-1] > self.length
                or checkpoints[-1]*self.q > self.m or checkpoints[-1] > int(self.active.sum())):
            raise ValueError("Requested group count exceeds the observation budget")
        if self.backend.torch:
            return self._torch(y, ridge, checkpoints)
        return self._numpy(np.asarray(y), np.asarray(ridge), checkpoints)

    def _numpy(self, y, ridge, checkpoints):
        if y.ndim != 2 or y.shape[1] != self.m or np.shape(ridge) != (len(y),) or np.any(ridge <= 0):
            raise ValueError("Invalid whitened observations or regularizer")
        outputs = {k: np.zeros((len(y), self.q, self.length), complex) for k in checkpoints}
        for i, target in enumerate(y):
            residual, chosen = target.copy(), []
            for k in range(1, checkpoints[-1]+1):
                correlations = (self.flat.conj().T @ residual).reshape(self.length, self.q)
                score = np.sum(abs(correlations)**2, axis=1)/np.maximum(self.norm, 1e-300)
                score[~self.active] = -np.inf
                score[chosen] = -np.inf
                chosen.append(int(np.argmax(score)))
                matrix = self.a[:, chosen, :].reshape(self.m, -1)
                gram = matrix.conj().T @ matrix + ridge[i]*np.eye(k*self.q)
                z = np.linalg.solve(gram, matrix.conj().T @ target)
                residual = target-matrix@z
                if k in outputs:
                    c = np.einsum("kqr,kr->kq", self.transforms[chosen], z.reshape(k, self.q))
                    outputs[k][i][:, chosen] = c.T
        return outputs

    def _torch(self, y, ridge, checkpoints):
        t = self.backend.torch
        if y.ndim != 2 or y.shape[1] != self.m or tuple(ridge.shape) != (len(y),) or bool((ridge <= 0).any()):
            raise ValueError("Invalid whitened observations or regularizer")
        batch = len(y)
        target = y[..., None]
        residual = target
        masked = (~self.active)[None].expand(batch, -1).clone()
        eye = t.eye(self.q, dtype=self.a.dtype, device=self.a.device)[None]
        chosen, active_matrix, factor, rhs, outputs = [], None, None, None, {}
        for k in range(1, checkpoints[-1]+1):
            corr = (self.flat.mH @ residual[..., 0].T).T.reshape(batch, self.length, self.q)
            score = t.sum(t.abs(corr).square(), dim=-1)/self.norm.clamp_min(1e-30)[None]
            score = score.masked_fill(masked, -t.inf)
            group = t.argmax(score, dim=1)
            masked.scatter_(1, group[:, None], True)
            chosen.append(group)
            new = self.a[:, group, :].permute(1, 0, 2).contiguous()
            block = new.mH @ new + ridge[:, None, None]*eye
            if factor is None:
                factor = t.linalg.cholesky((block+block.mH)*.5)
                active_matrix = new
                rhs = new.mH @ target
            else:
                cross = active_matrix.mH @ new
                v = t.linalg.solve_triangular(factor, cross, upper=False)
                schur = block-v.mH@v
                small = t.linalg.cholesky((schur+schur.mH)*.5)
                zeros = t.zeros((batch, factor.shape[-2], self.q), dtype=factor.dtype, device=factor.device)
                factor = t.cat((t.cat((factor, zeros), dim=-1), t.cat((v.mH, small), dim=-1)), dim=-2)
                active_matrix = t.cat((active_matrix, new), dim=-1)
                rhs = t.cat((rhs, new.mH @ target), dim=-2)
            z = t.cholesky_solve(rhs, factor)
            residual = target-active_matrix@z
            if k in checkpoints:
                groups = t.stack(chosen, dim=1)
                active = (self.transforms[groups] @ z.reshape(batch, k, self.q, 1))[..., 0]
                coeff = t.zeros((batch, self.length, self.q), dtype=self.a.dtype, device=self.a.device)
                coeff.scatter_(1, groups[..., None].expand(-1, -1, self.q), active)
                outputs[k] = coeff.transpose(1, 2).contiguous()
        return outputs
