"""Four-symbol batches with exact physical delays and separate supervision."""
from __future__ import annotations

import time

import numpy as np
import torch

from src.data.four_data import FourDataDataset


def taps_to_matrices(taps, delays=None):
    """Unitary transform of circular time operators, in fftshift carrier order."""
    n, length = taps.shape[-2:]
    if delays is None:
        delays = torch.arange(length, device=taps.device)
    rows = torch.arange(n, device=taps.device)[:, None]
    columns = (rows-delays) % n
    matrix = taps.new_zeros(*taps.shape[:-2], n, n)
    matrix.scatter_(-1, columns.expand(*taps.shape[:-2], n, length), taps)
    matrix = torch.fft.fft(matrix, dim=-2, norm="ortho")
    matrix = torch.fft.ifft(matrix, dim=-1, norm="ortho")
    return torch.fft.fftshift(matrix, dim=(-2, -1))


class FourDataTrainingSet:
    """Cache one split's reference matrices in RAM, never on disk or in a model.

    Training caches complex64 matrices; final baseline scoring caches complex128.
    All transforms from persisted taps/coefficients are evaluated in complex128.
    """

    def __init__(self, path, *, device="cpu", input_view="reference_assisted",
                 frame_limit=None, reference_precision="complex64"):
        self.device = torch.device(device)
        self.precision = getattr(torch, reference_precision)
        with FourDataDataset(path, input_view=input_view) as ds:
            self.channel, self.split, self.input_view = ds.channel, ds.split, ds.input_view
            self.config, self.snr_db = ds.config, ds.snr_db
            self.frame_count = min(ds.frame_count, frame_limit or ds.frame_count)
            self.full_frame_count = ds.frame_count
            self.identity = dict(model_identity=str(ds._h.attrs["model_identity"]),
                                 schema=str(ds._h.attrs["schema"]), split=ds.split,
                                 frame_count=self.frame_count, input_view=ds.input_view,
                                 calibration_identity=str(ds._h.attrs.get("calibration_identity", "")))
            self.n = ds.config["nfft"]
            self.basis = torch.as_tensor(ds.sampled_basis, device=device, dtype=torch.complex128)
            self.delays = torch.as_tensor(ds.delays, device=device, dtype=torch.long)
            self.masks = {k: torch.as_tensor(v, device=device, dtype=torch.bool) for k,v in ds.masks.items()}
            self.known = torch.as_tensor(ds.known, device=device, dtype=self.precision)
            self.covariance = torch.as_tensor(ds._model["unit_rf_noise_covariance"][:],
                                              device=device, dtype=self.precision)
            self.alpha = torch.as_tensor(ds._h["compensation/error_scales"][:], device=device)
            h, count = ds._h, self.frame_count
            self.window_index = h["samples/window_index"][:count]
            windows = sorted(set(self.window_index.tolist()))
            remap = {w:i for i,w in enumerate(windows)}
            self.window_index = np.asarray([remap[w] for w in self.window_index])
            self.records = h["windows/record"][:]
            self.clean = torch.as_tensor(h["samples/clean_observations"][:count])
            self.standard = torch.as_tensor(h["samples/standard_noise"][:count])
            self.variance = torch.as_tensor(h["samples/rf_noise_variance"][:, :count])
            self.coefficients = torch.as_tensor(h["samples/raw_coefficients"][:, :count])
            self.bits = torch.as_tensor(h["samples/bits"][:count].astype(np.int64))
            self.frame_identities = [ds.identity(i) for i in range(count)]
            self.expected_nmse = h["compensation/assisted_mean_nmse_linear"][:]
            self.reference = torch.empty(len(windows), 4, self.n, self.n, dtype=self.precision)
            begin = time.monotonic()
            for j, window in enumerate(windows):
                taps = torch.as_tensor(h["windows/reference_taps"][window], device=device,
                                       dtype=torch.complex128)
                self.reference[j] = taps_to_matrices(taps).to(dtype=self.precision, device="cpu")
                if j == 0 or (j+1) % 100 == 0 or j+1 == len(windows):
                    print(f"{self.channel} {self.split}: reference cache {j+1}/{len(windows)} "
                          f"({time.monotonic()-begin:.1f}s)", flush=True)

    def batches(self, batch_size, *, seed=None, snr_indices=None):
        selected = list(range(len(self.snr_db))) if snr_indices is None else list(snr_indices)
        if batch_size < 1 or not selected or any(not 0 <= i < len(self.snr_db) for i in selected):
            raise ValueError("Invalid four-data batch schedule")
        grid = np.stack(np.meshgrid(selected, np.arange(self.frame_count), np.arange(4),
                                    indexing="ij"), -1).reshape(-1, 3)
        if seed is not None:
            np.random.default_rng(seed).shuffle(grid)
        device = self.device
        shared = dict(data_mask=self.masks["data"], pilot_mask=self.masks["pilot"],
                      null_mask=~(self.masks["data"] | self.masks["pilot"]),
                      carrier_positions=torch.arange(self.n, device=device),
                      unit_covariance=self.covariance.to(torch.complex64))
        for start in range(0, len(grid), batch_size):
            keys = grid[start:start+batch_size]
            si, fi, di = (torch.as_tensor(keys[:,j].copy(), dtype=torch.long) for j in range(3))
            sd, dd = si.to(device), di.to(device)
            wi = torch.as_tensor(self.window_index[keys[:,1]])
            reference = self.reference[wi, di].to(device)
            coefficient = self.coefficients[si, fi].to(device=device, dtype=torch.complex128)
            taps = torch.einsum("bnq,bql->bnl", self.basis[dd], coefficient)
            raw = taps_to_matrices(taps, self.delays).to(self.precision)
            alpha = self.alpha[sd,None,None].to(reference.real.dtype)
            channel = raw if self.input_view == "raw" else reference+alpha*(raw-reference)
            variance = self.variance[si, fi].to(device)
            observation = (self.clean[fi, di].to(device)
                           + variance.sqrt()[:,None]*self.standard[fi, di].to(device))
            known = self.known[dd]
            inputs = dict(shared, observations=observation.to(torch.complex64),
                          initial_channel=channel.to(torch.complex64), known=known.to(torch.complex64),
                          rf_variance=variance.float())
            bits = self.bits[fi, di].to(device)
            targets = dict(reference=reference.to(torch.complex64), bits=bits,
                           classes=2*bits[...,0]+bits[...,1], snr_indices=sd)
            # Baseline-only double tensors stay outside the forward input whitelist.
            scoring = dict(channel=channel, reference=reference, observations=observation,
                           known=known, rf_variance=variance, keys=keys)
            yield inputs, targets, scoring
