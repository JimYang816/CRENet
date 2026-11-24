"""Lazy DDDD inputs with explicit raw and reference-assisted CSI views."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from data_generation.src.dataset.four_data_math import assisted_taps, reconstruct_taps
from data_generation.src.dataset.four_data_protocol import MODEL_SCHEMA, SCHEMA
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.transmitter.ofdm import qpsk


def data_system(inputs):
    channel, known, observed = inputs["csi"], inputs["known"], inputs["observations"]
    return observed-np.einsum("dij,dj->di", channel, known, optimize=True), channel[..., inputs["data_indices"]]


class FourDataDataset:
    """Read one four-symbol frame at one SNR; open one instance per worker."""

    def __init__(self, path, *, input_view=None):
        self.path = Path(path)
        self._h = h5py.File(self.path, "r")
        self._model = None
        try:
            h = self._h
            if h.attrs.get("schema") != SCHEMA or not h.attrs.get("complete") or not h.attrs.get("raw_complete"):
                raise ValueError("Four-data dataset identity is invalid or incomplete")
            if h.attrs.get("model_file") != "model.h5":
                raise ValueError("Invalid shared model path")
            self._model = h5py.File(self.path.parent/"model.h5", "r")
            m = self._model
            if (m.attrs.get("schema") != MODEL_SCHEMA or not m.attrs.get("complete")
                    or h.attrs.get("model_identity") != m.attrs.get("identity")):
                raise ValueError("Four-data shared model identity mismatch")
            self.input_view = input_view or h.attrs["default_input_view"]
            if self.input_view not in ("raw", "reference_assisted"):
                raise ValueError("Unknown four-data CSI input view")
            if self.input_view == "reference_assisted" and not h.attrs.get("reference_assisted_csi_adjustment"):
                raise ValueError("Reference-assisted view was not generated")
            self.config = json.loads(m.attrs["config_json"])
            self.channel, self.split = h.attrs["channel"], h.attrs["split"]
            self.snr_db = h["snr_db"][:]
            self.delays = m["delay_indices"][:]
            self.sampled_basis = m["sampled_basis"][:]
            self.known = m["known"][:]
            self.masks = {key: value[:] for key, value in m["masks"].items()}
            ns, nf = len(self.snr_db), self.frame_count
            if (h["samples/raw_coefficients"].shape != (ns, nf, 24, 192)
                    or h["samples/bits"].shape != (nf, 4, self.config["data_count"], 2)
                    or self.config.get("frame_structure") != "DDDD" or self.config["delay_length"] != 192
                    or self.delays.shape != (192,) or np.any(np.diff(self.delays) <= 0)
                    or self.delays[0] < 0 or self.delays[-1] >= self.config["nfft"]
                    or self.sampled_basis.shape != (4, self.config["nfft"], 24)
                    or h["compensation/error_scales"].shape != (ns,)
                    or not np.all(h["samples/complete"][:]) or not np.all(h["windows/complete"][:])):
                raise ValueError("Four-data stored dimensions or completion markers are invalid")
        except BaseException:
            self.close()
            raise

    @property
    def frame_count(self):
        return len(self._h["samples/window_index"])

    def __len__(self):
        return len(self.snr_db)*self.frame_count

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        si, fi = divmod(index, self.frame_count)
        return self.model_input(si, fi), self.labels(fi)

    def _check(self, si, fi):
        if not 0 <= si < len(self.snr_db) or not 0 <= fi < self.frame_count:
            raise IndexError((si, fi))

    def observations(self, si, fi):
        self._check(si, fi)
        h = self._h
        return h["samples/clean_observations"][fi]+np.sqrt(h["samples/rf_noise_variance"][si, fi])*h["samples/standard_noise"][fi]

    def raw_taps(self, si, fi):
        self._check(si, fi)
        return reconstruct_taps(self.sampled_basis, self._h["samples/raw_coefficients"][si, fi], self.delays)

    def reference_taps(self, fi):
        if not 0 <= fi < self.frame_count:
            raise IndexError(fi)
        wi = int(self._h["samples/window_index"][fi])
        return self._h["windows/reference_taps"][wi].astype(np.complex128)

    def csi_taps(self, si, fi):
        raw = self.raw_taps(si, fi)
        if self.input_view == "raw":
            return raw
        return assisted_taps(raw, self.reference_taps(fi), self._h["compensation/error_scales"][si])

    def raw_model_input(self, si, fi):
        return self._input(si, fi, self.raw_taps(si, fi), "raw")

    def model_input(self, si, fi):
        return self._input(si, fi, self.csi_taps(si, fi), self.input_view)

    def _input(self, si, fi, taps, view):
        return dict(input_view=view, observations=self.observations(si, fi),
            csi=np.stack([taps_to_matrix(value) for value in taps]), known=self.known.copy(),
            masks={key: value.copy() for key, value in self.masks.items()},
            noise=dict(rf_variance=float(self._h["samples/rf_noise_variance"][si, fi]),
                       unit_rx_covariance=self._model["unit_rf_noise_covariance"][:]),
            data_indices=np.flatnonzero(self.masks["data"]), pilot_indices=np.flatnonzero(self.masks["pilot"]),
            carrier_positions=np.arange(self.config["nfft"]))

    def labels(self, fi):
        if not 0 <= fi < self.frame_count:
            raise IndexError(fi)
        bits = self._h["samples/bits"][fi]
        return dict(bits=bits, symbols=qpsk(bits), classes=(2*bits[..., 0]+bits[..., 1]).astype(np.uint8),
            constellation=qpsk(np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.uint8)))

    def reference_csi(self, fi):
        return np.stack([taps_to_matrix(value) for value in self.reference_taps(fi)])

    def identity(self, fi):
        wi = int(self._h["samples/window_index"][fi])
        return dict(channel=self.channel, split=self.split, record=int(self._h["windows/record"][wi]),
            start_seconds=float(self._h["windows/start_seconds"][wi]), repeat=int(self._h["samples/repeat"][fi]))

    def close(self):
        if self._model is not None:
            self._model.close()
        self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
