"""Bounded-memory replay and full physical reference kernels for WATERMARK."""

import numpy as np
from scipy import sparse

from data_generation.src.receiver.chain import ReceiverChain, resample
from data_generation.src.transmitter.ofdm import window_indices


class BlockedReceiver:
    def __init__(self, c, block=256):
        self.chain = ReceiverChain(c)
        self.block = block
        self.ui_conj = self.chain.ui.conjugate()

    def native_rows(self, sounding, origin, indices):
        taps = sounding.interpolate(origin + indices / sounding.fs_delay)
        cols = indices[:, None] - np.arange(taps.shape[1])
        valid = cols >= 0
        counts = np.minimum(indices + 1, taps.shape[1])
        indptr = np.r_[0, np.cumsum(counts)]
        return sparse.csr_matrix(
            (taps[:, ::-1][valid[:, ::-1]], cols[:, ::-1][valid[:, ::-1]], indptr),
            shape=(len(indices), self.chain.u.shape[0]),
        )

    def reference_rows(self, sounding, origin, indices):
        out = []
        for begin in range(0, len(indices), self.block):
            selected = indices[begin : begin + self.block]
            forward, image = self.chain.r[selected], self.chain.ri[selected]
            support = np.union1d(forward.indices, image.indices)
            native = self.native_rows(sounding, origin, support)
            direct = (forward[:, support] @ native) @ self.chain.u
            mixed = (image[:, support] @ native.conjugate()) @ self.ui_conj
            block = (0.5 * (direct + mixed)).tocsr()
            block.eliminate_zeros()
            out.append(block)
        return sparse.vstack(out, format="csr")

    def reference(self, sounding, start, symbol):
        c, n = self.chain.c, self.chain.n
        origin = start - c["guard_samples"] / c["fs_baseband"]
        op = self.reference_rows(sounding, origin, window_indices(c, symbol)).tocoo()
        offset = op.col - (c["guard_samples"] + symbol * (n + c["cp"]))
        inside = (offset >= 0) & (offset < n + c["cp"])
        rows = op.row[inside]
        delays = (rows - (offset[inside] - c["cp"])) % n
        taps = np.zeros((n, n), complex)
        np.add.at(taps, (rows, delays), op.data[inside])
        leak = float(
            np.sum(abs(op.data[~inside]) ** 2) / max(np.sum(abs(op.data) ** 2), 1e-300)
        )
        return taps, leak

    def replay(self, sounding, origin, frame):
        c, chain = self.chain.c, self.chain
        tx = chain.transmit(frame.baseband)
        native = resample(tx * chain.phase_carrier.conj(), 1, 6, c)
        length = sounding.h.shape[1]
        delayed = np.lib.stride_tricks.sliding_window_view(
            np.pad(native, (length - 1, 0)), length
        )[:, ::-1]
        received = np.empty_like(native)
        for begin in range(0, len(native), self.block):
            end = min(begin + self.block, len(native))
            impulse = sounding.interpolate(
                origin + np.arange(begin, end) / sounding.fs_delay
            )
            received[begin:end] = np.einsum("ij,ij->i", impulse, delayed[begin:end])
        return np.real(resample(received, 6, 1, c)[: len(tx)] * chain.phase_carrier)
