"""Bounded CPU physical-window workers for four-data-symbol datasets."""
from __future__ import annotations

from collections import deque
import time

import numpy as np
from threadpoolctl import threadpool_limits

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.dataset.four_data_math import array_sha256
from data_generation.src.dataset.four_data_protocol import SPLIT_CODES
from data_generation.src.receiver.chain import relative_energy
from data_generation.src.receiver.estimation import input_windows
from data_generation.src.transmitter.ofdm import fft, make_frame, receive_symbols
from data_generation.src.receiver.blocked import BlockedReceiver

_CONTEXT = None


def initialize_worker(c, archive_path, threads):
    global _CONTEXT
    limits = threadpool_limits(threads)
    _CONTEXT = (c, WatermarkArchive(archive_path, c["channel"]), BlockedReceiver(c), limits)


def compute_window(c, archive, receiver, job):
    wi, split, record, start, repeats = job
    begin = time.perf_counter()
    sounding = archive.load(record)
    origin = start-c["guard_samples"]/c["fs_baseband"]
    duration = (4*(c["nfft"]+c["cp"])+2*c["guard_samples"])/c["fs_baseband"]
    if origin < 0 or origin+duration >= sounding.duration:
        raise ValueError(f"Physical window crosses recording boundary: {record}@{start}")
    refs = [receiver.reference(sounding, start, d) for d in range(4)]
    # Persisted reference precision defines reconstruction, residuals and scoring.
    truth = np.stack([value[0] for value in refs]).astype(np.complex64)
    samples = []
    for repeat in range(repeats):
        frame = make_frame(c, record, start, SPLIT_CODES[split], payload_repeat=repeat)
        clean_rf = receiver.replay(sounding, origin, frame)
        clean = receive_symbols(receiver.chain.receive(clean_rf), c)
        seed = (c["seed"], SPLIT_CODES[split], record, round(start*1000), 1, repeat)
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        standard = receive_symbols(receiver.chain.receive(rng.standard_normal(len(clean_rf))), c)
        variance = np.mean(abs(clean)**2)/receiver.chain.noise_gain/10**(np.asarray(c["snr_db"])/10)
        prediction = fft(np.sum(truth*input_windows(frame.x, c["nfft"]), axis=-1))
        residual = clean-prediction
        samples.append(dict(repeat=repeat, clean=clean, standard=standard, bits=frame.bits,
            variance=variance, payload_seed=frame.seed, noise_seed=seed,
            physical_sha256=array_sha256(clean, standard, frame.bits, variance),
            model_residual=residual, model_error=[relative_energy(residual[d], clean[d]) for d in range(4)]))
    # The persisted precision is used consistently for all on-disk NMSE and assistance.
    return wi, dict(reference=truth, reference_sha256=array_sha256(truth), samples=samples,
        diagnostics=dict(intersymbol_leakage=[item[1] for item in refs], sounding_sha256=sounding.sha256,
            model_mismatch_policy=c["model_mismatch_policy"], seconds=time.perf_counter()-begin))


def _worker(job):
    c, archive, receiver, limits = _CONTEXT
    return compute_window(c, archive, receiver, job)


def window_results(jobs, c, archive, receiver, *, pool=None, prefetch=1):
    if pool is None:
        for job in jobs:
            yield compute_window(c, archive, receiver, job)
        return
    source, pending = iter(jobs), deque()
    for _ in range(prefetch):
        job = next(source, None)
        if job is not None:
            pending.append(pool.submit(_worker, job))
    try:
        while pending:
            result = pending.popleft().result()
            job = next(source, None)
            if job is not None:
                pending.append(pool.submit(_worker, job))
            yield result
    finally:
        for future in pending:
            future.cancel()
