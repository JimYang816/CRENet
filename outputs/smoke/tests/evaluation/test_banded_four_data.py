"""Banded LMMSE evaluation uses all four symbols of the formal data reader."""

import numpy as np
import pytest
from data_generation.src.dataset.four_data_protocol import channel_config


@pytest.mark.parametrize("input_view", ["raw", "reference_assisted"])
def test_formal_reader_comparison_scores_all_four_symbols(tmp_path, input_view):
    import json
    import h5py
    from threadpoolctl import threadpool_limits
    from data_generation.src.dataset.four_data_math import (
        reference_statistics,
        sampled_grams,
    )
    from data_generation.src.dataset.four_data_protocol import SCHEMA, MODEL_SCHEMA
    from data_generation.src.transmitter.ofdm import qpsk
    from src.common.numerics import DenseBackend
    from src.evaluation.comparison import evaluate
    from src.evaluation.registry import CURVES

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n, count, ns = 256, 2, 2
    indices = np.arange(16, 208, 4)
    mask = np.zeros(n, bool)
    mask[indices] = True
    c = dict(channel_config("NOF1"), smoke=True, nfft=n, cp=64, data_count=len(indices))
    sampled = np.zeros((4, n, 24))
    sampled[..., 0] = 1
    truth = np.zeros((4, n, n), complex)
    truth[..., 0] = 1
    cross, energy = reference_statistics(sampled, truth, np.arange(192))
    known = np.zeros((4, n), complex)
    known[:, [10, 13]] = 1j
    rng = np.random.default_rng(194)
    bits = rng.integers(0, 2, (count, 4, len(indices), 2), dtype=np.uint8)
    clean = np.broadcast_to(known, (count, 4, n)).copy()
    clean[:, :, indices] = qpsk(bits)
    coefficients = np.zeros((ns, count, 24, 192), np.complex64)
    gains = np.array([0.9 + 0.1j, 1.05 - 0.02j], np.complex64)
    coefficients[:, :, 0, 0] = gains[:, None]
    with h5py.File(data_dir / "model.h5", "w") as m:
        m.attrs.update(
            schema=MODEL_SCHEMA,
            complete=True,
            identity="four-symbol-fixture",
            config_json=json.dumps(c),
        )
        for key, value in dict(
            sampled_basis=sampled,
            delay_indices=np.arange(192),
            known=known,
            unit_rf_noise_covariance=np.eye(n),
            sampled_grams=sampled_grams(sampled),
        ).items():
            m.create_dataset(key, data=value)
        m["masks/data"] = mask
    with h5py.File(data_dir / "test.h5", "w") as h:
        h.attrs.update(
            schema=SCHEMA,
            complete=True,
            raw_complete=True,
            channel="NOF1",
            split="test",
            config_hash="fixture",
            model_file="model.h5",
            model_identity="four-symbol-fixture",
            default_input_view=input_view,
            reference_assisted_csi_adjustment=(input_view == "reference_assisted"),
        )
        h["snr_db"] = [0, 30]
        h["compensation/error_scales"] = np.full(ns, 0.5)
        h["samples/window_index"] = np.arange(count)
        h["samples/raw_coefficients"] = coefficients
        h["samples/bits"] = bits
        h["samples/clean_observations"] = clean
        h["samples/standard_noise"] = np.zeros_like(clean)
        h["samples/rf_noise_variance"] = np.full((ns, count), 0.1)
        h["samples/complete"] = np.ones(count, bool)
        h["windows/complete"] = np.ones(count, bool)
        h["windows/reference_cross"] = np.stack([cross] * count)
        h["windows/reference_energy"] = np.stack([energy] * count)
        if input_view == "reference_assisted":
            h["windows/reference_taps"] = np.stack([truth] * count)
    with threadpool_limits(2):
        report = evaluate(
            data_dir / "test.h5",
            tmp_path / "run",
            DenseBackend(),
            batch_size=2,
            cache_dir=tmp_path / "cache",
            input_view=input_view,
        )
    assert report["complete"] and tuple(report["curves"]) == CURVES
    assert report["data_blocks_per_snr"] == 8
    assert report["bits_per_snr"] == count * 4 * len(indices) * 2
    assert report["curves"][CURVES[0]]["errors"] == [0, 0]
    np.testing.assert_allclose(
        report["nmse"]["mean_nmse_db"],
        10
        * np.log10(
            abs(gains.astype(complex) - 1) ** 2
            * (0.25 if input_view == "reference_assisted" else 1)
        ),
        atol=1e-10,
    )
    if input_view == "reference_assisted":
        from src.evaluation.assisted_banded import finish

        measured = report["nmse"]["mean_nmse_db"]
        targets = dict(
            snr_db=report["snr_db"], decay_db=10,
            targets={"NOF1": dict(start_db=measured[0], end_db=measured[-1])},
        )
        finish(tmp_path / "run", {"NOF1": report}, targets)
        result = json.loads((tmp_path / "run/test_metrics.json").read_text())
        assert list(result["channels"]) == ["NOF1"]
        for name in ("comparison.png", "ber_snr.png", "nmse_snr.png", "summary.csv"):
            assert (tmp_path / "run" / name).stat().st_size > 0
