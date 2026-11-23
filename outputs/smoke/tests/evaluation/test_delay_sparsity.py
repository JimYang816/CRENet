import numpy as np
import pytest

from data_generation.src.receiver.chain import taps_to_matrix
from src.evaluation.delay_sparsity import TapEnergy, energy_counts, run, scope_metrics


def test_energy_counts_uses_the_smallest_number_and_exact_threshold():
    counts, curve, order = energy_counts([2.0, 0.0, 7.0, 1.0])
    assert counts == {"k90": 2, "k95": 3, "k99": 3}
    assert order.tolist() == [2, 0, 3, 1]
    assert curve[-1] == 1.0
    with pytest.raises(ValueError):
        energy_counts([0.0, 0.0])


def test_reference_matrix_and_dpss_group_energy_round_trip():
    n, length = 8, 4
    basis = np.ones((2, n, 1))
    coefficients = np.array([[3, 2j, 1, 0]], dtype=np.complex128)
    projected = np.einsum("dnq,ql->dnl", basis, coefficients)
    reference = np.stack([taps_to_matrix(x) for x in projected]).astype(np.complex64)
    true, pred, cross, error = TapEnergy(basis).calculate(reference, coefficients)
    np.testing.assert_allclose(true[:, :length], pred, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(true[:, length:], 0, atol=1e-12)
    np.testing.assert_allclose(cross, pred, rtol=1e-7, atol=1e-7)
    metrics, _, _ = scope_metrics(true.sum(axis=0), pred.sum(axis=0), cross.sum(axis=0),
                                  error.sum(), length, block_size=2)
    assert [metrics[f"reference_k{x}"] for x in (90, 95, 99)] == [2, 3, 3]
    assert [metrics[f"projected_k{x}"] for x in (90, 95, 99)] == [2, 3, 3]
    assert metrics["projection_nmse"] < 1e-12
    assert metrics["projected_top_k95_nmse"] < 1e-7
    assert metrics["reference_block2_k95"] == 2


def test_outside_support_and_projection_error_are_not_hidden():
    n, length = 8, 4
    basis = np.ones((2, n, 1))
    coefficients = np.array([[3, 2j, 1, 0]], dtype=np.complex128)
    full = np.zeros((2, n, n), dtype=np.complex128)
    full[:, :, :length] = np.einsum("dnq,ql->dnl", basis, coefficients)
    full[:, :, 6] = 2.0
    reference = np.stack([taps_to_matrix(x) for x in full]).astype(np.complex64)
    true, pred, cross, error = TapEnergy(basis).calculate(reference, coefficients)
    metrics, _, _ = scope_metrics(true.sum(axis=0), pred.sum(axis=0), cross.sum(axis=0),
                                  error.sum(), length, block_size=2)
    assert metrics["reference_k95"] == 3
    assert metrics["outside_support_fraction"] == pytest.approx(4 / 18, rel=1e-7)
    assert metrics["projection_nmse"] == pytest.approx(4 / 18, rel=1e-7)
    assert metrics["projected_top_k95_nmse"] == pytest.approx(4 / 18, rel=1e-7)


def test_joint_support_pools_two_blocks_instead_of_averaging_counts():
    true1 = np.array([9, 1, 0, 0], float)
    true2 = np.array([0, 0, 9, 1], float)
    metrics, _, _ = scope_metrics(true1 + true2, true1 + true2, true1 + true2,
                                  0, length=4, block_size=2)
    assert metrics["reference_k90"] == 2
    assert metrics["reference_k95"] == 3


def test_compact_dataset_analysis_counts_physical_windows_once(tmp_path):
    from outputs.smoke.data_generation_tests.test_compact import fixture
    from data_generation.src.dataset.compact_export import generate
    from src.common.numerics import DenseBackend

    folders = {}
    for channel in ("NOF1", "NCS1"):
        config, prior, chain, archive = fixture()
        config["channel"] = channel
        data, log_dir = tmp_path / channel, tmp_path / (channel + "_log")
        log_dir.mkdir()
        generate(data, log_dir, config, archive, chain, prior, DenseBackend(), {"commit": "test"})
        (data / "manifest.json").write_text("{}", encoding="utf-8")
        folders[channel] = data
    report = run(folders, tmp_path / "analysis", "cpu")
    for channel in folders:
        for split in ("train", "validation", "test"):
            assert report["datasets"][channel][split]["window_count"] == 1
            assert report["statistics"][channel][split]["joint"]["reference_k95"]["n"] == 1
    assert (tmp_path / "analysis/per_window.csv").is_file()
    assert (tmp_path / "analysis/test_delay_sparsity.png").is_file()
