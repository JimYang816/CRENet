"""Frozen deployment preserves learned predictions, including CSI feedback."""
import copy

import pytest
import torch

from src.algorithms.crenet.inference import prepare_frozen_inference
from src.algorithms.crenet.model import CRENet
from outputs.smoke.tests.algorithms.crenet.test_crenet import inputs


@pytest.mark.parametrize("noise_mode", ["diagonal", "full"])
@pytest.mark.parametrize("learned_head", [False, True])
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_preparation_preserves_predictions_on_distinct_inputs(noise_mode, learned_head, precision):
    torch.set_num_threads(2)
    model = CRENet(layers=3, bandwidth=1, detector_mode="budgeted",
                       initial_soft_damping=.8, noise_mode=noise_mode, precision=precision)
    with torch.no_grad():
        for p in model.corrector.parameters():
            p.add_(.01*torch.randn_like(p))
        for layer in model.equalizers:
            layer.csi_loading_log.add_(.2)
            layer.temperature_log.add_(.1)
            if learned_head:
                for p in layer.classifier.parameters():
                    p.add_(.01*torch.randn_like(p))
    model.eval().requires_grad_(False)
    batches = [inputs(size, 2, getattr(torch, precision)) for size in (19, 19, 23)]
    batches[1]["observations"] *= .7
    with torch.inference_mode():
        expected = [model(**batch) for batch in batches]
        prepare_frozen_inference(model)
        prepare_frozen_inference(model)  # The production default remains bitwise equivalent.
        for batch, ref in zip(batches, expected):
            actual = model(**batch)
            for name in ("probabilities", "log_probabilities", "symbols", "variances", "bands"):
                torch.testing.assert_close(getattr(actual, name), getattr(ref, name), rtol=0, atol=0)
            for left, right in zip(actual.diagnostics, ref.diagnostics):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    model.train()
    with pytest.raises(RuntimeError, match="cannot be trained"):
        model(**batches[0])


def test_preparation_rejects_trainable_parameters():
    model = CRENet(detector_mode="budgeted").eval()
    with pytest.raises(ValueError, match="frozen weights"):
        prepare_frozen_inference(model)


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("noise_mode", ["diagonal", "full"])
@pytest.mark.parametrize("cache_static,reuse_residual", [(True, False), (False, True), (True, True)])
def test_optimized_learned_models_preserve_all_states_and_reset_sample_cache(precision, noise_mode,
                                                                          cache_static, reuse_residual):
    torch.set_num_threads(2)
    torch.manual_seed(48)
    model = CRENet(layers=8, bandwidth=2, detector_mode="budgeted", noise_mode=noise_mode,
                   precision=precision, initial_soft_damping=.8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.03*torch.randn_like(parameter))
    model.eval().requires_grad_(False)
    optimized = copy.deepcopy(model)
    original_weights = {name: p.clone() for name, p in optimized.named_parameters()}
    original_parameters = dict(optimized.named_parameters())
    prepare_frozen_inference(optimized, cache_static_correction=cache_static,
                             reuse_correction_residual=reuse_residual)
    prepare_frozen_inference(optimized, cache_static_correction=cache_static,
                             reuse_correction_residual=reuse_residual)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in optimized.parameters())
    tolerance = dict(atol=2e-5, rtol=2e-5) if precision == "complex64" else dict(atol=1e-12, rtol=1e-12)
    with torch.inference_mode():
        for index, (size, batch) in enumerate(((23, 2), (23, 2), (19, 1), (23, 2))):
            item = inputs(size, batch, getattr(torch, precision))
            # Same geometry, different static CSI/noise/pilots must never reuse a previous sample.
            item["initial_channel"] *= .4+index
            item["rf_variance"] *= 1+index
            item["known"] *= 1j**index
            expected, actual = model(**item), optimized(**item)
            for name in ("probabilities", "log_probabilities", "symbols", "variances", "bands"):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name), **tolerance)
            for left, right in zip(actual.diagnostics, expected.diagnostics):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], **tolerance)
        for name, value in original_weights.items():
            torch.testing.assert_close(original_parameters[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("mode,passes,repeats,enabled", [
    ("iterative", 1, 1, True), ("once", 3, 2, True),
    ("fixed", 1, 1, True), ("iterative", 1, 1, False)])
def test_residual_reuse_counts_and_control_paths(monkeypatch, mode, passes, repeats, enabled):
    from src.algorithms.crenet.operators import BandChannel
    torch.set_num_threads(2)
    model = CRENet(layers=3, bandwidth=1, detector_mode="budgeted", noise_mode="diagonal",
                   correction_mode=mode, correction_passes=passes, equalizer_repeats=repeats)
    with torch.no_grad():
        model.corrector.head.weight.add_(.01*torch.randn_like(model.corrector.head.weight))
    model.eval().requires_grad_(False)
    item = inputs()
    count = [0]
    original = BandChannel.matvec
    def counted(self, value):
        count[0] += 1
        return original(self, value)
    monkeypatch.setattr(BandChannel, "matvec", counted)
    with torch.inference_mode():
        expected = model(**item, correction_enabled=enabled)
        before = count[0]
        count[0] = 0
        prepare_frozen_inference(model, cache_static_correction=True, reuse_correction_residual=True)
        actual = model(**item, correction_enabled=enabled)
        savings = 3 if mode == "iterative" and enabled else 1 if mode == "once" and enabled else 0
        assert count[0] == before-savings
        torch.testing.assert_close(actual.symbols, expected.symbols, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual.bands, expected.bands, atol=2e-5, rtol=2e-5)
