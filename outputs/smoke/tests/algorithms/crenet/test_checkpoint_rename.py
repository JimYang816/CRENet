"""A name-only migration must not weaken checkpoint provenance checks."""

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.crenet import CRENet, FEEDBACK_LAYOUT
from src.algorithms.crenet import checkpoint
from src.algorithms.crenet.legacy_identity import (
    LEGACY_INFERENCE_CODE, LEGACY_SCHEMA, RENAMED_INFERENCE_CODE,
)


def test_legacy_checkpoint_requires_both_exact_code_identities_and_view(tmp_path, monkeypatch):
    assert checkpoint.source_fingerprint(checkpoint.INFERENCE_CODE) == RENAMED_INFERENCE_CODE
    model = CRENet(layers=1)
    payload = dict(schema=LEGACY_SCHEMA, inference_code=LEGACY_INFERENCE_CODE.copy(),
                   input_view="compensated", feedback_layout=FEEDBACK_LAYOUT,
                   training_scope="one_channel_all_snr_D1_D2",
                   model_config=model.config, model=model.state_dict())
    path = tmp_path/"old.pt"
    torch.save(payload, path)
    loaded, _ = checkpoint.load_model(path)
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items())

    payload["input_view"] = "raw"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load_model(path)
    payload["input_view"] = "compensated"
    payload["inference_code"] = dict(LEGACY_INFERENCE_CODE, **{"src.common.numerics": "different"})
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load_model(path)

    payload["inference_code"] = LEGACY_INFERENCE_CODE.copy()
    torch.save(payload, path)
    monkeypatch.setattr(checkpoint, "source_fingerprint", lambda _: {"modified": "source"})
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load_model(path)
