"""Pre-revision checkpoints cannot be reused on the new raw data."""

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.crenet import CRENet, FEEDBACK_LAYOUT
from src.algorithms.crenet import checkpoint
from src.algorithms.crenet.legacy_identity import (
    LEGACY_INFERENCE_CODE, FORMAL_INFERENCE_CODE, LEGACY_SCHEMA,
)


@pytest.mark.parametrize("historical_code", [LEGACY_INFERENCE_CODE, FORMAL_INFERENCE_CODE])
def test_legacy_checkpoint_is_rejected_for_both_views(tmp_path, historical_code):
    model = CRENet(layers=1)
    payload = dict(schema=LEGACY_SCHEMA, inference_code=historical_code.copy(),
                   input_view="compensated", feedback_layout=FEEDBACK_LAYOUT,
                   training_scope="one_channel_all_snr_D1_D2",
                   model_config=model.config, model=model.state_dict())
    path = tmp_path/"old.pt"
    torch.save(payload, path)
    for view in ("compensated", "raw"):
        payload["input_view"] = view
        torch.save(payload, path)
        with pytest.raises(ValueError, match="identity mismatch"):
            checkpoint.load_model(path)
