"""CSI-view-bound CRENet checkpoints, independent of historical EPNet models."""
from __future__ import annotations

from pathlib import Path

import torch

from src.algorithms.crenet.model import FEEDBACK_LAYOUT, CRENet
from src.algorithms.crenet.legacy_identity import matches_renamed_checkpoint
from src.evaluation.fingerprints import source_fingerprint
from src.data.contract import DEFAULT_INPUT_VIEW

SCHEMA = "watermark-crenet-view-band14-v3"
INPUT_VIEW = DEFAULT_INPUT_VIEW
INFERENCE_CODE = ("src.data.contract", "src.algorithms.crenet.model", "src.algorithms.crenet.operators",
                  "src.algorithms.crenet.lmmse",
                  "src.algorithms.crenet.dpss",
                  "src.algorithms.crenet.checkpoint", "src.data.crenet",
                  "src.evaluation.inputs", "src.common.numerics",
                  "src.data.compact:CompactDataset",
                  "data_generation.src.dataset.compact_math:reconstruct",
                  "data_generation.src.receiver.dpss:DPSSModel")


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    partial.replace(path)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_model(path, device="cpu", *, input_view=INPUT_VIEW):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    current_code = source_fingerprint(INFERENCE_CODE)
    current = payload.get("schema") == SCHEMA and payload.get("inference_code") == current_code
    migrated = matches_renamed_checkpoint(payload, current_code)
    if (not (current or migrated) or payload.get("input_view") != input_view
            or payload.get("feedback_layout") != FEEDBACK_LAYOUT
            or payload.get("training_scope") != "one_channel_all_snr_D1_D2"):
        raise ValueError("CRENet checkpoint input view/feedback layout/code identity mismatch")
    configuration = dict(payload["model_config"])
    if not payload.get("inference_correction_enabled", True):
        configuration["correction_mode"] = "fixed"
        configuration["correction_passes"] = 1
    model = CRENet(**configuration).to(device)
    model.load_state_dict(payload["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite CRENet checkpoint")
    model.eval().requires_grad_(False)
    return model, payload


def validate_dataset(payload, dataset):
    if (getattr(dataset, "input_view", None) != payload.get("input_view")
            or payload["channel"] != dataset.channel
            or payload["model_identity"] != str(dataset._h.attrs["model_identity"])
            or payload["data_schema"] != str(dataset._h.attrs["schema"])
            or payload["snr_db"] != dataset.snr_db.tolist()
            or payload["data_indices"] != dataset._indices.tolist()):
        raise ValueError("CRENet checkpoint belongs to a different data model/view/channel")
