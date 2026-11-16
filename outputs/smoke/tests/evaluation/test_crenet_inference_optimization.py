"""Exercise paired scoring, provenance and timing on independent tiny data."""
import torch
from threadpoolctl import threadpool_limits

from outputs.smoke.data_generation_tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.algorithms.crenet.checkpoint import SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE, atomic_save, cpu_state
from src.algorithms.crenet.model import CRENet
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.fingerprints import source_fingerprint
from src.evaluation.crenet_inference_optimization import run_channel


def test_paired_pipeline_preserves_checkpoint_and_writes_both_curves(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(20261116)
    with threadpool_limits(1):
        config, prior, chain, archive = fixture()
        directory, generation = tmp_path/"data", tmp_path/"generation"
        generation.mkdir()
        generate(directory, generation, config, archive, chain, prior, DenseBackend(),
                 dict(commit="test", archive_sha256="test"))
        model = CRENet(layers=8, bandwidth=2, window_size=6, center_size=3, hidden_size=4,
                       detector_mode="budgeted", noise_mode="diagonal", initial_soft_damping=.8)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.01*torch.randn_like(parameter))
        checkpoint = tmp_path/"model.pt"
        with CompactDataset(directory/"train.h5") as dataset:
            atomic_save(checkpoint, dict(schema=SCHEMA, input_view="compensated", model_config=model.config,
                feedback_layout=FEEDBACK_LAYOUT, inference_code=source_fingerprint(INFERENCE_CODE),
                model=cpu_state(model), training_scope="one_channel_all_snr_D1_D2", channel=dataset.channel,
                data_schema=str(dataset._h.attrs["schema"]), model_identity=str(dataset._h.attrs["model_identity"]),
                data_indices=dataset._indices.tolist(), snr_db=dataset.snr_db.tolist()))
        original = checkpoint.read_bytes()
        result = run_channel(directory, checkpoint, tmp_path/"report", validation_frames=1, test_frames=1, repeats=1)
        assert result["complete"] and result["validation_allclose"] and result["test_allclose"]
        assert checkpoint.read_bytes() == original
        assert result["learned_weights_unchanged"]
        assert result["final_decision_disagreements"] == 0
        assert result["mean_ber"]["before"] == result["mean_ber"]["both"]
        for split in ("validation", "test"):
            for name in ("ber_snr.png", "nmse_snr.png", "metrics.json"):
                assert (tmp_path/"report"/split/name).is_file()
        assert set(result["timing"]["forward"]) == {"before", "both", "static_only", "residual_only", "legacy_tf32"}
