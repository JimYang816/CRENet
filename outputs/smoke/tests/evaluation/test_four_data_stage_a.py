"""Stage-A diagnosis must train only its selected detector parameters."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from src.algorithms.crenet.model import CRENet
from src.evaluation.four_data_stage_a import detector_changes, detector_optimizer, run, soft_ber_loss, train_interval
from src.training.four_data_crenet import run_epoch, selection_key


class SmallPhysicalDataset:
    def __init__(self):
        self.device = torch.device("cpu")
        self.snr_db = np.asarray([10.])
        self.yielded = 0
        generator = torch.Generator().manual_seed(42)
        n, batch = 12, 8
        data = torch.arange(n) % 3 == 0
        pilot = torch.arange(n) % 3 == 1
        bits = torch.randint(0, 2, (batch, int(data.sum()), 2), generator=generator)
        symbols = ((1-2*bits[..., 0]) + 1j*(1-2*bits[..., 1])) / 2**.5
        known = torch.zeros(batch, n, dtype=torch.complex64)
        known[:, pilot] = (1+1j) / 2**.5
        x = known.clone()
        x[:, data] = symbols
        real = torch.randn(batch, n, n, generator=generator)
        imag = torch.randn(batch, n, n, generator=generator)
        channel = torch.eye(n, dtype=torch.complex64)[None] + .04*torch.complex(real, imag)
        noise = .2*torch.complex(torch.randn(batch, n, generator=generator),
                                 torch.randn(batch, n, generator=generator))
        self.inputs = dict(observations=(channel@x[..., None])[..., 0]+noise,
            initial_channel=channel, known=known, data_mask=data, pilot_mask=pilot,
            null_mask=~(data | pilot), carrier_positions=torch.arange(n),
            rf_variance=torch.full((batch,), .08), unit_covariance=torch.eye(n, dtype=torch.complex64))
        self.targets = dict(classes=2*bits[..., 0]+bits[..., 1], bits=bits,
                            reference=channel, snr_indices=torch.zeros(batch, dtype=torch.long))

    def batches(self, batch_size, *, seed=None):
        for _ in range(5):
            self.yielded += 1
            yield self.inputs, self.targets, {}


@pytest.mark.parametrize("rates,changed,frozen", [
    (dict(scalars=1e-3, neural=0.), "scalars", "neural"),
    (dict(scalars=0., neural=1e-3), "neural", "scalars"),
])
def test_training_changes_selected_group_and_keeps_other_group_fixed(rates, changed, frozen):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = CRENet(layers=2, bandwidth=2, window_size=4, center_size=2, hidden_size=4,
                   detector_mode="budgeted", noise_mode="diagonal", initial_soft_damping=.8)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    dataset = SmallPhysicalDataset()
    optimizer = detector_optimizer(model, rates)
    result = train_interval(model, dataset, optimizer, batch_size=8, steps=2, seed=7)
    changes = detector_changes(model, initial)
    assert result["steps"] == 2 and dataset.yielded == 2
    assert result["first_gradient_l2"][changed] > 0
    assert changes[changed]["changed_tensors"] > 0 and changes[changed]["l2"] > 0
    assert changes[frozen]["changed_tensors"] == 0
    assert all(torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters()
               if name.startswith("corrector.") or name == "eta_logits")
    assert 0 <= run_epoch(model, dataset, stage="A", batch_size=8, lambda_h=0.)["ber"] <= 1


def test_diagnosis_compares_frozen_initial_model_without_opening_test(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    options = dict(layers=2, bandwidth=2, window_size=4, center_size=2, hidden_size=4,
                   detector_mode="budgeted", noise_mode="diagonal", initial_soft_damping=.8)
    config = dict(model=options, input_view="reference_assisted", training=dict(seed=9))
    model = CRENet(**options)
    checkpoint = dict(identity="identity", stage="A", epoch=0, model=model.state_dict())
    recorded = run_epoch(model, SmallPhysicalDataset(), stage="A", batch_size=8, lambda_h=0.)
    checkpoint["validation"] = recorded
    provenance = dict(identity="identity", config=config, data=dict(manifest_sha256="manifest"))
    monkeypatch.setattr("src.evaluation.four_data_stage_a._verified_source",
                        lambda *_: (provenance, checkpoint))
    opened = []

    def open_dataset(path, **_):
        opened.append(path.stem)
        dataset = SmallPhysicalDataset()
        dataset.channel = "NOF1"
        dataset.records = np.asarray([1] if path.stem == "train" else [2])
        dataset.targets["reference"] = dataset.targets["reference"]*1.05
        return dataset

    monkeypatch.setattr("src.evaluation.four_data_stage_a.FourDataTrainingSet", open_dataset)
    previous = tmp_path/"previous"
    previous.mkdir()
    (previous/"best_A.pt").write_bytes(b"checkpoint")
    output = tmp_path/"result"
    result = run(config, tmp_path, previous, output, device="cpu", variants=["scalars_1e-3"],
                 intervals=1, steps=2, batch_size=8)
    assert result["complete"] and not result["test_used"]
    assert opened == ["train", "validation"]
    assert result["variants"]["scalars_1e-3"]["intervals"][0]["changes"]["scalars"]["l2"] > 0
    assert (output/"scalars_1e-3.pt").is_file()
    assert (output/"ber_snr.png").is_file() and (output/"nmse_snr.png").is_file()
    with pytest.raises(FileExistsError):
        run(config, tmp_path, previous, output, device="cpu", variants=["scalars_1e-3"],
            intervals=1, steps=2, batch_size=8)


def test_soft_ber_penalizes_wrong_bit_margins_and_has_finite_gradients():
    logits = torch.tensor([[[[4., -1., -1., -1.]], [[3., -1., -1., -1.]]]], requires_grad=True)
    output = SimpleNamespace(log_probabilities=logits.log_softmax(-1))
    mask = torch.tensor([True])
    correct = soft_ber_loss(output, torch.tensor([[0]]), mask)
    wrong = soft_ber_loss(output, torch.tensor([[3]]), mask)
    assert 0 < correct < wrong < 1
    wrong.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_soft_ber_training_updates_detector_parameters():
    torch.set_num_threads(1)
    model = CRENet(layers=2, bandwidth=2, window_size=4, center_size=2, hidden_size=4,
                   detector_mode="budgeted", noise_mode="diagonal", initial_soft_damping=.8)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = detector_optimizer(model, dict(scalars=1e-3, neural=1e-4))
    metrics = train_interval(model, SmallPhysicalDataset(), optimizer, batch_size=8, steps=2,
                             seed=1, objective="softber")
    changes = detector_changes(model, initial)
    assert metrics["objective"] == "softber" and np.isfinite(metrics["loss"])
    assert changes["scalars"]["changed_tensors"] and changes["neural"]["changed_tensors"]


def test_diagnosis_can_fine_tune_detector_after_csi_repair(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    options = dict(layers=2, bandwidth=2, window_size=4, center_size=2, hidden_size=4,
                   detector_mode="budgeted", noise_mode="diagonal", initial_soft_damping=.8)
    config = dict(model=options, input_view="reference_assisted", training=dict(seed=9, lambda_h=.1))
    model = CRENet(**options)
    dataset = SmallPhysicalDataset()
    dataset.targets["reference"] = dataset.targets["reference"]*1.05
    checkpoint = dict(identity="identity", stage="B", epoch=10, model=model.state_dict(),
        validation=run_epoch(model, dataset, stage="B", batch_size=8, lambda_h=.1))
    provenance = dict(identity="identity", config=config, data=dict(manifest_sha256="manifest"))
    monkeypatch.setattr("src.evaluation.four_data_stage_a._verified_source",
                        lambda *_: (provenance, checkpoint))

    def open_dataset(path, **_):
        instance = SmallPhysicalDataset()
        instance.channel = "NOF1"
        instance.records = np.asarray([1] if path.stem == "train" else [2])
        instance.targets["reference"] = instance.targets["reference"]*1.05
        return instance

    monkeypatch.setattr("src.evaluation.four_data_stage_a.FourDataTrainingSet", open_dataset)
    previous = tmp_path/"previous"
    previous.mkdir()
    (previous/"best_B.pt").write_bytes(b"checkpoint")
    result = run(config, tmp_path, previous, tmp_path/"result", device="cpu", start="B",
                 variants=["scalars_1e-3"], intervals=1, steps=2, batch_size=8)
    assert result["complete"] and result["source_checkpoint_stage"] == "B"
    assert result["source_checkpoint_epoch"] == 10 and not result["test_used"]
    variant = result["variants"]["scalars_1e-3"]
    assert variant["intervals"][0]["changes"]["scalars"]["l2"] > 0
    assert variant["beats_baseline"] == (selection_key("C", variant["best_validation"])
                                          < selection_key("C", result["baseline"]))
