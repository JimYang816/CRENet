"""Check that smoke training cannot accidentally use the formal output root."""

import importlib
import json
from pathlib import Path
import sys

import pytest

from src.evaluation.result_cache import DEFAULT_CACHE
from src.evaluation.fingerprints import ROOT
from src.common.outputs import resolve_output_root


@pytest.mark.parametrize(
    ("module_name", "run_name", "suffix", "workflow"),
    [
        ("scripts.train.crenet", "crenet_dddd_raw", "joint/nof1", False),
        ("scripts.run.crenet", "crenet_dddd_raw", "", True),
        ("scripts.train.oamp_net2", "oamp_net2_dddd_raw", "nof1", False),
        ("scripts.train.udnet", "udnet_dddd_raw", "nof1", False),
    ],
)
@pytest.mark.parametrize(("smoke", "explicit"), [(False, False), (True, False), (True, True)])
def test_training_cli_routes_smoke_to_separate_directory(monkeypatch, tmp_path, module_name, run_name, suffix, workflow, smoke, explicit):
    module = importlib.import_module(module_name)
    seen = []

    def capture(*args, **kwargs):
        seen.append(Path(kwargs["output"] if workflow else args[1]))

    monkeypatch.setattr(module, "run_channel" if workflow else "train", capture)
    monkeypatch.setattr(module.torch, "set_num_threads", lambda _: None)
    argv = [module_name, "--channel", "NOF1", "--device", "cpu"]
    if smoke:
        argv += ["--max-frames", "2"]
    if explicit:
        argv += ["--output", str(tmp_path / "chosen")]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    root = tmp_path / "chosen" if explicit else Path("outputs") / ("smoke" if smoke else "experiment") / run_name
    assert seen == [root / suffix]


def test_main_evaluation_configuration_uses_formal_results():
    config = json.loads(Path("configs/evaluation/ber_comparison.json").read_text(encoding="utf-8"))
    formal = Path("outputs/experiment")
    assert Path(config["output"]).is_relative_to(formal)
    for channel in config["checkpoints"].values():
        assert all(Path(path).is_relative_to(formal) for path in channel.values())
    assert DEFAULT_CACHE == ROOT / formal / "_cache"


def test_smoke_output_cannot_escape_through_configured_relative_path():
    output = resolve_output_root("outputs/experiment/../../outside", smoke=True)
    assert output == Path("outputs/smoke/outside")


@pytest.mark.parametrize(
    ("module_name", "config_path", "run_name", "comparison"),
    [
        ("scripts.evaluate.compare", "configs/evaluation/ber_comparison.json", "equalizer_comparison_dddd_raw", True),
        ("scripts.evaluate.udnet", "configs/training/udnet.json", "udnet_dddd_raw", False),
        ("scripts.evaluate.oamp_net2", "configs/training/oamp_net2.json", "oamp_net2_dddd_raw", False),
    ],
)
def test_evaluation_of_smoke_data_keeps_reports_and_cache_separate(monkeypatch, tmp_path, module_name, config_path, run_name, comparison):
    module = importlib.import_module(module_name)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    dataset = Path("outputs/smoke/example/data")
    config["datasets"] = {"NOF1": str(dataset / "test.h5" if comparison else dataset)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    seen = []

    def capture(*args, **kwargs):
        seen.append((Path(args[1] if comparison else args[2]), Path(kwargs["cache_dir"])))

    monkeypatch.setattr(module, "evaluate", capture)
    if hasattr(module, "torch"):
        monkeypatch.setattr(module.torch, "set_num_threads", lambda _: None)
    argv = [module_name, "--config", str(config_path), "--channel", "NOF1", "--device", "cpu"]
    if comparison:
        argv.append("--baselines-only")
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    run = Path("outputs/smoke") / run_name / "nof1"
    assert seen == [(run if comparison else run / "evaluation", Path("outputs/smoke/_cache"))]
