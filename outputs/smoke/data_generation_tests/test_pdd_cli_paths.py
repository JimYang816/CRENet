"""Exercise the data-generation CLI's smoke output boundary."""

from pathlib import Path

from data_generation.scripts import pdd
from data_generation.src.channel.watermark import replay
from data_generation.src.dataset import pdd_calibration
from data_generation.src.receiver import chain


def test_smoke_calibration_and_run_stay_under_smoke(monkeypatch, tmp_path):
    calls = []

    class Archive:
        def __init__(self, *args):
            pass

        def audit(self, records):
            return {"records": {}}

    class Receiver:
        def __init__(self, config):
            pass

        def validate_noise(self):
            return {}

    def calibrate(calibration, run, *args):
        calls.append((Path(calibration), Path(run)))

    monkeypatch.setattr(pdd, "ROOT", tmp_path)
    monkeypatch.setattr(pdd, "identities", lambda *args: [])
    monkeypatch.setattr(pdd, "physical_identities", lambda *args: [])
    monkeypatch.setattr(pdd.subprocess, "check_output", lambda *args, **kwargs: "test-commit")
    monkeypatch.setattr(replay, "WatermarkArchive", Archive)
    monkeypatch.setattr(chain, "ReceiverChain", Receiver)
    monkeypatch.setattr(pdd_calibration, "calibrate", calibrate)

    pdd.main(["calibrate", "--smoke", "--device", "cpu", "--run-id", "layout"])

    run = tmp_path / "outputs" / "smoke" / "layout_smoke"
    assert calls == [(run / "calibration.h5", run)]
    assert (run / "run_identity.json").is_file()
    assert not (tmp_path / "data" / "interim").exists()
    assert not (tmp_path / "data" / "processed").exists()
