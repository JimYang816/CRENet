"""Historical receiver costs on common, explicitly recorded receiver inputs.

The legacy source runs in separate interpreters. Checkpoint source validation is
retained, but historical dataset validation is deliberately not invoked: current
DDDD inputs are used only to measure cost, never BER or checkpoint quality.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "single_tap_mmse": "Single-tap LMMSE",
    "full_matrix_lmmse": "Full-matrix LMMSE",
    "mmse_dfe": "MMSE-DFE",
    "dd_ce_lmmse": "DD-DPSS",
    "udnet": "UDNet",
    "oamp_net2": "OAMP-Net2",
    "crenet": "CRENet",
}
COMPLEXITY = {
    "single_tap_mmse": "O(D)",
    "full_matrix_lmmse": "O(F)",
    "mmse_dfe": "O(F + D^2)",
    "dd_ce_lmmse": "O(F + N^2*r + N*r^2 + r^3 + N^2*log(N))",
    "udnet": "O(N^2 + W_u*(w^3 + T_u*(w^2 + w*h)))",
    "oamp_net2": "O(F + T_o*D^2)",
    "crenet": "O(T_c*(N^2 + N*b*c + N*c^2 + W_c*w*h))",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_source(run, commit):
    archive = run / "historical_source.zip"
    source = run / "historical_source"
    subprocess.run(["git", "archive", "--format=zip", "--output=" + str(archive), commit], cwd=ROOT, check=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            target = (source / member).resolve()
            if not target.is_relative_to(source.resolve()):
                raise ValueError("Unsafe historical source archive")
            if not member.endswith("/") and target.exists() and target.read_bytes() != bundle.read(member):
                raise ValueError("Historical source changed: " + str(target))
        bundle.extractall(source)
    return source


def sample_keys(snr_count, frame_count, samples_per_snr):
    if not 1 <= samples_per_snr <= frame_count:
        raise ValueError("Invalid number of timing samples")
    frames = np.linspace(0, frame_count - 1, samples_per_snr, dtype=int)
    return [(si, int(fi), j % 4) for si in range(snr_count) for j, fi in enumerate(frames)]


def prepare_inputs(path, configuration, smoke):
    import h5py

    manifest = {}
    with h5py.File(path, "w") as destination:
        for name in ("NOF1", "NCS1"):
            group = destination.create_group(name)
            if smoke:
                n = 64
                rng = np.random.default_rng(configuration["seed"])
                data = np.arange(n) % 2 == 0
                pilot = np.arange(n) % 4 == 1
                known = np.where(pilot, (1 + 1j) / np.sqrt(2), 0)[None].repeat(4, 0)
                h = np.eye(n) + .01 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
                group["channel"] = h[None]
                group["observations"] = rng.normal(size=(1, n)) + 1j * rng.normal(size=(1, n))
                group["variances"] = [.1]
                group["symbol_indices"] = [0]
                group["covariance"] = np.eye(n, dtype=complex)
                group["basis"] = np.linalg.qr(rng.normal(size=(n, 4)))[0][None].repeat(4, 0)
                group["delays"] = np.arange(16)
                prior = group.create_group("prior/000")
                prior.attrs.update(start=0, stop=16)
                prior["temporal"], prior["delay"] = np.eye(4), np.eye(16)
                metadata = dict(synthetic=True, keys=[[0, 0, 0]], snr_db=[10.])
            else:
                from src.data.four_data import FourDataDataset

                dataset_path = ROOT / configuration["datasets"][name]
                with FourDataDataset(dataset_path, input_view=configuration["input_view"]) as dataset:
                    n = dataset.config["nfft"]
                    keys = sample_keys(len(dataset.snr_db), dataset.frame_count, configuration["samples_per_snr"])
                    channel = group.create_dataset("channel", (len(keys), n, n), dtype="complex128")
                    observation = group.create_dataset("observations", (len(keys), n), dtype="complex128")
                    variances = []
                    for i, (si, fi, di) in enumerate(keys):
                        inputs = dataset.model_input(si, fi)
                        channel[i], observation[i] = inputs["csi"][di], inputs["observations"][di]
                        variances.append(inputs["noise"]["rf_variance"])
                    known = dataset.known
                    data, pilot = dataset.masks["data"], dataset.masks["pilot"]
                    group["variances"] = variances
                    group["symbol_indices"] = [key[2] for key in keys]
                    group["covariance"] = dataset._model["unit_rf_noise_covariance"][:]
                    group["basis"], group["delays"] = dataset.sampled_basis, dataset.delays
                    dataset._model.copy("prior/blocks", group, name="prior")
                    metadata = dict(synthetic=False, keys=keys, snr_db=dataset.snr_db.tolist(),
                                    path=str(dataset_path), model_identity=str(dataset._h.attrs["model_identity"]),
                                    calibration_identity=str(dataset._h.attrs.get("calibration_identity", "")),
                                    input_view=dataset.input_view)
            group["known"] = known
            group["data_indices"] = np.flatnonzero(data)
            group["data_mask"], group["pilot_mask"], group["null_mask"] = data, pilot, ~(data | pilot)
            metadata.update(nfft=n, data_count=int(data.sum()), pilot_count=int(pilot.sum()), blocks=len(group["channel"]))
            manifest[name] = metadata
    return manifest


def summarize_times(milliseconds, sample_count):
    values = np.asarray(milliseconds, dtype=float)
    rounds = values.reshape(-1, sample_count).mean(axis=1)
    return dict(median_round_ms=float(np.median(rounds)), round_mean_ms=rounds.tolist(),
                round_iqr_ms=np.quantile(rounds, [.25, .75]).tolist(),
                latency_p50_ms=float(np.median(values)), latency_p90_ms=float(np.quantile(values, .9)),
                measured_blocks=len(values), samples_ms=values.tolist())


def worker(options):
    # A direct-file subprocess has not imported this checkout's src package.
    sys.path.insert(0, options["source"])
    import scipy  # Load the scientific runtime before torch on Windows.
    import h5py
    import torch
    from threadpoolctl import threadpool_limits
    from src.common.numerics import DenseBackend
    from src.algorithms.interface import EqualizerBatch
    from src.evaluation.fingerprints import numerical_environment

    torch.set_num_threads(options["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(options["seed"])
    _threadpools = threadpool_limits(options["threads"])
    identifier, device = options["algorithm"], options["device"]
    backend = DenseBackend(device)
    with h5py.File(options["inputs"], "r") as stream:
        group = stream[options["channel"]]
        arrays = {key: group[key][:] for key in ("channel", "observations", "variances", "covariance", "known",
                  "data_indices", "data_mask", "pilot_mask", "null_mask", "symbol_indices", "basis", "delays")}
        blocks = [(int(g.attrs["start"]), int(g.attrs["stop"]), g["temporal"][:], g["delay"][:])
                  for g in group["prior"].values()]
    n, count = arrays["channel"].shape[-1], len(arrays["channel"])
    started = time.perf_counter()
    models = {"udnet": ("src.algorithms.udnet", "UDNet"),
              "oamp_net2": ("src.algorithms.oamp_net2", "OAMPNet2"),
              "crenet": ("src.algorithms.icr_mmse", "ICRMMSEnet")}
    model, checkpoint_identity = None, None
    if identifier in models:
        module, class_name = models[identifier]
        if options["smoke"]:
            controls = dict(detector_mode="budgeted", initial_soft_damping=.8, initial_csi_loading=.01) if identifier == "crenet" else {}
            model = getattr(importlib.import_module(module + ".model"), class_name)(**controls).to(device)
            model.eval().requires_grad_(False)
        else:
            checkpoint = Path(options["checkpoint"])
            checkpoint_identity = dict(path=str(checkpoint), sha256=file_hash(checkpoint))
            model, payload = importlib.import_module(module + ".checkpoint").load_model(checkpoint, device)
            if payload["channel"] != options["channel"]:
                raise ValueError("Checkpoint channel mismatch")
            checkpoint_identity.update(schema=payload["schema"], model_config=payload["model_config"],
                                       training_input_view=payload.get("input_view"), inference_code=payload["inference_code"])
    parameters = sum(p.numel() for p in model.parameters()) if model is not None else 0
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) if model is not None else 0
    indices, covariance = arrays["data_indices"], arrays["covariance"]
    whitener = None
    if identifier in ("full_matrix_lmmse", "mmse_dfe", "dd_ce_lmmse", "oamp_net2"):
        from src.algorithms.likelihood import make_whitener
        whitener = make_whitener(covariance, backend)
    preprocessor, modes = None, None
    if identifier == "udnet":
        from src.data.udnet import LocalPreprocessor, WindowPlan
        preprocessor = LocalPreprocessor(WindowPlan.build(n, indices), covariance, device)
    if identifier == "dd_ce_lmmse":
        from src.algorithms.dd_ce_lmmse.detector import ChannelModes, build_modes
        modes = []
        for basis in arrays["basis"]:
            original = build_modes(basis, blocks, options["dd_rank"], backend)
            # Prior blocks address compact tap slots. Scatter the selected delay
            # vectors onto the actual physical delays of the current receiver.
            delays = np.zeros_like(backend.numpy(original.delay))
            delays[:, arrays["delays"]] = backend.numpy(original.delay)[:, :len(arrays["delays"])]
            spectrum = np.fft.fftshift(np.fft.fft(delays, axis=-1), axes=-1)
            modes.append(ChannelModes(original.time, backend.array(delays), backend.array(spectrum)))
    from src.evaluation.registry import AVAILABLE_ALGORITHMS
    modules = {algorithm.identifier: algorithm.module for algorithm in AVAILABLE_ALGORITHMS}
    modules.update(udnet="src.algorithms.udnet.inference", oamp_net2="src.algorithms.oamp_net2.inference")
    equalize = importlib.import_module(modules[identifier]).equalize_batch if identifier in modules else None
    channels, observations = backend.array(arrays["channel"]), backend.array(arrays["observations"])
    known = backend.array(arrays["known"])
    if identifier == "crenet":
        tensor = lambda value: torch.as_tensor(value, device=device)
        channels, observations, known = tensor(arrays["channel"]), tensor(arrays["observations"]), tensor(arrays["known"])
        static = dict(data_mask=tensor(arrays["data_mask"]), pilot_mask=tensor(arrays["pilot_mask"]),
                      null_mask=tensor(arrays["null_mask"]), carrier_positions=torch.arange(n, device=device),
                      unit_covariance=tensor(covariance))
        variances = tensor(arrays["variances"])
    backend.synchronize()
    setup_seconds = time.perf_counter() - started

    def predict(i):
        di = int(arrays["symbol_indices"][i])
        h, y = channels[i:i+1], observations[i:i+1]
        if identifier == "crenet":
            output = model(observations=y, initial_channel=h, known=known[di],
                           rf_variance=variances[i:i+1], return_diagnostics=False, **static)
            return output.probabilities[:, -1, static["data_mask"]].cpu().numpy()
        cancelled = y if identifier == "single_tap_mmse" else y - h @ known[di]
        # Never reuse per-sample Gram/eigen/SVD caches between timed calls.
        batch = EqualizerBatch(h, y, cancelled, arrays["variances"][i:i+1], indices, covariance)
        kwargs = {}
        if whitener is not None:
            kwargs["whitener"] = whitener
        if identifier == "dd_ce_lmmse":
            kwargs.update(known=arrays["known"][di], modes=modes[di], prior_scale=options["dd_prior_scale"])
        if identifier == "udnet":
            kwargs.update(model=model, preprocessor=preprocessor)
        if identifier == "oamp_net2":
            kwargs["model"] = model
        return equalize(batch, backend, **kwargs).symbols

    # Hook counts cover all neural linear/convolution branches, independently of
    # parameter sharing. They are not presented as total receiver FLOPs.
    neural_macs = [0]
    hooks = []
    if model is not None:
        def count_macs(layer, inputs, output):
            if isinstance(layer, torch.nn.Linear):
                neural_macs[0] += output.numel() * layer.in_features
            else:
                neural_macs[0] += output.numel() * layer.weight[0].numel()
        hooks = [layer.register_forward_hook(count_macs) for layer in model.modules()
                 if isinstance(layer, (torch.nn.Linear, torch.nn.Conv1d))]
    with threadpool_limits(options["threads"]), torch.inference_mode():
        output = predict(0)
        for hook in hooks:
            hook.remove()
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite receiver prediction")
        for j in range(options["warmup"]):
            predict(j % count)
        backend.synchronize()
        rng = np.random.default_rng(options["seed"])
        milliseconds = []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for repeat in range(options["repeats"]):
            for i in rng.permutation(count):
                backend.synchronize()
                tick = time.perf_counter_ns()
                output = predict(int(i))
                backend.synchronize()
                milliseconds.append((time.perf_counter_ns() - tick) / 1e6)
                if not np.isfinite(output).all():
                    raise FloatingPointError("Nonfinite timed receiver prediction")
    if checkpoint_identity and file_hash(options["checkpoint"]) != checkpoint_identity["sha256"]:
        raise RuntimeError("Checkpoint changed during timing")
    return dict(algorithm=identifier, channel=options["channel"], parameters=parameters, parameter_bytes=parameter_bytes,
                neural_linear_convolution_macs=neural_macs[0], setup_seconds=setup_seconds,
                model_config=model.config if model is not None else None, checkpoint=checkpoint_identity,
                numerical_environment=numerical_environment(backend),
                peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
                window_count=len(preprocessor.plan.positions) if preprocessor else None,
                **summarize_times(milliseconds, count))


def write_table(run, results):
    rows = []
    for identifier, label in LABELS.items():
        pair = {r["channel"]: r for r in results if r["algorithm"] == identifier}
        if set(pair) != {"NOF1", "NCS1"} or pair["NOF1"]["parameters"] != pair["NCS1"]["parameters"]:
            raise ValueError("Incomplete or structurally different channel pair")
        rows.append(dict(algorithm=label, complexity=COMPLEXITY[identifier], parameters=pair["NOF1"]["parameters"],
                         nof1_ms=pair["NOF1"]["median_round_ms"], ncs1_ms=pair["NCS1"]["median_round_ms"]))
    with (run / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| 算法 | 每符号主要计算复杂度 | 可学习参数量 | NOF1 ms/符号 | NCS1 ms/符号 |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['algorithm']} | `{row['complexity']}` | {row['parameters']:,} | {row['nof1_ms']:.3f} | {row['ncs1_ms']:.3f} |")
    (run / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/seven_algorithm_efficiency.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiment/seven_algorithm_efficiency_20261127")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        options = json.loads(args.worker.read_text(encoding="utf-8"))
        write_json(options["result"], worker(options))
        return
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("repeats", "warmup", "threads", "samples_per_snr", "dd_rank"):
        if type(configuration[key]) is not int or configuration[key] < 1:
            raise ValueError("Invalid benchmark control: " + key)
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "summary.json").exists():
        raise ValueError("Choose a new output directory; an existing result must not be overwritten")
    if args.smoke:
        configuration.update(repeats=1, warmup=1)
    source = export_source(run, configuration["source_commit"])
    inputs = run / "receiver_inputs.h5"
    manifest = prepare_inputs(inputs, configuration, args.smoke)
    identity = dict(sha256=file_hash(inputs), bytes=inputs.stat().st_size)
    write_json(run / "input_manifest.json", manifest)
    results = []
    for identifier in LABELS:
        for channel in ("NOF1", "NCS1"):
            key = identifier + "_" + channel.lower()
            result_path = run / (key + ".json")
            options = dict(configuration, source=str(source), inputs=str(inputs), channel=channel, algorithm=identifier,
                           device=args.device, smoke=args.smoke, result=str(result_path))
            if identifier in configuration["checkpoints"]:
                options["checkpoint"] = str(ROOT / configuration["checkpoints"][identifier].format(channel=channel.lower()))
            specification = run / (key + "_options.json")
            write_json(specification, options)
            print("Measuring " + key, flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(specification)], cwd=source, check=True)
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            write_json(run / "progress.json", dict(completed=len(results), total=14, latest=key))
    if file_hash(inputs) != identity["sha256"]:
        raise RuntimeError("Receiver inputs changed during benchmark")
    table = write_table(run, results)
    summary = dict(complete=True, smoke=args.smoke, device=args.device, source_commit=configuration["source_commit"],
                   created_at_utc=datetime.now(timezone.utc).isoformat(),
                   measurement_code_sha256=file_hash(__file__), legacy_source_archive_sha256=file_hash(run / "historical_source.zip"),
                   configuration=configuration, inputs=identity, input_manifest=manifest, results=results, table=table,
                   timing="batch=1, preloaded device CSI/y; pilot cancellation, sample whitening/preprocessing, all layers and host output; synchronized wall clock; median of repeat means",
                   excluded="disk I/O, dense CSI reconstruction, host-to-device input transfer, checkpoint loading, covariance whitening-factor setup, DD prior-mode setup, labels and scoring",
                   checkpoint_dataset_validation=False, purpose="cost only; historical weights on current receiver inputs, no BER or accuracy claim",
                   complexity_definition="F=N^2*D+N*D^2+D^3; dense full-covariance whitening included; initialization excluded; fixed legacy precisions preserved")
    write_json(run / "summary.json", summary)
    print((run / "comparison.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
