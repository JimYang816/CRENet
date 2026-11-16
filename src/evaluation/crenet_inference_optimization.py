"""Paired frozen-inference equivalence, accuracy and steady-state timing.

Both paths consume the same accepted compensated inputs and checkpoint. Labels
are accessed only after both predictions. Timing uses real validation inputs,
alternates implementation order, and excludes supervision and disk reads.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from src.algorithms.crenet.checkpoint import INFERENCE_CODE, load_model, validate_dataset
from src.algorithms.crenet.inference import prepare_frozen_inference
from src.algorithms.crenet.model import csi_nmse
from src.common.experiment import write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.crenet import supervision
from src.data.crenet_dense_training import dense_input_batches
from src.data.crenet_training import ReferenceCache
from src.evaluation.crenet import RecordMetrics, _initial_nmse, _record_disjoint
from src.evaluation.fingerprints import DataIdentity, numerical_environment, source_fingerprint


VARIANTS = {"before": (False, False), "legacy_tf32": (False, False), "static_only": (True, False),
            "residual_only": (False, True), "both": (True, True)}
FIELDS = ("probabilities", "log_probabilities", "symbols", "variances", "bands")
CANDIDATES = ("residual_only", "both")
# Fixed before full validation/test. NCS1's FP32 oracle itself differs from
# identical promoted FP64 weights by ~2.5e-5 in probability on validation probes.
ATOL, RTOL = 1e-4, 1e-4
CODE = INFERENCE_CODE+("src.algorithms.crenet.inference", "src.algorithms.crenet.legacy_identity",
                       "src.evaluation.crenet_inference_optimization",
                       "src.evaluation.crenet", "src.data.crenet_dense_training",
                       "src.data.crenet_training", "src.data.crenet_sampling")


@contextmanager
def _comparison_precision():
    previous = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


def materialize(cpu, device):
    """Exactly the existing complex128 receiver reconstruction, then model cast."""
    source = cpu["initial_channel"]
    inputs = {key: value.to(device) for key, value in cpu.items() if key != "initial_channel"}
    inputs["initial_channel"] = source.dense(device)
    return inputs


def compare_outputs(expected, actual, stats, data_mask):
    """Track maximum elementwise error/tolerance, including every unfolded layer."""
    for name in FIELDS:
        left, right = getattr(expected, name), getattr(actual, name)
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise FloatingPointError("Nonfinite comparison output: "+name)
        difference = (left-right).abs()
        row = stats.setdefault(name, dict(max_absolute_error=0., max_tolerance_ratio=0.))
        row["max_absolute_error"] = max(row["max_absolute_error"], float(difference.max()))
        if name == "bands":
            # CSI has physical units and may be thousands in amplitude. Use
            # exactly the model's per-sample channel scale for absolute error.
            base = expected.initial_channel
            scale = base.abs().square().sum(-1).mean(-1).sqrt().clamp_min(1e-12)
            scale = scale[:, None, None, None]
            row["max_normalized_absolute_error"] = max(row.get("max_normalized_absolute_error", 0.),
                                                       float((difference/scale).max()))
            ratio = difference/(ATOL*scale+RTOL*left.abs())
        else:
            ratio = difference/(ATOL+RTOL*left.abs())
        row["max_tolerance_ratio"] = max(row["max_tolerance_ratio"], float(ratio.max()))
    left, right = expected.symbols[..., data_mask], actual.symbols[..., data_mask]
    changed = ((left.real < 0) != (right.real < 0)).sum((0, 2))
    changed += ((left.imag < 0) != (right.imag < 0)).sum((0, 2))
    return changed.cpu().numpy()


def _models(checkpoint, dataset, device, variants):
    model, payload = load_model(checkpoint, device)
    validate_dataset(payload, dataset)
    if model.config["detector_mode"] != "budgeted" or model.config["layers"] != 8:
        raise ValueError("This comparison requires the current eight-layer residual model")
    models = {}
    for name in variants:
        candidate = copy.deepcopy(model)
        prepare_frozen_inference(candidate, cache_static_correction=VARIANTS[name][0],
                                 reuse_correction_residual=VARIANTS[name][1])
        # Preparation may wrap modules, but every learned tensor must be intact.
        parameters = {key.replace(".original.", "."): value for key, value in candidate.named_parameters()}
        for key, value in model.named_parameters():
            if key not in parameters or not torch.equal(value, parameters[key]):
                raise AssertionError("Inference preparation changed learned weights: "+key)
        if len(parameters) != len(dict(model.named_parameters())) or candidate.config != model.config:
            raise AssertionError("Inference preparation changed model structure/configuration")
        models[name] = candidate
    return models, payload


@_comparison_precision()
def evaluate_pair(dataset_path, checkpoint, output, *, device="cpu", batch_size=2, frame_limit=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    backend = DenseBackend(device)
    with CompactDataset(dataset_path, input_view="compensated") as dataset, torch.inference_mode():
        if dataset.split not in ("validation", "test"):
            raise ValueError("Only fixed validation/test splits may be compared")
        _record_disjoint(dataset)
        models, payload = _models(checkpoint, dataset, device, ("before", *CANDIDATES, "legacy_tf32"))
        accumulators = {name: RecordMetrics(8) for name in models}
        statistics_by_field = {name: {} for name in CANDIDATES}
        disagreements = {name: np.zeros((len(dataset.snr_db), 8), dtype=np.int64) for name in CANDIDATES}
        blocks, batches = 0, 0
        predictions_seconds = {name: 0. for name in models}
        cache = ReferenceCache(512*2**20)
        for cpu, keys in dense_input_batches(dataset, batch_size=batch_size, frame_limit=frame_limit,
                                             precision=models["before"].config["precision"], reference_cache=cache):
            inputs = materialize(cpu, device)
            outputs = {}
            order = list(models) if batches % 2 == 0 else list(reversed(models))
            for name in order:
                torch.backends.cudnn.allow_tf32 = name == "legacy_tf32"
                backend.synchronize()
                tick = time.perf_counter()
                outputs[name] = models[name](**inputs, return_diagnostics=False)
                backend.synchronize()
                predictions_seconds[name] += time.perf_counter()-tick
            torch.backends.cudnn.allow_tf32 = False
            mask = inputs["data_mask"]
            for name in CANDIDATES:
                disagreements[name][keys[0][0]] += compare_outputs(outputs["before"], outputs[name],
                                                                  statistics_by_field[name], mask)
            # Independent labels/reference are opened after both forwards finish.
            precision = models["before"].config["precision"]
            target = supervision(dataset, keys, device=device, precision=precision, include_reference=False)
            # Exact saved reference matrices; reuse only the bounded RAM cache,
            # after prediction, to avoid decompressing the same window twice.
            references = [cache.get_full(dataset, int(dataset._h["samples/window_index"][fi]), d, precision)
                          for _, fi, d in keys]
            target["reference_channel"] = torch.cat(references).to(device)
            before_nmse = _initial_nmse(inputs["initial_channel"], target["reference_channel"])
            records = target["records"].cpu().tolist()
            bit_count = 2*int(mask.sum())
            for name, prediction in outputs.items():
                predicted = prediction.symbols[..., mask]
                truth = target["symbols"][:, None, :]
                errors = ((predicted.real < 0) != (truth.real < 0)).sum(-1)
                errors += ((predicted.imag < 0) != (truth.imag < 0)).sum(-1)
                after = csi_nmse(prediction, target["reference_channel"])
                nmse = torch.cat((before_nmse[:, None], after), -1).cpu().numpy()
                errors = errors.cpu().numpy()
                for i, ((si, _, _), record) in enumerate(zip(keys, records)):
                    accumulators[name].add(si, record, errors[i], bit_count, nmse[i])
            blocks += len(keys)
            batches += 1
            if batches % 50 == 0:
                progress = dict(phase=dataset.split, blocks=blocks, batches=batches,
                                elapsed_seconds=time.perf_counter()-started, numerical=statistics_by_field)
                write_json(output/"progress.json", progress)
                print(f"{dataset.channel} {dataset.split}: {blocks} blocks", flush=True)
            del outputs, inputs, target
        metrics = {name: accumulator.report(dataset.snr_db) for name, accumulator in accumulators.items()}
        deltas = {name: [] for name in CANDIDATES}
        for name in CANDIDATES:
            for si, (left, right) in enumerate(zip(metrics["before"], metrics[name])):
                deltas[name].append(dict(snr_db=left["snr_db"], total_bits=left["total_bits"],
                    final_decision_disagreements=int(disagreements[name][si, -1]),
                    layer_decision_disagreements=disagreements[name][si].tolist(),
                    final_error_count_change=right["bit_errors"][-1]-left["bit_errors"][-1],
                    final_ber_change=right["ber"][-1]-left["ber"][-1],
                    final_nmse_db_change=right["mean_nmse_db"][-1]-left["mean_nmse_db"][-1],
                    final_error_count_change_from_legacy_tf32=right["bit_errors"][-1]-metrics["legacy_tf32"][si]["bit_errors"][-1],
                    final_nmse_db_change_from_legacy_tf32=right["mean_nmse_db"][-1]-metrics["legacy_tf32"][si]["mean_nmse_db"][-1]))
        equivalent = {name: all(row["max_tolerance_ratio"] <= 1 for row in fields.values())
                      for name, fields in statistics_by_field.items()}
        result = dict(complete=True, channel=dataset.channel, split=dataset.split, input_view="compensated",
            blocks=blocks, batches=batches, frame_limit=frame_limit, batch_size=batch_size,
            model_config=payload["model_config"], learned_weights_unchanged=True,
            numerical=dict(atol=ATOL, rtol=RTOL, allclose=equivalent, fields=statistics_by_field,
                           band_absolute_tolerance_units="per-sample initial-channel RMS row norm, as used by model normalization",
                           bitwise_equal={name: all(row["max_absolute_error"] == 0 for row in fields.values())
                                          for name, fields in statistics_by_field.items()}),
            per_snr=metrics, differences=deltas, forward_seconds_during_scoring=predictions_seconds,
            elapsed_seconds=time.perf_counter()-started,
            input_backend="dense_device: identical reconstructed tensor supplied to both paths",
            timing_note="Scoring-pass timings are descriptive; use repeated validation benchmark for latency.")
        write_json(output/"metrics.json", result)
        _plot(result, output)
        write_json(output/"progress.json", dict(phase="completed", blocks=blocks, allclose=equivalent))
        # A failed candidate remains a failed check, but must not suppress the
        # other channel's accuracy/timing audit. Never widen tolerances here.
        return result


def _plot(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for field, filename, ylabel in (("ber", "ber_snr.png", "BER"),
                                    ("mean_nmse_db", "nmse_snr.png", "Mean full-matrix CSI NMSE (dB)")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for name, style in (("before", "o-"), ("residual_only", "s-."), ("both", "x--"), ("legacy_tf32", ":")):
            rows = report["per_snr"][name]
            ax.plot([r["snr_db"] for r in rows], [r[field][-1] for r in rows], style, label=name)
        if field == "ber":
            ax.set_yscale("log")
        ax.set(xlabel="SNR (dB)", ylabel=ylabel, title=report["channel"]+" / "+report["split"])
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory/filename, dpi=180)
        plt.close(fig)


@_comparison_precision()
def benchmark_validation(dataset_path, checkpoint, output, *, device="cpu", batch_size=2, repeats=11):
    """Paired wall-clock medians, including Python dispatch and all eight layers."""
    backend = DenseBackend(device)
    with CompactDataset(dataset_path, input_view="compensated") as dataset, torch.inference_mode():
        if dataset.split != "validation":
            raise ValueError("Timing selection uses validation inputs only")
        models, payload = _models(checkpoint, dataset, device, VARIANTS)
        # Two frames at every SNR and both D blocks; no symbol/CSI supervision.
        sources = list(dense_input_batches(dataset, batch_size=batch_size,
            frame_indices=list(range(min(batch_size, dataset.frame_count))),
            precision=models["before"].config["precision"], reference_cache=ReferenceCache(256*2**20)))
        batches = [materialize(cpu, device) for cpu, _ in sources]
        blocks = sum(len(keys) for _, keys in sources)
        for _ in range(3):
            for name, model in models.items():
                torch.backends.cudnn.allow_tf32 = name == "legacy_tf32"
                for inputs in batches:
                    model(**inputs, return_diagnostics=False)
        backend.synchronize()
        modes = {}
        for mode in ("forward", "reconstruction_and_forward"):
            samples = {name: [] for name in models}
            names = list(models)
            for repeat in range(repeats):
                order = names[repeat % len(names):]+names[:repeat % len(names)]
                if repeat % 2:
                    order = list(reversed(order))
                for name in order:
                    torch.backends.cudnn.allow_tf32 = name == "legacy_tf32"
                    backend.synchronize()
                    tick = time.perf_counter()
                    for index, (cpu, _) in enumerate(sources):
                        inputs = batches[index] if mode == "forward" else materialize(cpu, device)
                        models[name](**inputs, return_diagnostics=False)
                    backend.synchronize()
                    samples[name].append((time.perf_counter()-tick)*1000/blocks)
            torch.backends.cudnn.allow_tf32 = False
            modes[mode] = {name: dict(ms_per_block_median=statistics.median(values),
                                     ms_per_block_min=min(values), ms_per_block_max=max(values),
                                     samples_ms_per_block=values) for name, values in samples.items()}
            baseline = modes[mode]["before"]["ms_per_block_median"]
            for row in modes[mode].values():
                row["relative_to_before"] = row["ms_per_block_median"]/baseline
        result = dict(complete=True, channel=dataset.channel, input_view="compensated", split="validation",
            batch_size=batch_size, repeats=repeats, warmup_rounds=3, blocks_per_repeat=blocks,
            variants={name: dict(cache_static_correction=flags[0], reuse_correction_residual=flags[1])
                      for name, flags in VARIANTS.items()}, measurements=modes,
            model_config=payload["model_config"], learned_weights_unchanged=True,
            parameters=sum(p.numel() for p in models["before"].parameters()),
            definition="Synchronized wall-clock, alternating variant order; static per-sample caching is charged on every forward. No labels, disk I/O, checkpoint load or scoring in timing.",
            reconstruction_definition="Host-to-device transfer and existing complex128 dense CSI reconstruction, then complex64 forward; source receiver fields are preloaded in host RAM.",
            numerical_environment=numerical_environment(backend),
            precision_policy="All variants use full FP32 convolution except explicit legacy_tf32=True audit; weights and complex64 operators are identical.",
            torch_settings=dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, threads=torch.get_num_threads()))
        write_json(output, result)
        return result


def run_channel(directory, checkpoint, output, *, device="cpu", batch_size=2, repeats=11,
                validation_frames=None, test_frames=None):
    directory, checkpoint, output = Path(directory), Path(checkpoint), Path(output)
    if (output/"summary.json").exists():
        raise FileExistsError("Use a new comparison output directory")
    output.mkdir(parents=True, exist_ok=True)
    identity = DataIdentity({name: directory/name for name in ("model.h5", "validation.h5", "test.h5")}
                            | {"checkpoint": checkpoint})
    code = source_fingerprint(CODE)
    write_json(output/"provenance.json", dict(data=identity.files, code=code, atol=ATOL, rtol=RTOL))
    # Acceptance tolerance is fixed before touching test. No fitting or selection.
    validation = evaluate_pair(directory/"validation.h5", checkpoint, output/"validation",
        device=device, batch_size=batch_size, frame_limit=validation_frames)
    timing = benchmark_validation(directory/"validation.h5", checkpoint, output/"timing.json",
                                   device=device, batch_size=batch_size, repeats=repeats)
    test = evaluate_pair(directory/"test.h5", checkpoint, output/"test",
                          device=device, batch_size=batch_size, frame_limit=test_frames)
    identity.assert_unchanged()
    if source_fingerprint(CODE) != code:
        raise RuntimeError("Source changed during paired evaluation")
    selected = "residual_only"
    result = dict(complete=True, channel=test["channel"], learned_weights_unchanged=True,
        default_variant="before", candidate_variant=selected,
        candidate_numerical_checks_passed=validation["numerical"]["allclose"][selected] and test["numerical"]["allclose"][selected],
        validation_allclose=validation["numerical"]["allclose"], test_allclose=test["numerical"]["allclose"],
        test_blocks=test["blocks"], final_decision_disagreements={name: sum(r["final_decision_disagreements"] for r in rows)
                                                               for name, rows in test["differences"].items()},
        mean_ber={name: float(np.mean([r["ber"][-1] for r in rows])) for name, rows in test["per_snr"].items()},
        max_absolute_nmse_db_change={name: max(abs(r["final_nmse_db_change"]) for r in rows)
                                    for name, rows in test["differences"].items()},
        timing=timing["measurements"], data_identity=identity.files, code=code)
    write_json(output/"summary.json", result)
    return result
