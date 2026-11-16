"""Exact inference preparation for frozen residual-unfolding parameters.

Training/checkpoint modules stay unchanged. Static correction convolutions are
cached only within one forward, and corrected residuals reuse the previous
residual plus the exact band increment. These algebraic rearrangements can
change floating-point rounding; the two switches retain the old deployment
path for numerical and timing comparisons. No learned weight is approximated.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional

from src.algorithms.crenet.model import (BudgetedMMSELayer, CRENet, CRENetOutput, CSIFeedback, _real_pair)
from src.algorithms.crenet.operators import BandChannel, band_indices
from src.algorithms.crenet.dpss import DPSSChannel, base_local


def _feedback_from_residual(initial_channel, channel, residual, mean, variance,
                            data_mask, pilot_mask, *, use_uncertainty):
    """Same 14 physical features, accepting an already normalized residual.

    Form the shared residual in original channel units before normalization.
    This avoids amplifying rounding by scaling a subtracted residual back up.
    """
    batch, size, _ = initial_channel.shape
    columns, valid = band_indices(size, channel.bandwidth, initial_channel.device)
    h0, ht = base_local(initial_channel, channel.bandwidth), channel.local_entries()
    xj = mean[:, columns]
    vj = variance[:, columns] if use_uncertainty else torch.zeros_like(variance[:, columns])
    masks = [mask[columns].to(variance.dtype)[None].expand(batch, -1, -1)[..., None]
             for mask in (data_mask, pilot_mask)]
    features = torch.cat((_real_pair(h0), _real_pair(ht), _real_pair(xj), vj[..., None],
        _real_pair(residual[..., None]*xj.conj()), _real_pair(ht*vj), *masks,
        valid.to(variance.dtype)[None, ..., None].expand(batch, -1, -1, -1)), -1)
    return CSIFeedback(features*valid[None, ..., None], _real_pair(residual))


class FrozenCSICorrector(nn.Module):
    """Split conv1 before its nonlinearity, retaining all original parameters."""

    def __init__(self, corrector, forward_cache):
        super().__init__()
        self.width = corrector.width
        self.conv1, self.conv2, self.head = corrector.conv1, corrector.conv2, corrector.head
        self._forward_cache = forward_cache
        # Flattening is offset-major: initial CSI (2), masks (3), then noise (3).
        static = [14*k+j for k in range(self.width) for j in (0, 1, 11, 12, 13)]
        static += list(range(14*self.width+2, 14*self.width+5))
        dynamic = [i for i in range(14*self.width+5) if i not in static]
        for name, indices in (("static", static), ("dynamic", dynamic)):
            index = torch.tensor(indices, device=self.conv1.weight.device)
            self.register_buffer(name+"_indices", index, persistent=False)
            self.register_buffer(name+"_weight", self.conv1.weight.index_select(1, index), persistent=False)

    def forward(self, feedback, noise_features):
        if self.training:
            raise RuntimeError("Prepared correction layers cannot be trained")
        features = torch.cat((feedback.offsets.flatten(-2), feedback.residual, noise_features), -1)
        features = features.transpose(1, 2).to(self.conv1.weight.dtype)
        shared = self._forward_cache
        if "correction_static" not in shared:
            shared["correction_static"] = functional.conv1d(
                features.index_select(1, self.static_indices), self.static_weight, self.conv1.bias, padding=1)
        dynamic = functional.conv1d(features.index_select(1, self.dynamic_indices),
                                    self.dynamic_weight, padding=1)
        hidden = functional.silu(shared["correction_static"]+dynamic)
        output = self.head(functional.silu(self.conv2(hidden))).transpose(1, 2)
        real, imaginary, gate = output.split(self.width, dim=-1)
        return torch.complex(real, imaginary), gate.sigmoid()


class FrozenBudgetedMMSELayer(nn.Module):
    def __init__(self, layer, forward_cache=None):
        super().__init__()
        self.original = layer
        self._forward_cache = forward_cache
        self.register_buffer("step_multiplier", 2*layer.step_logit.sigmoid())
        self.register_buffer("loading_coefficient", functional.softplus(layer.csi_loading_log))
        self.register_buffer("uncertainty_coefficient", functional.softplus(layer.uncertainty_log))
        self.register_buffer("temperature_multiplier", layer.temperature_log.exp().clamp(.01, 100))
        self.register_buffer("log_damping", None if layer.soft_damping_logit is None else
                             functional.logsigmoid(layer.soft_damping_logit))
        self.register_buffer("log_previous_damping", None if layer.soft_damping_logit is None else
                             functional.logsigmoid(-layer.soft_damping_logit))
        self.zero_classifier = all(torch.count_nonzero(p).item() == 0 for p in layer.classifier.parameters())
        self._window_geometry = None

    def _window_logits(self, features):
        batch, size, _ = features.shape
        layer = self.original
        count = (size+layer.center_size-1)//layer.center_size
        key = (size, features.device)
        if self._window_geometry is None or self._window_geometry[0] != key:
            left = (layer.window_size-layer.center_size)//2
            starts = torch.arange(count, device=features.device)[:, None]*layer.center_size-left
            positions = starts+torch.arange(layer.window_size, device=features.device)
            valid = ((positions >= 0) & (positions < size))[None, ..., None]
            self._window_geometry = (key, positions.clamp(0, size-1), valid)
        _, indices, valid = self._window_geometry
        windows = features[:, indices]*valid
        hidden = torch.tanh(layer.hidden(windows.flatten(-2).to(layer.hidden.weight.dtype)))
        centers = layer.classifier(hidden).reshape(batch, count*layer.center_size, 4)
        return centers[:, :size]

    @property
    def csi_loading_log(self):
        return self.original.csi_loading_log

    def forward(self, channel, observations, known, mean, variance, data_mask, pilot_mask,
                null_mask, positions, rf_variance, covariance_factor, constellation,
                use_uncertainty=True, covariance_diagonal=None, column_weights=None,
                base_column_energy=None, return_diagnostics=True,
                previous_log_probabilities=None, return_log_probabilities=False,
                precomputed_residual=None):
        if self.training:
            raise RuntimeError("Prepared unfolding layers cannot be trained")
        if covariance_diagonal is None:
            if covariance_factor is None:
                raise ValueError("Diagonal noise requires a covariance diagonal")
            covariance_diagonal = covariance_factor.abs().square().sum(-1)
        solve_cov = ((lambda value: value/covariance_diagonal) if covariance_factor is None else
                     (lambda value: torch.cholesky_solve(value[..., None], covariance_factor).squeeze(-1)))
        weights = covariance_diagonal.reciprocal() if column_weights is None else column_weights
        diagonal = channel.column_energy(weights, base_energy=base_column_energy).clamp_min(1e-20)
        shared = self._forward_cache if self._forward_cache is not None else {}
        if "count" not in shared:
            shared["count"] = data_mask.sum().clamp_min(1)
        count = shared["count"]
        energy = (diagonal*data_mask).sum(-1)/count
        loading = self.loading_coefficient*energy
        regularizer = rf_variance+loading
        denominator = diagonal+regularizer[:, None]
        data_mean = mean*data_mask
        residual = (observations-channel.matvec(data_mean+known)
                    if precomputed_residual is None else precomputed_residual)
        weighted_residual = solve_cov(residual)
        matched_residual = channel.adjoint(weighted_residual)*data_mask
        gradient = matched_residual-regularizer[:, None]*data_mean
        direction = gradient/denominator
        direction_prediction = channel.matvec(direction)
        numerator = (gradient.conj()*direction).real.sum(-1).clamp_min(0)
        curvature = (direction_prediction.conj()*solve_cov(direction_prediction)).real.sum(-1)
        curvature = curvature+regularizer*direction.abs().square().sum(-1)
        step = self.step_multiplier*numerator/curvature.clamp_min(1e-20)
        estimate = data_mean+step[:, None]*direction
        feedback_variance = variance if use_uncertainty else torch.zeros_like(variance)
        uncertainty = (feedback_variance*data_mask).sum(-1)/count
        tau = (regularizer[:, None]/diagonal+self.uncertainty_coefficient*uncertainty[:, None]
               ).clamp_min(1e-5)*self.temperature_multiplier
        logits = -(estimate[..., None]-constellation).abs().square()/tau[..., None]
        if not self.zero_classifier:
            if "masks" not in shared:
                shared["masks"] = torch.stack((data_mask, pilot_mask, null_mask), -1).to(variance.dtype)
                shared["coordinates"] = (positions.to(variance.dtype)-positions.float().mean())/max(len(positions), 1)
            masks, coordinates = shared["masks"], shared["coordinates"]
            features = torch.cat((_real_pair(data_mean), _real_pair(estimate),
                _real_pair(matched_residual/denominator), _real_pair(direction),
                feedback_variance[..., None], torch.log1p(rf_variance[:, None]/diagonal)[..., None],
                masks[None].expand(len(mean), -1, -1), coordinates[None, :, None].expand(len(mean), -1, -1)), -1)
            logits = logits+self._window_logits(features).to(logits.dtype)
        logp = logits.log_softmax(-1)
        if self.log_damping is not None:
            if previous_log_probabilities is None:
                raise ValueError("Soft-state damping requires the previous QPSK distribution")
            logp = torch.logaddexp(logp+self.log_damping, previous_log_probabilities+self.log_previous_damping)
        probabilities = logp.exp()*data_mask[None, :, None]
        next_mean = (probabilities*constellation).sum(-1)+known
        next_variance = (1-next_mean.abs().square()).clamp_min(0)*data_mask
        detail = {}
        if return_diagnostics:
            before = ((residual.conj()*weighted_residual).real.sum(-1)
                      +regularizer*data_mean.abs().square().sum(-1))
            after_residual = residual-step[:, None]*direction_prediction
            after = ((after_residual.conj()*solve_cov(after_residual)).real.sum(-1)
                     +regularizer*estimate.abs().square().sum(-1))
            detail = dict(residual=residual, linear_step=step, csi_loading=loading,
                          linear_objective_before=before, linear_objective_after=after,
                          state_change_energy=((next_mean-mean)*data_mask).abs().square().mean(-1))
        result = probabilities, next_mean, next_variance, detail
        return (*result, logp) if return_log_probabilities else result


def _residual_reuse_forward(self, observations, initial_channel, known, data_mask, pilot_mask, null_mask,
                           carrier_positions, rf_variance, unit_covariance, *, correction_enabled=True,
                           detach_feedback=False, return_diagnostics=True, reuse_base_energy=True):
    """Frozen dense orchestration; keep CRENet.forward as the checkpoint oracle.

    The input checks, normalizations and state updates mirror that oracle. Only
    the post-correction residual is passed explicitly to the prepared detector.
    Non-correcting and DPSS paths retain the original implementation.
    """
    if self.training:
        raise RuntimeError("Prepared unfolding layers cannot be trained")
    if (not correction_enabled or self.config["correction_mode"] == "fixed"
            or isinstance(initial_channel, DPSSChannel)):
        return CRENet.forward(self, observations, initial_channel, known, data_mask, pilot_mask, null_mask,
            carrier_positions, rf_variance, unit_covariance, correction_enabled=correction_enabled,
            detach_feedback=detach_feedback, return_diagnostics=return_diagnostics,
            reuse_base_energy=reuse_base_energy)
    if (initial_channel.ndim != 3 or initial_channel.shape[-1] != initial_channel.shape[-2]
            or observations.shape != initial_channel.shape[:2] or not initial_channel.is_complex()
            or observations.dtype != initial_channel.dtype or observations.device != initial_channel.device):
        raise ValueError("Expected matching complex batched observations and full square initial CSI")
    initial_channel = initial_channel.to(dtype=self.constellation.dtype)
    observations = observations.to(dtype=self.constellation.dtype)
    rf_variance = rf_variance.to(device=initial_channel.device, dtype=observations.real.dtype)
    batch, size = observations.shape
    for mask in (data_mask, pilot_mask, null_mask):
        if mask.shape != (size,) or mask.dtype != torch.bool or mask.device != observations.device:
            raise ValueError("Masks must be boolean vectors on the input device")
    if not torch.all(data_mask.to(torch.int8)+pilot_mask+null_mask == 1):
        raise ValueError("Data, pilot and null masks must form a disjoint full physical grid")
    if (carrier_positions.shape != (size,) or unit_covariance.shape != (size, size)
            or rf_variance.shape != (batch,) or not torch.isfinite(rf_variance).all()
            or torch.any(rf_variance < 0) or not torch.isfinite(initial_channel).all()
            or not torch.isfinite(observations).all()):
        raise ValueError("Invalid physical carrier/noise/channel inputs")
    if known.shape == (size,):
        known = known[None].expand(batch, -1)
    if known.shape != observations.shape or not torch.isfinite(known).all():
        raise ValueError("Known symbols must have shape [N] or [batch,N]")
    known = known.to(device=initial_channel.device, dtype=initial_channel.dtype)*pilot_mask
    real_dtype = observations.real.dtype
    diagonal = unit_covariance.diagonal().real.to(device=initial_channel.device, dtype=real_dtype).clamp_min(1e-20)
    column_weights = diagonal.reciprocal()
    shared_energy = ((initial_channel.abs().square()*column_weights[..., None]).sum(-2)
                     if reuse_base_energy else None)
    covariance_factor = None
    if self.config["noise_mode"] == "full":
        covariance = unit_covariance.to(device=initial_channel.device, dtype=initial_channel.dtype)
        covariance_factor = self._noise_factor(covariance, initial_channel.dtype, initial_channel.device)
        covariance_factor = covariance_factor[None].expand(batch, -1, -1)
    scale = initial_channel.abs().square().sum(-1).mean(-1).sqrt().clamp_min(1e-12)
    entry_scale = scale/math.sqrt(size)
    normalized_initial = initial_channel/scale[:, None, None]
    normalized_noise = rf_variance/scale.square()
    noise_features = torch.stack((torch.log1p(normalized_noise)[:, None].expand(-1, size),
        torch.log(diagonal)[None].expand(batch, -1),
        (carrier_positions.to(real_dtype)/max(size, 1))[None].expand(batch, -1)), -1)
    bands = torch.zeros(batch, size, 2*self.config["bandwidth"]+1,
                        dtype=initial_channel.dtype, device=initial_channel.device)
    columns, valid = band_indices(size, self.config["bandwidth"], initial_channel.device)
    mean = known
    variance = data_mask.to(real_dtype)[None].expand(batch, -1)
    probabilities, states, variances, channels, diagnostics, log_states = [], [], [], [], [], []
    previous_log = observations.real.new_full((*observations.shape, 4), -math.log(4))
    for index, equalizer in enumerate(self.equalizers):
        # Preparation requires frozen weights; detach still preserves the API.
        feedback_mean = mean.detach() if detach_feedback else mean
        feedback_variance = variance.detach() if detach_feedback else variance
        should_correct = self.config["correction_mode"] == "iterative" or index == 0
        residual = None
        passes = self.config["correction_passes"] if index == 0 else 1
        if should_correct:
            for correction_pass in range(passes):
                normalized_channel = BandChannel(normalized_initial, bands/scale[:, None, None])
                raw_residual = observations-BandChannel(initial_channel, bands).matvec(feedback_mean)
                feedback = _feedback_from_residual(normalized_initial, normalized_channel, raw_residual/scale[:, None],
                    feedback_mean, feedback_variance, data_mask, pilot_mask,
                    use_uncertainty=self.config["use_uncertainty"])
                increment, gate = self.corrector(feedback, noise_features)
                if not self.config["use_gate"]:
                    gate = torch.ones_like(gate)
                eta_index = min(index+correction_pass, len(self.equalizers)-1)
                old_bands = bands
                bands = bands+self.damping[eta_index]*(increment*gate).to(initial_channel.dtype)*entry_scale[:, None, None]*valid
                # Use the actual stored difference, including addition rounding.
                delta_prediction = ((bands-old_bands)*feedback_mean[:, columns]*valid).sum(-1)
                residual = raw_residual-delta_prediction
        channel = BandChannel(initial_channel, bands)
        for repeat in range(self.config["equalizer_repeats"]):
            if repeat:
                feedback_mean = mean.detach() if detach_feedback else mean
                feedback_variance = variance.detach() if detach_feedback else variance
                residual = None  # A changed soft state needs a new channel action.
            p, mean, variance, detail, previous_log = equalizer(channel, observations, known, feedback_mean,
                feedback_variance, data_mask, pilot_mask, null_mask, carrier_positions, rf_variance,
                covariance_factor, self.constellation.to(initial_channel.dtype),
                use_uncertainty=self.config["use_uncertainty"], covariance_diagonal=diagonal,
                column_weights=column_weights, base_column_energy=shared_energy,
                return_diagnostics=return_diagnostics,
                previous_log_probabilities=previous_log.detach() if detach_feedback else previous_log,
                return_log_probabilities=True, precomputed_residual=residual)
        probabilities.append(p)
        states.append(mean)
        variances.append(variance)
        channels.append(channel)
        diagnostics.append(detail)
        log_states.append(previous_log)
    return CRENetOutput(torch.stack(probabilities, 1), torch.stack(states, 1), torch.stack(variances, 1),
                        tuple(channels), tuple(diagnostics), log_probabilities=torch.stack(log_states, 1))


def prepare_frozen_inference(model, *, cache_static_correction=True, reuse_correction_residual=True):
    """Prepare an already frozen model in place; do not save it as a checkpoint.

    Call after load_model and device/dtype selection. Loading the original
    checkpoint afresh always restores the trainable representation.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Inference preparation requires eval mode and frozen weights")
    if model.config["detector_mode"] == "budgeted":
        options = (cache_static_correction, reuse_correction_residual)
        if hasattr(model, "_frozen_options"):
            if model._frozen_options != options:
                raise ValueError("Load a fresh model to change frozen inference options")
            return model
        model._frozen_options = options
        if not hasattr(model, "_budgeted_forward_cache"):
            model._budgeted_forward_cache = {}
            model.register_forward_pre_hook(lambda owner, args: owner._budgeted_forward_cache.clear())
        model.equalizers = nn.ModuleList(FrozenBudgetedMMSELayer(layer, model._budgeted_forward_cache).eval()
            if type(layer) is BudgetedMMSELayer else layer for layer in model.equalizers)
        if cache_static_correction:
            model.corrector = FrozenCSICorrector(model.corrector, model._budgeted_forward_cache).eval()
        if reuse_correction_residual:
            model.forward = MethodType(_residual_reuse_forward, model)
    return model
