"""Full-matrix CSI errors with one base subtraction for all unfolding layers."""
import torch

from src.algorithms.crenet.operators import band_indices


def full_matrix_nmse(output, reference):
    """Include every matrix entry; accumulate energies in float64."""
    difference = output.initial_channel-reference
    denominator = reference.abs().double().square().sum((-2,-1)).clamp_min(1e-30)
    initial = difference.abs().double().square().sum((-2,-1))
    bandwidth = output.channels[0].bandwidth
    columns, valid = band_indices(reference.shape[-1], bandwidth, reference.device)
    rows = torch.arange(reference.shape[-1], device=reference.device)[:,None]
    local = difference[:, rows, columns]
    values = []
    for channel in output.channels:
        extra = (2*(local.conj()*channel.bands).real+channel.bands.abs().square())*valid
        values.append((initial+extra.double().sum((-2,-1))).clamp_min(0)/denominator)
    return initial/denominator, torch.stack(values, 1)
