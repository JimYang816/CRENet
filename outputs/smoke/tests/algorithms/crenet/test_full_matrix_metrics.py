"""Exact full-support CSI error and correction gradients."""
from types import SimpleNamespace

import torch

from src.algorithms.crenet.metrics import full_matrix_nmse
from src.algorithms.crenet.operators import BandChannel, band_indices


def test_full_matrix_nmse_counts_out_of_band_errors_and_has_correct_gradient():
    torch.manual_seed(12)
    base = torch.randn(2,9,9,dtype=torch.complex128)
    reference = torch.randn_like(base)
    bands = torch.randn(2,9,5,dtype=torch.complex128,requires_grad=True)
    channel = BandChannel(base,bands)
    output = SimpleNamespace(initial_channel=base,channels=[channel])
    initial,measured = full_matrix_nmse(output,reference)
    columns,valid = band_indices(9,2)
    dense = base.clone().scatter_add(2,columns[None].expand(2,-1,-1),bands*valid)
    expected = (dense-reference).abs().square().sum((-2,-1))/reference.abs().square().sum((-2,-1))
    torch.testing.assert_close(measured[:,0],expected)
    a, = torch.autograd.grad(measured.sum(),bands,retain_graph=True)
    b, = torch.autograd.grad(expected.sum(),bands)
    torch.testing.assert_close(a,b)
