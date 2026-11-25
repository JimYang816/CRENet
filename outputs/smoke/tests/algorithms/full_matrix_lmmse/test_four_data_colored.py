"""Four-data scorer versus independent colored covariance normal equations."""
import torch

from src.evaluation.four_data_crenet import full_matrix_lmmse


def test_lmmse_matches_independent_colored_covariance_normal_equations():
    torch.manual_seed(25)
    h = torch.randn(2,10,10,dtype=torch.complex128)
    y = torch.randn(2,10,dtype=torch.complex128)
    known = torch.zeros_like(y)
    known[:,1] = 1+1j
    data = torch.arange(10)%2 == 0
    a = torch.randn(10,10,dtype=torch.complex128)
    covariance = a@a.mH+torch.eye(10,dtype=torch.complex128)*.4
    variance = torch.tensor([.1,.001],dtype=torch.float64)
    actual = full_matrix_lmmse(h,y,known,variance,data,torch.linalg.cholesky(covariance))
    hd = h[:,:,data]
    cancelled = y-(h@known[...,None])[...,0]
    weighted = torch.linalg.solve(covariance,hd)
    gram = hd.mH@weighted+variance[:,None,None]*torch.eye(int(data.sum()))
    rhs = hd.mH@torch.linalg.solve(covariance,cancelled[...,None])
    expected = torch.linalg.solve(gram,rhs)[...,0]
    torch.testing.assert_close(actual,expected)
