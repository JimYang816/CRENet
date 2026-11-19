"""Independent selected-Qeff/L dimension, rank, norm and covariance checks."""
from types import SimpleNamespace

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from data_generation.src.config import load_config
from data_generation.src.receiver.four_data import fixed_four_data_model,FourDataEstimator
from data_generation.src.receiver.four_data_covariance import collect_joint_moments
from data_generation.src.receiver.estimation import FrozenPrior,input_windows
from data_generation.src.transmitter.ofdm import fft
from src.common.numerics import DenseBackend
from src.evaluation.four_data_search import budget_grid,pilot_matrix,rank_check,projection_curve,candidate_moments


def test_budget_grid_never_includes_more_coefficients_than_observations():
    grid=budget_grid()
    assert grid and all(1<=q<=27 and 16<=length<=1024 and q*length<=1536 for q,length in grid)
    assert (24,64) in grid and (3,512) in grid
    assert (5,512) not in grid and (12,224) not in grid


def test_count_is_insufficient_rank_and_condition_rejections_are_separate():
    rng=np.random.default_rng(31)
    matrix=rng.normal(size=(20,6))+1j*rng.normal(size=(20,6))
    backend=DenseBackend()
    with threadpool_limits(2):
        assert rank_check(matrix,backend)["accepted"]
        deficient=matrix.copy()
        deficient[:,-1]=deficient[:,0]
        assert not rank_check(deficient,backend)["accepted"]
        assert rank_check(deficient,backend)["numerical_rank"]==5
        unstable=matrix.copy()
        unstable[:,-1]=unstable[:,0]+1e-8*unstable[:,-1]
        result=rank_check(unstable,backend,max_condition=1e6)
        assert result["numerical_rank"]==6 and not result["accepted"]


def test_pilot_matrix_and_projection_match_literal_reconstruction():
    rng=np.random.default_rng(17)
    random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    z,_=np.linalg.qr(random((28,3)))
    basis=z.reshape(4,7,3)
    known=random((4,7))
    rows=[np.array([0,1,3,6])]*4
    coeff=random((3,4))
    taps=basis@coeff
    direct=fft(np.sum(taps*input_windows(known,4),axis=-1))
    np.testing.assert_allclose(pilot_matrix(basis,known,rows,4)@coeff.ravel(),
        np.concatenate([direct[s,r] for s,r in enumerate(rows)]),atol=1e-13)
    truth=random((2,4,7,7))
    cross=np.einsum("dnq,bdnl->bdql",basis.conj(),truth,optimize=True)
    energy=np.sum(abs(truth)**2,axis=(-2,-1))
    curves=projection_curve(basis,cross,energy,[2,4,7])
    for length,actual in curves.items():
        fitted=np.zeros_like(truth)
        fitted[...,:length]=np.einsum("dnq,bql->bdnl",basis,cross.sum(axis=1)[...,:length])
        expected=np.mean(np.sum(abs(fitted-truth)**2,axis=(-2,-1))/energy)
        np.testing.assert_allclose(actual,expected,rtol=1e-13)


def test_reused_physical_pilot_response_preserves_cross_moments():
    config=load_config()
    config.update(nfft=14,cp=6,fft_advance=1,guard_samples=6)
    model=fixed_four_data_model(config,count=2,length=3)
    factor=linalg.cholesky(model.sampled_basis.conj().T@model.sampled_basis)
    model.basis=model.basis@linalg.inv(factor)
    rng=np.random.default_rng(83)
    coeff=rng.normal(size=(10,2,3))+1j*rng.normal(size=(10,2,3))
    prior=FrozenPrior.fit_coefficients(model,coeff,0,1,(51,),delay_block_size=2)
    known=np.zeros((4,14),complex)
    known[:,[1,5]]=1
    masks=dict(data=np.broadcast_to(np.isin(np.arange(14),[3,4]),(4,14)),tracking=np.ones((4,14),bool))
    selected=rng.normal(size=(56,10))+1j*rng.normal(size=(56,10))
    physical=dict(noise=np.eye(14,dtype=complex),pilot=selected,data=np.eye(56,dtype=complex)*.1,
        metadata=dict(prior_windows=[(51,2)]*10,physical_data_off_diagonal_D_fraction=0.))
    with threadpool_limits(2):
        raw,metadata=candidate_moments(prior,known,masks,physical,DenseBackend())
    np.testing.assert_allclose(raw["model_pilot"]+raw["pilot_residual"],selected,atol=1e-15)
    basis=model.sampled_basis.reshape(4,14,2)
    matrix=pilot_matrix(basis,known,[np.arange(14)]*4,3)
    expected=matrix@coeff.reshape(10,-1).T
    np.testing.assert_allclose(raw["model_pilot"],expected,atol=2e-15)
    np.testing.assert_allclose(raw["coefficients"]@raw["pilot_residual"].conj().T/10,
        coeff.reshape(10,-1).T@(selected-expected).conj().T/10,atol=2e-14)
