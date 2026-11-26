"""Independent identities for attribution and physically consistent source stress."""
from types import SimpleNamespace

import numpy as np
import torch

from src.algorithms.crenet.operators import BandChannel
from src.evaluation.crenet_generalization import correction_decomposition,perturb_source


def test_decomposition_agrees_with_dense_error_for_non_circular_support():
    torch.manual_seed(37)
    h=torch.randn(3,11,11,dtype=torch.complex128)
    ref=torch.randn_like(h)
    bands=torch.randn(3,11,5,dtype=torch.complex128)
    channel=BandChannel(h,bands)
    result=SimpleNamespace(initial_channel=h,channels=(channel,))
    value=correction_decomposition(result,ref)
    direct=(channel.dense()-ref).abs().square().sum((-2,-1))/ref.abs().square().sum((-2,-1))
    measured=value['initial_nmse']+value['correction_energy'][:,0]+value['error_cross_term'][:,0]
    torch.testing.assert_close(direct,measured,rtol=1e-12,atol=1e-12)


def test_source_error_stress_and_phase_rotation_preserve_observation_pairing():
    torch.manual_seed(24)
    ref=torch.randn(2,8,8,dtype=torch.complex128)
    h=ref+.1*torch.randn_like(ref)
    x=torch.randn(2,8,dtype=torch.complex128)
    y=(ref@x[...,None])[...,0]
    inputs=dict(initial_channel=h,observations=y,rf_variance=torch.ones(2))
    result,truth=perturb_source(inputs,ref,error_multiplier=3.,phase=np.pi/2)
    torch.testing.assert_close(result['initial_channel']-truth,3j*(h-ref))
    torch.testing.assert_close(result['observations'],(truth@x[...,None])[...,0])
    torch.testing.assert_close(result['rf_variance'],inputs['rf_variance'])
    torch.testing.assert_close(inputs['initial_channel'],h)


def test_complete_diagnostic_keeps_checkpoint_frozen(monkeypatch,tmp_path):
    from outputs.smoke.tests.training.test_four_data_experiment import fake_reader
    from src.algorithms.crenet.model import CRENet
    from src.evaluation.crenet_generalization import run
    torch.set_num_threads(1)
    fake_reader(monkeypatch)
    model=CRENet(layers=2,bandwidth=2,window_size=4,center_size=2,hidden_size=4,
                 detector_mode='budgeted',noise_mode='diagonal',initial_soft_damping=.8)
    checkpoint=tmp_path/'model.pt'
    torch.save(dict(config=model.config,model=model.state_dict()),checkpoint)
    before=checkpoint.read_bytes()
    result=run(dict(checkpoint=str(checkpoint),frames=2,batch_size=8,
                    datasets=[dict(channel='NOF1',split='validation',path=str(tmp_path/'validation.h5'))]),
               tmp_path/'diagnostic')
    assert result['complete'] and not result['target_used_for_training_or_selection']
    assert checkpoint.read_bytes()==before
    modes={r['mode'] for r in result['reports'][0]['summary']}
    assert {'original','no_correction','b16_input','b16_corrected','source_error_x2','source_phase_quarter_turn'}<=modes
