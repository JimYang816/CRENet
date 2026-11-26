"""Check the requested CE-BEM half bandwidth in receiver FFT coordinates."""
import numpy as np
import pytest

from data_generation.src.receiver.ce_bem import ce_basis
from data_generation.src.transmitter.ofdm import fft, ifft


def test_ce_basis_has_33_integer_frequency_shifts_in_every_symbol():
    n, cp, advance = 1024, 480, 32
    total = 4*(n+cp)
    basis = ce_basis({"nfft": n, "cp": cp}, 16)
    assert basis.shape == (total, 33)
    np.testing.assert_allclose(np.linalg.norm(basis, axis=0), 1, atol=1e-13)

    pilot_index = n//2+23
    pilot = np.zeros(n, dtype=complex)
    pilot[pilot_index] = 1
    offsets = np.arange(-16, 17)
    columns = np.arange(len(offsets))
    for symbol in range(4):
        start = symbol*(n+cp)+cp-advance
        frequency = fft((basis[start:start+n]*ifft(pilot)[:, None]).T).T
        expected = np.zeros_like(frequency)
        expected[pilot_index+offsets, columns] = np.exp(2j*np.pi*offsets*start/n)/np.sqrt(total)
        np.testing.assert_allclose(frequency, expected, rtol=0, atol=2e-15)


@pytest.mark.parametrize("half_bandwidth", [-1, 16.5, 512])
def test_ce_basis_rejects_invalid_half_bandwidth(half_bandwidth):
    with pytest.raises(ValueError, match="integer half bandwidth"):
        ce_basis({"nfft": 1024, "cp": 480}, half_bandwidth)
