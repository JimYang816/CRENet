"""Complex exponential basis on the OFDM subcarrier grid."""
from __future__ import annotations

import numpy as np

from data_generation.src.pdd import SYMBOL_COUNT


def ce_basis(config, half_bandwidth_bins):
    """Use integer subcarrier offsets over the full frame, including the CPs."""
    n = config["nfft"]
    if type(half_bandwidth_bins) is not int or not 0 <= 2*half_bandwidth_bins < n:
        raise ValueError("CE-BEM requires a nonnegative integer half bandwidth below Nyquist")
    total = SYMBOL_COUNT*(n+config["cp"])
    samples = np.arange(total) % n
    offsets = np.arange(-half_bandwidth_bins, half_bandwidth_bins+1)
    return np.exp((2j*np.pi/n)*samples[:, None]*offsets)/np.sqrt(total)
