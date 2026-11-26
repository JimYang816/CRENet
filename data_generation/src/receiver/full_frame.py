"""Explicit fixed-order full-frame DPSS basis and diagnostics."""

import numpy as np
from scipy.signal.windows import dpss


def model_basis(c):
    total = 4 * (c["nfft"] + c["cp"])
    nw = total * c["dpss_half_bandwidth_hz"] / c["fs_baseband"]
    vectors, concentration = dpss(
        total, nw, Kmax=c["dpss_expected_count"], sym=True, norm=2, return_ratios=True
    )
    return vectors.T.copy(), concentration


def basis_diagnostics(c, concentration):
    total = 4 * (c["nfft"] + c["cp"])
    nw = total * c["dpss_half_bandwidth_hz"] / c["fs_baseband"]
    if len(concentration) != c["dpss_expected_count"]:
        raise ValueError("DPSS concentration count does not match the fixed order")
    return dict(
        order=c["dpss_expected_count"],
        time_bandwidth_product=float(nw),
        tail_ratio=float(max(0.0, 1 - np.sum(concentration) / (2 * nw))),
        tail_threshold=float(c["dpss_tail_ratio"]),
        tail_threshold_enforced=False,
        order_selection="explicit fixed order",
    )
