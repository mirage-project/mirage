"""Shared numerical acceptance measure for 27B GPU probes."""

import torch


def metric(actual, expected):
    delta = actual.float() - expected.float()
    peak = float(expected.float().abs().max())
    rms = float(delta.norm() / expected.float().norm().clamp_min(1e-20))
    maximum = float(delta.abs().max())
    return dict(exact=torch.equal(actual, expected),
                different=int((actual != expected).sum()),
                max_abs=maximum, relative_rms=rms, reference_peak=peak,
                accepted=rms <= .02 and maximum <= .05 * peak + 1e-5)
