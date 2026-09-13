"""Illustrative food-odor encoding, with explicitly engineered stereo gain."""
import numpy as np


def odor_input_rates(concentration):
    c = np.asarray(concentration,dtype=float)
    if c.shape != (2,) or not np.isfinite(c).all() or (c<0).any() or (c>1).any():
        raise ValueError('Expected two normalized odor samples in 0…1')
    mean = float(c.mean())
    base = 110*mean/(mean+.04)
    # A small antenna baseline is noisy with only 68 modeled ORNs. Enhance
    # contrast before Poisson encoding, explicitly an artificial sensory gain,
    # not a claim about independent receptor transduction or binding kinetics.
    stereo = float(np.clip(16*(c[0]-c[1])/(c.sum()+1e-9),-.85,.85))
    return np.clip(base*np.array([1+stereo,1-stereo]),0,220)
