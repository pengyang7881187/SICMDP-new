import scipy
import numpy as np
from numpy import ndarray


# GPU version: torch-discounted-cumsum package.
# Batch version of discount_cumsum in postprocess.py.
def batch_discount_cumsum(x: ndarray, gamma: float) -> ndarray:
    assert x.ndim == 2
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[:, ::-1], axis=1)[:, ::-1]


def batch_z_score_normalization(x: ndarray) -> ndarray:
    assert x.ndim == 2
    return (x - np.mean(x, axis=0, keepdims=True)) / np.maximum(np.std(x, axis=0, keepdims=True), 1e-4)
