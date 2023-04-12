import os
import sys
import math
import glob
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch

from tbparse import SummaryReader
from typing import List, Optional, Tuple
from ray.rllib.policy.sample_batch import SampleBatch

from scipy.interpolate import CubicSpline

linestyle_tuple = [
    ('loosely dotted', (0, (1, 10))),
    ('dotted', (0, (1, 1))),
    ('densely dotted', (0, (1, 1))),

    ('loosely dashed', (0, (5, 10))),
    ('dashed', (0, (5, 5))),
    ('densely dashed', (0, (5, 1))),

    ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('densely dashdotted', (0, (3, 1, 1, 1))),

    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def smooth_curve(y: np.ndarray, weight=0.7) -> np.ndarray:
    last = y[0]
    size = y.size
    smoothed = np.zeros((size))
    smoothed[0] = y[0]
    for i in range(size):
        if i == 0:
            pass
        smoothed_val = last * weight + (1 - weight) * y[i]
        smoothed[i] = smoothed_val
        last = smoothed_val
    return smoothed


def plot_CI(array2D: np.ndarray, label: str, color: str, syntax: str = '-', smooth: float = 0., markersize=14):
    array2D = array2D.T
    shape0 = array2D.shape[0]
    shape1 = array2D.shape[1]
    mean = np.mean(array2D, axis=1)
    x = np.arange(shape0)
    y = mean.copy()
    y_smooth = smooth_curve(y, smooth)
    total_mark = 6
    # print(shape0 // total_mark)
    plt.plot(x, y_smooth, syntax, label=label, markevery=shape0 // total_mark, linewidth=1.5, color=color, markersize=markersize)
    if shape0 == 1:
        return
    std = np.std(array2D, axis=1)
    y1 = mean - 1.96 * (std / np.sqrt(shape1))
    y2 = mean + 1.96 * (std / np.sqrt(shape1))
    y1_smooth = smooth_curve(y1, smooth)
    y2_smooth = smooth_curve(y2, smooth)
    plt.fill_between(x, y1_smooth, y2_smooth, alpha=0.1, color=color)
    return