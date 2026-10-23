"""Validity-aware temporal operations; never interpolate an unobserved hand."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def pose_validity(poses, side):
    finite = np.isfinite(poses[side]).all(axis=1)
    key = side+'_valid'
    return finite & poses[key].astype(bool) if key in poses else finite


def filter_valid_runs(values, valid, sigma=1.):
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if values.shape[0] != len(valid):
        raise ValueError('Values and validity must have the same time dimension')
    result = values.copy()
    indices = np.flatnonzero(valid)
    for group in np.split(indices, np.flatnonzero(np.diff(indices)>1)+1):
        if len(group):
            result[group] = gaussian_filter1d(values[group], sigma, axis=0, mode='nearest')
    return result
