from __future__ import annotations

import numpy as np
from tsdownsample import M4Downsampler, NaNM4Downsampler


def m4(x: np.ndarray, y: np.ndarray, width: int) -> np.ndarray:
    """Indices to keep (sorted): per pixel column the first, last, min and max."""
    n_out = 4 * width
    if len(x) <= n_out:
        return np.arange(len(x))
    y = np.asarray(y, dtype=np.float64)
    if np.isnan(y).any():
        idx = NaNM4Downsampler().downsample(x, y, n_out=n_out)
    else:
        idx = M4Downsampler().downsample(x, y, n_out=n_out)
    return np.sort(np.asarray(idx, dtype=np.int64))
