"""Unit-energy square Gray QAM and max-log soft demapping.

LLR convention: log(P(bit=0)/P(bit=1)); positive values favor bit zero.
The first half of each label belongs to the real coordinate.
"""

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def constellation(order: int) -> tuple[np.ndarray, np.ndarray]:
    side = int(np.sqrt(order))
    if order < 4 or side * side != order or side & (side - 1):
        raise ValueError("QAM order must be a square power of two, at least four")
    q = side.bit_length() - 1
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    points, labels = [], []
    for i in range(side):
        for j in range(side):
            gray_i, gray_j = i ^ (i >> 1), j ^ (j >> 1)
            labels.append([(gray_i >> b) & 1 for b in range(q - 1, -1, -1)]
                          + [(gray_j >> b) & 1 for b in range(q - 1, -1, -1)])
            points.append(levels[i] + 1j * levels[j])
    points = np.asarray(points) / np.sqrt(2.0 * (order - 1) / 3.0)
    labels = np.asarray(labels, dtype=np.uint8)
    points.setflags(write=False)
    labels.setflags(write=False)
    return points, labels


def modulate(bits: np.ndarray, order: int) -> np.ndarray:
    points, labels = constellation(order)
    bits = np.asarray(bits)
    width = labels.shape[1]
    if bits.ndim < 1 or bits.shape[-1] % width:
        raise ValueError("Last bit dimension must be divisible by bits per symbol")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Bits must contain only zeros and ones")
    grouped = bits.reshape(*bits.shape[:-1], -1, width)
    weights = 1 << np.arange(width - 1, -1, -1)
    lookup = np.empty(order, dtype=int)
    lookup[labels @ weights] = np.arange(order)
    return points[lookup[grouped @ weights]]


def hard_demodulate(symbols: np.ndarray, order: int) -> np.ndarray:
    points, labels = constellation(order)
    symbols = np.asarray(symbols)
    nearest = np.argmin(np.abs(symbols[..., None] - points) ** 2, axis=-1)
    return labels[nearest].reshape(*symbols.shape[:-1], -1)


def maxlog_llr_exhaustive(symbols: np.ndarray, variance: np.ndarray | float,
               order: int) -> np.ndarray:
    """Soft output for *unbiased* scalar complex observations.

    variance is E|noise|^2, not the variance of one real component.
    Detector-specific bias and residual interference must be corrected before
    calling this routine. It does not create an MIMO soft-output model itself.
    """
    points, labels = constellation(order)
    symbols = np.asarray(symbols)
    variance = np.broadcast_to(np.asarray(variance), symbols.shape)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("Complex disturbance variance must be finite and positive")
    distances = np.abs(symbols[..., None] - points) ** 2
    llr = np.stack([
        (np.min(distances[..., labels[:, b] == 1], axis=-1)
         - np.min(distances[..., labels[:, b] == 0], axis=-1)) / variance
        for b in range(labels.shape[1])
    ], axis=-1)
    return llr.reshape(*symbols.shape[:-1], -1)


def maxlog_llr(symbols: np.ndarray, variance: np.ndarray | float,
               order: int) -> np.ndarray:
    """Exact square-QAM max-log demapper using independent real/imag axes.

    The minimization over the opposite coordinate cancels in each bit LLR.
    This established separability is applied identically to every detector.
    The exhaustive complex-constellation version remains a test oracle.
    """
    points,labels=constellation(order)
    symbols=np.asarray(symbols)
    variance=np.broadcast_to(np.asarray(variance),symbols.shape)
    if np.any(~np.isfinite(symbols)) or np.any(~np.isfinite(variance)) or np.any(variance<=0):
        raise ValueError('Finite observations and positive finite complex variance required')
    side=int(np.sqrt(order)); half=labels.shape[1]//2
    levels=points[::side].real
    axis_labels=labels[::side,:half]
    output=[]
    for coordinate in (symbols.real,symbols.imag):
        distances=(coordinate[...,None]-levels)**2
        output.extend((np.min(distances[...,axis_labels[:,b]==1],axis=-1)
                      -np.min(distances[...,axis_labels[:,b]==0],axis=-1))/variance
                      for b in range(half))
    return np.stack(output,axis=-1).reshape(*symbols.shape[:-1],-1)
