"""Normalized channel models for bounded feasibility experiments."""

import numpy as np
from scipy.linalg import toeplitz, cholesky


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def conditional_rayleigh(rng: np.random.Generator, antennas: int, users: int,
                         correlation: float = 0.0,
                         error_fraction: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate H = Hbar + E with independent zero-mean conditional error.

    E||H[:,u]||^2 = 1. Hbar has Toeplitz receive-side correlation;
    E has iid entries of variance error_fraction/antennas. Hbar is the receiver
    conditional mean. The scalar error fraction is an ORACLE input in this
    diagnostic model; pilot-derived counterparts are required for final claims.
    """
    if antennas < users or users < 1:
        raise ValueError("Require antennas >= users >= 1")
    if not 0 <= correlation < 1 or not 0 <= error_fraction < 1:
        raise ValueError("Correlation and error fraction must be in [0,1)")
    factor = cholesky(toeplitz(correlation ** np.arange(antennas)), lower=True)
    mean = np.sqrt((1 - error_fraction) / antennas) * factor @ complex_normal(rng, (antennas, users))
    error = np.sqrt(error_fraction / antennas) * complex_normal(rng, (antennas, users))
    return mean + error, mean, error_fraction / antennas


def linear_system(channel_mean: np.ndarray, received: np.ndarray,
                  noise_variance: float, error_variance_per_entry: float = 0.0):
    """LMMSE system for independent unit-energy streams and iid conditional error.

    Colored/heterogeneous channel errors are not silently supported by this
    scalar-covariance reference. All variances are complex variances.
    """
    channel_mean = np.asarray(channel_mean)
    if noise_variance <= 0 or error_variance_per_entry < 0:
        raise ValueError("Noise must be positive and channel error nonnegative")
    effective_variance = noise_variance + channel_mean.shape[1] * error_variance_per_entry
    gram = channel_mean.conj().T @ channel_mean
    system = gram + effective_variance * np.eye(gram.shape[0])
    rhs = channel_mean.conj().T @ received
    return system, rhs, effective_variance
