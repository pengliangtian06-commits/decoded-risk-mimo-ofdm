"""Custom CP-OFDM waveform and comb-pilot receiver.

This is a reproducible baseband model, not a claim of complete 5G-NR waveform
conformance. Data are passed through an actual finite-impulse-response time
convolution, not directly through a per-subcarrier detection shortcut.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import toeplitz, cholesky
from .channels import complex_normal


@dataclass(frozen=True)
class OFDMConfig:
    antennas: int = 64
    users: int = 16
    fft_size: int = 512
    cyclic_prefix: int = 32
    assumed_taps: int = 8
    pilot_seed: int = 547

    def validate(self):
        if self.antennas < self.users or self.users < 1 or self.fft_size % self.users:
            raise ValueError("Require B>=U and a whole number of comb pilots per user")
        if not 0 < self.assumed_taps <= self.cyclic_prefix + 1:
            raise ValueError("Assumed channel must fit the cyclic prefix")
        if self.fft_size // self.users <= self.assumed_taps:
            raise ValueError("Need more pilots per user than assumed taps to estimate noise from residuals")


def exponential_pdp(taps: int, decay: float = 0.5) -> np.ndarray:
    if taps < 1 or decay <= 0:
        raise ValueError("Invalid tap count or delay decay")
    pdp = np.exp(-decay * np.arange(taps))
    return pdp / pdp.sum()


def sample_taps(rng: np.random.Generator, config: OFDMConfig, pdp: np.ndarray,
                correlation: float = 0.0) -> np.ndarray:
    config.validate()
    pdp = np.asarray(pdp)
    if np.any(pdp < 0) or not np.isclose(pdp.sum(), 1) or len(pdp) > config.cyclic_prefix + 1:
        raise ValueError("PDP must be normalized and fit within the cyclic prefix")
    if not 0 <= correlation < 1:
        raise ValueError("Receive correlation must be in [0,1)")
    factor = cholesky(toeplitz(correlation ** np.arange(config.antennas)), lower=True)
    draws = complex_normal(rng, (config.antennas, config.users * len(pdp)))
    draws = (factor @ draws).reshape(config.antennas, config.users, len(pdp))
    return draws * np.sqrt(pdp[None, None, :] / config.antennas)


def pilot_grid(config: OFDMConfig) -> np.ndarray:
    config.validate()
    rng = np.random.default_rng(config.pilot_seed)
    grid = np.zeros((config.users, config.fft_size), complex)
    for user in range(config.users):
        bins = np.arange(user, config.fft_size, config.users)
        phases = rng.integers(0, 4, len(bins))
        grid[user, bins] = np.sqrt(config.users) * (1j ** phases)
    return grid


def make_grid(symbols: np.ndarray, config: OFDMConfig) -> tuple[np.ndarray, int]:
    symbols = np.asarray(symbols, dtype=complex)
    if symbols.ndim != 2 or symbols.shape[0] != config.users or symbols.shape[1] < 1:
        raise ValueError("Symbols must be U by number of symbols per user")
    data_symbols = int(np.ceil(symbols.shape[1] / config.fft_size))
    grid = np.zeros((config.users, data_symbols + 1, config.fft_size), complex)
    grid[:, 0] = pilot_grid(config)
    grid[:, 1:].reshape(config.users, -1)[:, :symbols.shape[1]] = symbols
    return grid, symbols.shape[1]


def modulate_grid(grid: np.ndarray, config: OFDMConfig) -> np.ndarray:
    waveform = np.fft.ifft(grid, axis=-1, norm="ortho")
    if config.cyclic_prefix:
        waveform = np.concatenate([waveform[..., -config.cyclic_prefix:], waveform], axis=-1)
    return waveform.reshape(config.users, -1)


def time_channel(waveform: np.ndarray, taps: np.ndarray, noise_variance: float,
                 rng: np.random.Generator) -> np.ndarray:
    if noise_variance < 0:
        raise ValueError("Noise variance must be nonnegative")
    antennas, users, length = taps.shape
    if waveform.shape[0] != users:
        raise ValueError("Channel/user dimension mismatch")
    received = np.zeros((antennas, waveform.shape[1] + length - 1), complex)
    for delay in range(length):
        received[:, delay:delay + waveform.shape[1]] += taps[:, :, delay] @ waveform
    received = received[:, :waveform.shape[1]]
    if noise_variance:
        received += np.sqrt(noise_variance) * complex_normal(rng, received.shape)
    return received


def demodulate_waveform(received: np.ndarray, config: OFDMConfig) -> np.ndarray:
    length = config.fft_size + config.cyclic_prefix
    if received.shape[0] != config.antennas or received.shape[1] % length:
        raise ValueError("Invalid OFDM waveform dimensions")
    blocks = received.reshape(config.antennas, -1, length)
    return np.fft.fft(blocks[..., config.cyclic_prefix:], axis=-1, norm="ortho")


def estimate_from_pilots(received_grid: np.ndarray, config: OFDMConfig,
                         assumed_pdp: np.ndarray) -> dict:
    """Comb LS projection, residual noise estimate and per-tap shrinkage.

    No true channel, noise sample or transmitted data bit is accepted by this
    interface. A configured marginal PDP is an assumed receiver prior. The
    scalar error covariance is exact only under the stated independent-error
    model; receive-correlation/PDP mismatch requires robustness evaluation.
    """
    config.validate()
    assumed_pdp = np.asarray(assumed_pdp)
    length = config.assumed_taps
    if len(assumed_pdp) != length or np.any(assumed_pdp < 0) or not np.isclose(assumed_pdp.sum(), 1):
        raise ValueError("Invalid assumed power-delay profile")
    pilots = pilot_grid(config)
    least_squares = np.empty((config.antennas, config.users, length), complex)
    residual_sum, degrees = 0., 0
    pilot_count = config.fft_size // config.users
    for user in range(config.users):
        bins = np.arange(user, config.fft_size, config.users)
        fourier = np.exp(-2j * np.pi * bins[:, None] * np.arange(length)[None, :] / config.fft_size)
        observed = received_grid[:, 0, bins]
        channel_samples = observed / pilots[user, bins]
        ls = channel_samples @ fourier.conj() / pilot_count
        least_squares[:, user] = ls
        residual = observed - (ls @ fourier.T) * pilots[user, bins]
        residual_sum += float(np.sum(abs(residual) ** 2))
        degrees += config.antennas * (pilot_count - length)
    noise = max(residual_sum / degrees, 1e-15)
    tap_noise_variance = noise / (config.users * pilot_count)
    prior = assumed_pdp / config.antennas
    shrinkage = prior / (prior + tap_noise_variance)
    posterior_taps = least_squares * shrinkage[None, None, :]
    error_variance = float(np.sum(prior * tap_noise_variance / (prior + tap_noise_variance)))
    channel_mean = np.fft.fft(posterior_taps, n=config.fft_size, axis=-1)
    return {"channel_mean": channel_mean, "noise_variance": noise,
            "channel_error_variance_per_entry": error_variance,
            "effective_variance": noise + config.users * error_variance,
            "pilot_residual_degrees_of_freedom": degrees,
            "tap_noise_variance": tap_noise_variance}
