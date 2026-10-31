import numpy as np
import pytest
from mimo_receiver.ofdm import (OFDMConfig, exponential_pdp, sample_taps, pilot_grid,
    make_grid, modulate_grid, time_channel, demodulate_waveform, estimate_from_pilots)
from mimo_receiver.channels import complex_normal


def test_time_convolution_matches_frequency_response_with_cp():
    rng = np.random.default_rng(1112)
    config = OFDMConfig(antennas=12, users=4, fft_size=64, cyclic_prefix=8, assumed_taps=4)
    taps = sample_taps(rng, config, exponential_pdp(4))
    symbols = complex_normal(rng, (4, 100))
    grid, count = make_grid(symbols, config)
    np.testing.assert_allclose(grid[:, 1:].reshape(4, -1)[:, :count], symbols)
    received = time_channel(modulate_grid(grid, config), taps, 0, rng)
    fft_received = demodulate_waveform(received, config)
    h = np.fft.fft(taps, n=config.fft_size, axis=-1)
    expected = np.einsum('buf,usf->bsf', h, grid)
    np.testing.assert_allclose(fft_received, expected, atol=1e-12)
    estimate = estimate_from_pilots(fft_received, config, exponential_pdp(4))
    np.testing.assert_allclose(estimate['channel_mean'], h, atol=1e-11)


def test_pilot_energy_and_receiver_noise_calibration():
    rng = np.random.default_rng(1113)
    config = OFDMConfig(antennas=32, users=8, fft_size=128, cyclic_prefix=16, assumed_taps=4)
    pilots = pilot_grid(config)
    np.testing.assert_allclose(np.sum(abs(pilots)**2, axis=1), config.fft_size)
    assert np.all(np.count_nonzero(pilots, axis=0) == 1)
    taps = sample_taps(rng, config, exponential_pdp(4))
    grid, _ = make_grid(complex_normal(rng, (8, 128)), config)
    received = time_channel(modulate_grid(grid, config), taps, 0.02, rng)
    estimate = estimate_from_pilots(demodulate_waveform(received, config), config, exponential_pdp(4))
    assert np.isclose(estimate['noise_variance'], 0.02, rtol=0.08)
    h = np.fft.fft(taps, n=config.fft_size, axis=-1)
    actual = np.mean(abs(h-estimate['channel_mean'])**2)
    assert np.isclose(actual, estimate['channel_error_variance_per_entry'], rtol=0.15)


def test_noise_estimation_requires_residual_degrees_of_freedom():
    with pytest.raises(ValueError):
        OFDMConfig(antennas=32, users=16, fft_size=128, assumed_taps=8).validate()
