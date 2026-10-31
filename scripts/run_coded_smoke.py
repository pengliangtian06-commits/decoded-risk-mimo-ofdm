#!/usr/bin/env python3
"""A true waveform/pilot/LDPC smoke run, with exact soft-output calibration.

Exact inverse-diagonal soft calibration is charged to this reference receiver.
It is NOT used to assert an iterative detector speedup.
"""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(name, "1")
import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from mimo_receiver.coding import LDPCCodec, LDPCConfig
from mimo_receiver.qam import modulate, maxlog_llr
from mimo_receiver.ofdm import (OFDMConfig, exponential_pdp, sample_taps, make_grid,
    modulate_grid, time_channel, demodulate_waveform, estimate_from_pilots)


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.csv").exists():
        raise FileExistsError("Do not overwrite a recorded smoke run")
    config = OFDMConfig()
    codec = LDPCCodec(LDPCConfig())
    pdp = exponential_pdp(config.assumed_taps)
    rng = np.random.default_rng(1114)
    rows = []
    for snr in [8., 12., 20.]:
        payloads = rng.integers(0, 2, (config.users, 1024), dtype=np.uint8)
        encoded = np.stack([codec.encode(payload) for payload in payloads])
        symbols = modulate(encoded, 16)
        grid, valid_symbols = make_grid(symbols, config)
        taps = sample_taps(rng, config, pdp)
        noise = 10**(-snr/10)
        waveform = time_channel(modulate_grid(grid, config), taps, noise, rng)
        start = perf_counter()
        received = demodulate_waveform(waveform, config)
        estimate = estimate_from_pilots(received, config, pdp)
        estimated_at = perf_counter()
        h = estimate['channel_mean']
        lam = estimate['effective_variance']
        data_blocks = received.shape[1]-1
        detected = np.empty((config.users, data_blocks, config.fft_size), complex)
        variance = np.empty_like(detected.real)
        identity = np.eye(config.users)
        for tone in range(config.fft_size):
            mean = h[:, :, tone]
            factor = cho_factor(mean.conj().T@mean + lam*identity, lower=True, check_finite=False)
            x = cho_solve(factor, mean.conj().T@received[:, 1:, tone], check_finite=False)
            inverse_diagonal = cho_solve(factor, identity, check_finite=False).diagonal().real
            gain = np.maximum(1-lam*inverse_diagonal, 1e-10)
            detected[:, :, tone] = x/gain[:, None]
            variance[:, :, tone] = np.maximum(1/gain[:, None]-1, 1e-12)
        flattened = detected.reshape(config.users, -1)[:, :valid_symbols]
        variances = variance.reshape(config.users, -1)[:, :valid_symbols]
        llrs = maxlog_llr(flattened, variances, 16)
        detected_at = perf_counter()
        decoded_results = [codec.decode(row) for row in llrs]
        finished = perf_counter()
        decoded = np.stack([item[0] for item in decoded_results])
        errors = np.sum(decoded != payloads, axis=1)
        htrue = np.fft.fft(taps, n=config.fft_size, axis=-1)
        row = {'snr_db':snr,'users':config.users,'information_bits':payloads.size,
               'bit_errors':int(errors.sum()),'blocks':config.users,
               'block_errors':int(np.count_nonzero(errors)),
               'crc_failures':sum(item[1] for item in decoded_results),
               'estimated_noise':estimate['noise_variance'],'true_noise':noise,
               'estimated_channel_mse':estimate['channel_error_variance_per_entry'],
               'actual_channel_mse':float(np.mean(abs(h-htrue)**2)),
               'fft_and_estimation_seconds':estimated_at-start,
               'detection_and_llr_seconds':detected_at-estimated_at,
               'decoding_seconds':finished-detected_at,
               'total_receiver_seconds':finished-start,
               'ofdm_symbols':grid.shape[1],
               'airtime_samples':waveform.shape[1]}
        rows.append(row)
        print(json.dumps(row),flush=True)
    with (output/'results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'metadata.json').write_text(json.dumps({'stage':'development_smoke','seed':1114,
        'ofdm':config.__dict__,'code':codec.parameters(),'receiver':'exact LMMSE with exact inverse-diagonal LLR calibration',
        'limitations':['one channel per SNR','not a final BLER estimate','no iterative speedup claim','custom waveform, not full NR compliance']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
