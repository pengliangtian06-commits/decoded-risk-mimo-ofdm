#!/usr/bin/env python3
"""Development-only complete soft-output pipelines; concurrent timings exploratory.

Canonical independent unit is a frame/channel draw. Within-frame users are
retained as a paired vector, never silently promoted to independent channels.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
from dataclasses import replace
import concurrent.futures
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from mimo_receiver import _core
from mimo_receiver.coding import LDPCCodec, LDPCConfig
from mimo_receiver.qam import modulate
from mimo_receiver.ofdm import (OFDMConfig, exponential_pdp, sample_taps, make_grid,
    modulate_grid, time_channel, demodulate_waveform, estimate_from_pilots)
from mimo_receiver.receiver import DetectorPolicy, detect_soft
from mimo_receiver.tdl import quantized_pdp


def simulate(task):
    settings, condition, condition_id, draw = task
    (antennas, users), rho, snr = condition
    rng = np.random.default_rng(np.random.SeedSequence([settings['seed_root'], condition_id, draw]))
    cfg = OFDMConfig(antennas, users, settings['fft_size'], settings['cyclic_prefix'], settings['taps'])
    codec = LDPCCodec(LDPCConfig(qam_order=settings['qam_order'], max_iterations=settings['decoder_iterations']))
    payloads = rng.integers(0, 2, (users, 1024), dtype=np.uint8)
    encoded = np.stack([codec.encode(payload) for payload in payloads])
    grid, valid = make_grid(modulate(encoded, settings['qam_order']), cfg)
    profile=settings.get('channel_profile','exponential')
    profile_metadata=None
    if profile=='exponential':
        actual_pdp=exponential_pdp(cfg.assumed_taps)
    else:
        actual_pdp,profile_metadata=quantized_pdp(profile,settings['rms_delay_ns'],
                                                cfg.fft_size*settings.get('subcarrier_spacing_hz',15000))
    if settings.get('receiver_prior','source')=='matched':
        cfg=replace(cfg,assumed_taps=len(actual_pdp))
        cfg.validate();pdp=actual_pdp
    else:
        pdp=exponential_pdp(cfg.assumed_taps)
    taps = sample_taps(rng, cfg, actual_pdp, rho)
    noise = 10**(-snr/10)
    received = time_channel(modulate_grid(grid, cfg), taps, noise, rng)
    tick = perf_counter()
    frequency = demodulate_waveform(received, cfg)
    estimates = estimate_from_pilots(frequency, cfg, pdp)
    estimation_seconds = perf_counter() - tick
    # Independent random selection, before seeing any method's outcomes.
    audit_user = int(rng.integers(users))
    outputs = {}
    # Randomize method order to spread warm-up/cache/thermal effects. Formal
    # speed comparison still requires a separate serial repeated benchmark.
    for index in rng.permutation(len(settings['policies'])):
        policy = DetectorPolicy(**settings['policies'][index])
        detection = detect_soft(estimates['channel_mean'], frequency[:, 1:],
                                estimates['effective_variance'], policy,
                                settings['qam_order'], valid)
        tick = perf_counter()
        decoded = [codec.decode(values) for values in detection['llr']]
        decoder_seconds = perf_counter() - tick
        decoded_bits = np.stack([item[0] for item in decoded])
        errors = np.sum(decoded_bits != payloads, axis=1)
        hard = detection['llr'] < 0
        uncoded_errors = int(np.count_nonzero(hard != encoded))
        iterations = detection['iterations']
        row = {'method': policy.name, 'blocks': users,
               'block_errors': int(np.count_nonzero(errors)), 'bit_errors': int(errors.sum()),
               'per_user_bit_errors': errors.tolist(), 'audit_user': audit_user,
               'audit_user_error': bool(errors[audit_user] > 0),
               'crc_failures': sum(item[1] for item in decoded),
               'decoder_iterations': sum(item[2] for item in decoded),
               'uncoded_errors': uncoded_errors, 'uncoded_bits': int(encoded.size),
               'iterations': int(iterations.sum()), 'vectors': len(iterations),
               'iteration_p95': float(np.percentile(iterations, 95)),
               'stop_reasons': dict(Counter(detection['stop_reasons'])),
               'preparation_seconds': detection['preparation_seconds'],
               'solve_seconds': detection['solve_seconds'],
               'soft_calibration_seconds': detection['soft_calibration_seconds'],
               'llr_seconds': detection['llr_seconds'],
               'detection_total_seconds': detection['detection_total_seconds'],
               'decoder_seconds': decoder_seconds, 'estimation_seconds': estimation_seconds,
               'receiver_seconds': estimation_seconds + detection['detection_total_seconds'] + decoder_seconds,
               'exact_diagonal_tones': detection['exact_diagonal_tones']}
        row['gain_clip_count'] = detection.get('gain_clip_count',0)
        outputs[policy.name] = (row, detection['gain_proxy'])
    reference_row, reference_gain = outputs['exact']
    rows = []
    reference_errors = np.array(reference_row['per_user_bit_errors'])
    for name, (row, gain) in outputs.items():
        harmful = (np.array(row['per_user_bit_errors']) > 0) & (reference_errors == 0)
        reference_broadcast = reference_gain[:,None,:] if gain.ndim==3 else reference_gain
        row.update(extra_errors=int(harmful.sum()), audit_user_harm=bool(harmful[audit_user]),
                   gain_absolute_error_max=float(np.max(abs(gain-reference_broadcast))),
                   gain_absolute_error_mean=float(np.mean(abs(gain-reference_broadcast))))
        rows.append(row)
    return {'task_id': f'{condition_id}:{draw}', 'condition_id': condition_id,
            'draw': draw, 'antennas': antennas, 'users': users, 'rho': rho, 'snr_db': snr,
            'estimated_noise': estimates['noise_variance'], 'true_noise': noise,
            'channel_profile':profile,'profile_metadata':profile_metadata,
            'receiver_prior':settings.get('receiver_prior','source'),
            'airtime_samples': received.shape[1], 'rows': sorted(rows, key=lambda r:r['method'])}


def run(config_path, output):
    raw = config_path.read_bytes(); settings = json.loads(raw)
    if settings['stage'] != 'exploratory_development' or not 1100 <= settings['seed_root'] < 1200:
        raise ValueError('Development seed root required')
    if sum(p['name'] == 'exact' for p in settings['policies']) != 1:
        raise ValueError('One exact reference required')
    output.mkdir(parents=True, exist_ok=True)
    sources = ['scripts/run_soft_output_screen.py','cpp/core.cpp','setup.py',
               'pyproject.toml','requirements-lock.txt']
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'src/mimo_receiver').glob('*.py')]
    source_hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    metadata = {'settings': settings, 'config_sha256': hashlib.sha256(raw).hexdigest(),
                'source_hashes': source_hashes,
                'extension_sha256':hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
                'completed': False,
                'stage': 'exploratory', 'timing_limit': 'Concurrent development timings; not a speedup claim'}
    meta = output/'metadata.json'; path = output/'channel_records.jsonl'
    if meta.exists():
        previous = json.loads(meta.read_text())
        if any(previous[k] != metadata[k] for k in ('config_sha256', 'source_hashes')):
            raise ValueError('Refusing resume after config/source change')
    else:
        meta.write_text(json.dumps(metadata, indent=2))
        for source in sources:
            target=output/'source_snapshot'/source
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/source).read_bytes())
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    done = {r['task_id'] for r in records}
    if len(done) != len(records):
        raise ValueError('Duplicate task records')
    conditions = list(itertools.product(settings['antenna_user_pairs'], settings['correlations'], settings['snr_db']))
    tasks = [(settings, c, i, draw) for i, c in enumerate(conditions)
             for draw in range(settings['draws_per_condition']) if f'{i}:{draw}' not in done]
    total = len(conditions)*settings['draws_per_condition']
    print(f'{len(done)}/{total} saved, {len(tasks)} remaining', flush=True)
    tick = perf_counter()
    with path.open('a') as handle:
        with concurrent.futures.ProcessPoolExecutor(max_workers=settings['workers'],
                mp_context=multiprocessing.get_context('spawn')) as executor:
            for item in executor.map(simulate, tasks, chunksize=1):
                records.append(item)
                handle.write(json.dumps(item)+'\n'); handle.flush()
                print(f'{len(records)}/{total} saved {item["task_id"]}', flush=True)
    metadata.update(completed=True, channels=len(records), rows=sum(len(r['rows']) for r in records),
                    invocation_seconds=perf_counter()-tick)
    meta.write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/soft_output_screen.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.config, args.output)
