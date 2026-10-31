#!/usr/bin/env python3
"""Serial repeated complete-receiver benchmark with paired frozen observations."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import scipy
from mimo_receiver import _core
from mimo_receiver.coding import LDPCCodec, LDPCConfig
from mimo_receiver.qam import modulate
from mimo_receiver.ofdm import (OFDMConfig, exponential_pdp, sample_taps, make_grid,
    modulate_grid, time_channel, demodulate_waveform, estimate_from_pilots)
from mimo_receiver.receiver import DetectorPolicy, detect_soft, detect_soft_reference


def run(output, repetitions=25, warmup=3):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/soft_output_screen.json').read_text())
    policies = [DetectorPolicy(**p) for p in config['policies']]
    rows = []
    sources = ['scripts/benchmark_receiver.py', 'src/mimo_receiver/receiver.py',
               'src/mimo_receiver/ofdm.py', 'src/mimo_receiver/coding.py', 'cpp/core.cpp']
    hashes = {}
    for source in sources:
        data = (ROOT/source).read_bytes(); hashes[source] = hashlib.sha256(data).hexdigest()
        target = output/'source_snapshot'/source
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    metadata = {'stage': 'development_serial_timing', 'seed_root':1133,
        'source_hashes':hashes, 'extension_sha256':hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        'policies':[p.__dict__ for p in policies], 'repetitions':repetitions,
        'warmup':warmup, 'python':sys.version, 'numpy':np.__version__, 'scipy':scipy.__version__,
        'platform':platform.platform(), 'processor':platform.processor(),
        'scope':'One cached independent waveform per condition; repetitions measure timing, not independent BLER',
        'completed':False}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    rng_order = np.random.default_rng(1134)
    for condition_id, (users, rho) in enumerate([(16,0.), (16,.7), (32,0.), (32,.7)]):
        cfg = OFDMConfig(64, users)
        codec = LDPCCodec(LDPCConfig())
        rng = np.random.default_rng(np.random.SeedSequence([1133,condition_id]))
        payloads = rng.integers(0,2,(users,1024),dtype=np.uint8)
        encoded = np.stack([codec.encode(p) for p in payloads])
        grid, valid = make_grid(modulate(encoded,16),cfg)
        pdp = exponential_pdp(8)
        taps = sample_taps(rng,cfg,pdp,rho)
        waveform = time_channel(modulate_grid(grid,cfg),taps,10**(-12/10),rng)
        entries = [(p.name,p,'native') for p in policies]
        entries += [('exact-lapack-native',DetectorPolicy('exact-lapack-native',backend='lapack_cholesky'),'native')]
        entries += [('exact-scipy-solve',DetectorPolicy(),'solve'),
                    ('exact-scipy-lapack',DetectorPolicy(),'lapack')]
        with (output/'timings.jsonl').open('a') as handle:
            for repetition in range(-warmup,repetitions):
                for index in rng_order.permutation(len(entries)):
                    name,policy,implementation = entries[index]
                    start = perf_counter()
                    frequency = demodulate_waveform(waveform,cfg)
                    estimated = estimate_from_pilots(frequency,cfg,pdp)
                    estimation_seconds = perf_counter()-start
                    if implementation == 'native':
                        detection = detect_soft(estimated['channel_mean'],frequency[:,1:],
                            estimated['effective_variance'],policy,16,valid)
                    else:
                        detection = detect_soft_reference(estimated['channel_mean'],frequency[:,1:],
                            estimated['effective_variance'],policy,16,valid,diagonal_backend=implementation)
                    tick = perf_counter()
                    decoded = [codec.decode(llr) for llr in detection['llr']]
                    decoder_seconds = perf_counter()-tick
                    receiver_seconds = perf_counter()-start
                    if repetition < 0:
                        continue
                    errors = np.sum(np.stack([d[0] for d in decoded]) != payloads,axis=1)
                    row = {'condition_id':condition_id,'users':users,'rho':rho,'snr_db':12,
                        'repetition':repetition,'method':name,'implementation':implementation,
                        'estimation_seconds':estimation_seconds,
                        **{key:detection[key] for key in ['preparation_seconds','solve_seconds',
                            'soft_calibration_seconds','llr_seconds','detection_total_seconds']},
                        'decoder_seconds':decoder_seconds,'receiver_seconds':receiver_seconds,
                        'block_errors':int(np.count_nonzero(errors)),
                        'per_user_bit_errors':errors.tolist()}
                    rows.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
                if repetition >= 0 and repetition%5==0:
                    print(f'condition {condition_id}, repetition {repetition+1}/{repetitions}',flush=True)
    csv_rows = [{k:v for k,v in r.items() if k!='per_user_bit_errors'} for r in rows]
    with (output/'timings.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    metadata.update(completed=True,rows=len(rows))
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=25)
    args=parser.parse_args();run(args.output,args.repetitions)
