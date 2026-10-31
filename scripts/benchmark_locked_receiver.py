#!/usr/bin/env python3
"""Serial held-out full-receiver timings for every frozen candidate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,itertools,json,platform
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from mimo_receiver.coding import LDPCCodec,LDPCConfig
from mimo_receiver.qam import modulate
from mimo_receiver.ofdm import (OFDMConfig,exponential_pdp,sample_taps,make_grid,
    modulate_grid,time_channel,demodulate_waveform,estimate_from_pilots)
from mimo_receiver.receiver import DetectorPolicy,detect_soft
from run_locked_study import source_check

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(output):
    lock_path=ROOT/'research/protocols/study_002_lock.json'
    lock=json.loads(lock_path.read_text());source_check(lock)
    selection=json.loads((ROOT/lock['calibration_output']/'selection.json').read_text())
    assert selection['lock_sha256']==sha(lock_path)
    test=json.loads((ROOT/lock['test_output']/'metadata.json').read_text())
    if not test['completed']:raise ValueError('Complete independent reliability test required')
    output.mkdir(parents=True,exist_ok=False)
    settings=lock['settings'];policies=[DetectorPolicy(**p) for p in settings['policies']]
    conditions=list(itertools.product(settings['antenna_user_pairs'],settings['correlations'],settings['snr_db']))
    design={'seed_root':3150,'ordering_seed':4150,'frames_per_condition':4,
            'repetitions':5,'warmup':3,'method_order':'independent random permutation per replay',
            'frame_order':'fixed random permutation of 48 balanced source-condition frames'}
    metadata={'stage':'heldout_serial_complete_receiver_timing','lock_sha256':sha(lock_path),
        'design':design,'selected':selection['selections'][str(lock['primary_alpha'])]['selected'],
        'source_hashes':{**lock['source_hashes'],'scripts/benchmark_locked_receiver.py':sha(Path(__file__))},
        'platform':platform.platform(),'python':sys.version,
        'independent_frames':48,'timing_observations_per_method':240,
        'scope':'All receiver components from FFT through CRC, including native/Python conversion and dispatch',
        'completed':False}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for source in metadata['source_hashes']:
        target=output/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    rng_order=np.random.default_rng(design['ordering_seed'])
    frame_tasks=[(i,d) for i in range(len(conditions)) for d in range(design['frames_per_condition'])]
    tasks=[frame_tasks[i] for i in rng_order.permutation(len(frame_tasks))]
    codec=LDPCCodec(LDPCConfig())
    path=output/'timings.jsonl';total_rows=0
    with path.open('w') as handle:
        for frame_number,(condition_id,draw) in enumerate(tasks):
            (antennas,users),rho,snr=conditions[condition_id]
            cfg=OFDMConfig(antennas,users,settings['fft_size'],settings['cyclic_prefix'],settings['taps'])
            rng=np.random.default_rng(np.random.SeedSequence([design['seed_root'],condition_id,draw]))
            payloads=rng.integers(0,2,(users,1024),dtype=np.uint8)
            encoded=np.stack([codec.encode(p) for p in payloads])
            grid,valid=make_grid(modulate(encoded,16),cfg)
            pdp=exponential_pdp(cfg.assumed_taps);taps=sample_taps(rng,cfg,pdp,rho)
            waveform=time_channel(modulate_grid(grid,cfg),taps,10**(-snr/10),rng)
            known_errors={}
            for repetition in range(-design['warmup'],design['repetitions']):
                for index in rng_order.permutation(len(policies)):
                    policy=policies[index]
                    start=perf_counter()
                    frequency=demodulate_waveform(waveform,cfg)
                    estimated=estimate_from_pilots(frequency,cfg,pdp)
                    estimation_seconds=perf_counter()-start
                    detection=detect_soft(estimated['channel_mean'],frequency[:,1:],
                        estimated['effective_variance'],policy,16,valid)
                    tick=perf_counter();decoded=[codec.decode(v) for v in detection['llr']]
                    decoder_seconds=perf_counter()-tick
                    receiver_seconds=perf_counter()-start
                    errors=np.sum(np.stack([d[0] for d in decoded])!=payloads,axis=1).tolist()
                    if policy.name in known_errors:
                        assert errors==known_errors[policy.name], 'Repeated waveform changed decoding'
                    else:known_errors[policy.name]=errors
                    if repetition<0:continue
                    row={'frame_id':f'{condition_id}:{draw}','condition_id':condition_id,'draw':draw,
                        'frame_order':frame_number,'users':users,'rho':rho,'snr_db':snr,
                        'repetition':repetition,'method':policy.name,
                        'estimation_seconds':estimation_seconds,
                        **{k:detection[k] for k in ['preparation_seconds','solve_seconds',
                            'soft_calibration_seconds','llr_seconds','detection_total_seconds']},
                        'decoder_seconds':decoder_seconds,'receiver_seconds':receiver_seconds,
                        'block_errors':sum(e>0 for e in errors),'per_user_bit_errors':errors}
                    handle.write(json.dumps(row)+'\n');total_rows+=1
                handle.flush()
            print(f'{frame_number+1}/48 heldout timing frames completed',flush=True)
    source_check(lock)
    metadata.update(completed=True,rows=total_rows,records_sha256=sha(path))
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Serial benchmark completed',total_rows,'rows',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/final/receiver_timing_002')
    run(p.parse_args().output)
