#!/usr/bin/env python3
"""Resumable channel-paired coded screening, development seeds only."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse
import concurrent.futures
import csv
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from mimo_receiver import _core
from mimo_receiver.coding import LDPCCodec,LDPCConfig
from mimo_receiver.qam import modulate,maxlog_llr
from mimo_receiver.ofdm import (OFDMConfig,exponential_pdp,sample_taps,make_grid,
    modulate_grid,time_channel,demodulate_waveform,estimate_from_pilots)


def simulate(task):
    settings,condition,condition_id,seed=task
    (antennas,users),rho,snr=condition
    cfg=OFDMConfig(antennas=antennas,users=users,fft_size=settings['fft_size'],
                   cyclic_prefix=settings['cyclic_prefix'],assumed_taps=settings['taps'])
    codec=LDPCCodec(LDPCConfig(qam_order=settings['qam_order'],max_iterations=settings['decoder_iterations']))
    rng=np.random.default_rng(np.random.SeedSequence([seed,condition_id]))
    payloads=rng.integers(0,2,(users,1024),dtype=np.uint8)
    encoded=np.stack([codec.encode(payload) for payload in payloads])
    grid,valid=make_grid(modulate(encoded,settings['qam_order']),cfg)
    pdp=exponential_pdp(cfg.assumed_taps)
    taps=sample_taps(rng,cfg,pdp,rho)
    noise=10**(-snr/10)
    waveform=time_channel(modulate_grid(grid,cfg),taps,noise,rng)
    rx_start=perf_counter()
    received=demodulate_waveform(waveform,cfg)
    estimated=estimate_from_pilots(received,cfg,pdp)
    h=estimated['channel_mean'];lam=estimated['effective_variance']
    estimation_seconds=perf_counter()-rx_start
    data_symbols=received.shape[1]-1
    identity=np.eye(users)
    reference=np.empty((users,data_symbols,cfg.fft_size),complex)
    gains=np.empty((users,cfg.fft_size));variances=np.empty_like(gains)
    calibration_start=perf_counter()
    for tone in range(cfg.fft_size):
        mean=h[:,:,tone]
        factor=cho_factor(mean.conj().T@mean+lam*identity,lower=True,check_finite=False)
        reference[:,:,tone]=cho_solve(factor,mean.conj().T@received[:,1:,tone],check_finite=False)
        diagonal=cho_solve(factor,identity,check_finite=False).diagonal().real
        gain=np.maximum(1-lam*diagonal,1e-10)
        gains[:,tone]=gain;variances[:,tone]=np.maximum(1/gain-1,1e-12)
    calibration_seconds=perf_counter()-calibration_start
    true_h=np.fft.fft(taps,n=cfg.fft_size,axis=-1)
    rows=[]
    for method in settings['methods']:
        detector_start=perf_counter()
        iteration_sum=0
        if method['mode']=='exact':
            detected=reference.copy()
        else:
            detected=np.empty_like(reference)
            for tone in range(cfg.fft_size):
                prepared=_core.PreparedSystem(h[:,:,tone],lam,'pcg')
                result=prepared.solve(received[:,1:,tone],method['mode'],method['threshold'],
                                      2*users,2,settings['qam_order'])
                detected[:,:,tone]=result['symbols']
                iteration_sum+=sum(result['iterations'])
        solver_seconds=perf_counter()-detector_start
        unbiased=(detected/gains[:,None,:]).reshape(users,-1)[:,:valid]
        variance=np.broadcast_to(variances[:,None,:],detected.shape).reshape(users,-1)[:,:valid]
        llr=maxlog_llr(unbiased,variance,settings['qam_order'])
        decode_start=perf_counter()
        outputs=[codec.decode(values) for values in llr]
        decoding_seconds=perf_counter()-decode_start
        decoded=np.stack([item[0] for item in outputs])
        bit_counts=np.sum(decoded!=payloads,axis=1)
        rows.append({'condition_id':condition_id,'seed':seed,'antennas':antennas,'users':users,
            'rho':rho,'snr_db':snr,'method':method['name'],'mode':method['mode'],
            'threshold':method['threshold'],'blocks':users,'block_errors':int(np.count_nonzero(bit_counts)),
            'information_bits':int(payloads.size),'bit_errors':int(bit_counts.sum()),
            'crc_failures':sum(item[1] for item in outputs),'detected_vectors':cfg.fft_size*data_symbols,
            'iterations':iteration_sum,'decoder_iterations':sum(item[2] for item in outputs),
            'estimated_noise':estimated['noise_variance'],'true_noise':noise,
            'estimated_channel_mse':estimated['channel_error_variance_per_entry'],
            'actual_channel_mse':float(np.mean(abs(h-true_h)**2)),
            'estimation_seconds':estimation_seconds,'shared_exact_calibration_seconds':calibration_seconds,
            'diagnostic_solver_seconds':solver_seconds,'decoding_seconds':decoding_seconds,
            'airtime_samples':waveform.shape[1],
            'correct_information_bits':int(1024*np.sum(bit_counts==0))})
    return {'task_id':f'{condition_id}:{seed}','rows':rows}


def run(config_path,output):
    raw=config_path.read_bytes();settings=json.loads(raw)
    if settings['stage']!='exploratory_development' or not 1100<=settings['seed_start']<1200 or settings['seed_start']+settings['seeds']>1200:
        raise ValueError('This screening runner may use development seed family only')
    output.mkdir(parents=True,exist_ok=True)
    metadata_path=output/'metadata.json';records_path=output/'channel_records.jsonl'
    digest=hashlib.sha256(raw).hexdigest()
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        if metadata['config_sha256']!=digest:
            raise ValueError('Output directory belongs to another configuration')
    else:
        metadata={'config':settings,'config_sha256':digest,'stage':'exploratory',
                  'code_sha256':hashlib.sha256((ROOT/'cpp/core.cpp').read_bytes()).hexdigest(),
                  'completed':False,'reliability_only':True,
                  'timing_limit':'Exact posterior-diagonal soft calibration is shared; timing is not a production speedup comparison'}
        metadata_path.write_text(json.dumps(metadata,indent=2))
    existing=[]
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():existing.append(json.loads(line))
    done={item['task_id'] for item in existing}
    conditions=list(itertools.product(settings['antenna_user_pairs'],settings['correlations'],settings['snr_db']))
    tasks=[(settings,condition,index,seed) for index,condition in enumerate(conditions)
           for seed in range(settings['seed_start'],settings['seed_start']+settings['seeds'])
           if f'{index}:{seed}' not in done]
    started=perf_counter()
    print(f'{len(done)} tasks already saved; {len(tasks)} remaining; {settings["workers"]} workers',flush=True)
    context=multiprocessing.get_context('spawn')
    with records_path.open('a') as handle:
        with concurrent.futures.ProcessPoolExecutor(max_workers=settings['workers'],mp_context=context) as executor:
            futures=[executor.submit(simulate,task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                item=future.result();existing.append(item)
                handle.write(json.dumps(item)+'\n');handle.flush()
                print(f'{len(existing)}/{len(conditions)*settings["seeds"]} saved {item["task_id"]}',flush=True)
    rows=[row for item in existing for row in item['rows']]
    rows.sort(key=lambda row:(row['condition_id'],row['seed'],row['method']))
    with (output/'results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata.update(completed=True,channels=len(existing),rows=len(rows),last_invocation_seconds=perf_counter()-started)
    metadata_path.write_text(json.dumps(metadata,indent=2))
    print(f'Completed {len(existing)} paired channels, {len(rows)} method/channel records',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'configs/coded_screen.json')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.config,args.output)
