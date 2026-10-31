#!/usr/bin/env python3
"""Fixed-size prospective calibration/test runner with source/extension lock."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import concurrent.futures
from dataclasses import asdict
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
from mimo_receiver import _core
from mimo_receiver.risk import select_policy
from run_soft_output_screen import simulate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def task_for_draw(settings,draw):
    conditions=list(itertools.product(settings['antenna_user_pairs'],settings['correlations'],settings['snr_db']))
    rng=np.random.default_rng(np.random.SeedSequence([settings['seed_root'],85197,draw]))
    index=int(rng.integers(len(conditions)))
    return settings,conditions[index],index,draw


def source_check(lock,allow_rebuilt_extension=False):
    for path,digest in lock['source_hashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError(f'Locked source changed: {path}')
    if sha(Path(_core.__file__))!=lock['extension_sha256'] and not allow_rebuilt_extension:
        raise ValueError('Locked compiled extension changed')


def calibrate(records,lock,lock_sha,folder):
    events={p['name']:[] for p in lock['settings']['policies'] if p['name']!='exact'}
    for frame in records:
        for row in frame['rows']:
            if row['method']!='exact':events[row['method']].append(int(row['audit_user_harm']))
    output={'lock_sha256':lock_sha,'frames':len(records),'reference':'exact',
        'primary_alpha':lock['primary_alpha'],'delta':lock['delta'],
        'family_size':len(events),'event_counts':{k:sum(v) for k,v in events.items()},
        'costs_from_development':lock['development_costs'],'selections':{}}
    for alpha in lock['reported_alpha']:
        choice=select_policy(events,lock['development_costs'],alpha,lock['delta'])
        output['selections'][str(alpha)]=asdict(choice)
    (folder/'selection.json').write_text(json.dumps(output,indent=2))
    print('Primary selected policy:',output['selections'][str(lock['primary_alpha'])]['selected'],flush=True)


def run(lock_path,stage,output,allow_rebuilt_extension=False):
    lock=json.loads(lock_path.read_text());lock_sha=sha(lock_path)
    source_check(lock,allow_rebuilt_extension)
    if stage not in ('calibration','test'):raise ValueError('Unknown study stage')
    spec=lock['stages'][stage]
    if stage=='test':
        selection_path=ROOT/lock['calibration_output']/'selection.json'
        selection=json.loads(selection_path.read_text())
        if selection['lock_sha256']!=lock_sha:raise ValueError('Test requires completed selection for this exact lock')
    settings={**lock['settings'],'seed_root':spec['seed_root'],'stage':stage}
    output.mkdir(parents=True,exist_ok=True)
    meta_path=output/'metadata.json';record_path=output/'channel_records.jsonl'
    metadata={'stage':stage,'lock_sha256':lock_sha,'lock':lock,
        'settings':settings,'sample_size':spec['frames'],'workers':lock['workers'],
        'timing_scope':'Concurrent reliability run; timings not used for publication speed claims',
        'independent_unit':'one generated channel/frame; one uniform audit user chosen before outcomes',
        'condition_distribution':'iid uniform over the twelve declared source cells',
        'runtime_extension_sha256':sha(Path(_core.__file__)),
        'reproduction_with_rebuilt_extension':allow_rebuilt_extension,
        'completed':False}
    if meta_path.exists():
        prior=json.loads(meta_path.read_text())
        if (prior['lock_sha256']!=lock_sha or prior['stage']!=stage or
                prior['runtime_extension_sha256']!=metadata['runtime_extension_sha256']):
            raise ValueError('Output belongs to another locked study')
    else:
        meta_path.write_text(json.dumps(metadata,indent=2))
        for source in lock['source_hashes']:
            target=output/'source_snapshot'/source
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/source).read_bytes())
    records=[json.loads(l) for l in record_path.read_text().splitlines() if l.strip()] if record_path.exists() else []
    done={r['draw'] for r in records}
    if len(done)!=len(records) or any(d<0 or d>=spec['frames'] for d in done):
        raise ValueError('Duplicate or unexpected frame IDs')
    tasks=[task_for_draw(settings,d) for d in range(spec['frames']) if d not in done]
    print(f'{stage}: {len(done)}/{spec["frames"]} frames saved; {len(tasks)} remaining',flush=True)
    tick=perf_counter()
    with record_path.open('a') as handle:
        with concurrent.futures.ProcessPoolExecutor(max_workers=lock['workers'],
                mp_context=multiprocessing.get_context('spawn')) as executor:
            for record in executor.map(simulate,tasks,chunksize=1):
                records.append(record);handle.write(json.dumps(record)+'\n');handle.flush()
                if len(records)%32==0:
                    print(f'{stage}: {len(records)}/{spec["frames"]}',flush=True)
    if len(records)!=spec['frames']:raise RuntimeError('Fixed sample size not reached')
    source_check(lock,allow_rebuilt_extension)
    metadata.update(completed=True,frames=len(records),rows=sum(len(r['rows']) for r in records),
                    invocation_seconds=perf_counter()-tick,records_sha256=sha(record_path))
    meta_path.write_text(json.dumps(metadata,indent=2))
    if stage=='calibration':calibrate(records,lock,lock_sha,output)
    print(stage,'completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--lock',type=Path,default=ROOT/'research/protocols/study_002_lock.json')
    parser.add_argument('--stage',choices=['calibration','test'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-rebuilt-extension',action='store_true',
        help='Reproduction only: allow a rebuilt binary with exactly identical locked source')
    args=parser.parse_args();run(args.lock,args.stage,args.output,args.allow_rebuilt_extension)
