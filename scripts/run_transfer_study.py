#!/usr/bin/env python3
"""Frozen source pipelines on explicit channel/prior/modulation shifts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,concurrent.futures,hashlib,json,multiprocessing
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_locked_study import source_check
from run_soft_output_screen import simulate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run_task(task):
    settings,condition,arm_id,rho_id,draw=task
    record=simulate((settings,condition,rho_id,draw))
    record.update(task_id=f'{arm_id}:{rho_id}:{draw}',arm_id=arm_id,
        cluster_id=f'{rho_id}:{draw}')
    return record

def run(output):
    lock_path=ROOT/'research/protocols/study_002_lock.json';lock=json.loads(lock_path.read_text())
    source_check(lock)
    arms=[
      {'name':'TDL-A-100-source','channel_profile':'TDL-A','rms_delay_ns':100,'receiver_prior':'source','users':32,'snr_db':12,'qam_order':16},
      {'name':'TDL-C-100-source','channel_profile':'TDL-C','rms_delay_ns':100,'receiver_prior':'source','users':32,'snr_db':12,'qam_order':16},
      {'name':'TDL-A-100-matched','channel_profile':'TDL-A','rms_delay_ns':100,'receiver_prior':'matched','users':32,'snr_db':12,'qam_order':16},
      {'name':'TDL-C-100-matched','channel_profile':'TDL-C','rms_delay_ns':100,'receiver_prior':'matched','users':32,'snr_db':12,'qam_order':16},
      {'name':'TDL-A-300-source','channel_profile':'TDL-A','rms_delay_ns':300,'receiver_prior':'source','users':32,'snr_db':12,'qam_order':16},
      {'name':'TDL-C-300-source','channel_profile':'TDL-C','rms_delay_ns':300,'receiver_prior':'source','users':32,'snr_db':12,'qam_order':16},
      {'name':'Exponential-64QAM','channel_profile':'exponential','receiver_prior':'source','users':32,'snr_db':18,'qam_order':64},
      {'name':'TDL-A-100-64QAM','channel_profile':'TDL-A','rms_delay_ns':100,'receiver_prior':'source','users':32,'snr_db':18,'qam_order':64},
      {'name':'TDL-A-300-matched-64QAM','channel_profile':'TDL-A','rms_delay_ns':300,'receiver_prior':'matched','users':16,'snr_db':16,'qam_order':64},
      {'name':'TDL-C-300-matched-64QAM','channel_profile':'TDL-C','rms_delay_ns':300,'receiver_prior':'matched','users':16,'snr_db':16,'qam_order':64}]
    names=['exact','direct-i32','direct-i8','pcg-diagonal','yin8']
    policies=[p for p in lock['settings']['policies'] if p['name'] in names]
    metadata={'stage':'heldout_distribution_shift_evaluation','source_lock_sha256':sha(lock_path),
        'arms':arms,'correlations':[0.,.7],'draws_per_arm_correlation':128,'seed_root':5100,
        'policies':policies,'workers':4,'sampling_rate_hz':7.68e6,
        'sampling_unit':'Independent draw within each arm/correlation; common random streams across arms retained as cluster_id',
        'certificate_scope':'No source-mixture statistical guarantee is asserted for these shifted arms',
        'timing_scope':'Concurrent reliability study, no speed claim',
        'source_hashes':{**lock['source_hashes'],'scripts/run_transfer_study.py':sha(Path(__file__))},
        'completed':False}
    output.mkdir(parents=True,exist_ok=True);meta_path=output/'metadata.json';path=output/'channel_records.jsonl'
    if meta_path.exists():
        previous=json.loads(meta_path.read_text())
        for key in ('source_hashes','arms','policies','seed_root'):
            if previous[key]!=metadata[key]:raise ValueError('Transfer output/source mismatch')
    else:
        meta_path.write_text(json.dumps(metadata,indent=2))
        for source in metadata['source_hashes']:
            target=output/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/source).read_bytes())
    records=[json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    done={r['task_id'] for r in records};assert len(done)==len(records)
    tasks=[]
    for arm_id,arm in enumerate(arms):
        settings={**lock['settings'],**arm,'seed_root':5100,'policies':policies}
        for rho_id,rho in enumerate(metadata['correlations']):
            for draw in range(128):
                if f'{arm_id}:{rho_id}:{draw}' not in done:
                    tasks.append((settings,([64,arm['users']],rho,arm['snr_db']),arm_id,rho_id,draw))
    print(f'{len(records)}/2560 saved, {len(tasks)} remaining',flush=True)
    with path.open('a') as handle:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as executor:
            for record in executor.map(run_task,tasks,chunksize=1):
                records.append(record);handle.write(json.dumps(record)+'\n');handle.flush()
                if len(records)%32==0:print(f'{len(records)}/2560 transfer frames saved',flush=True)
    assert len(records)==2560
    source_check(lock)
    metadata.update(completed=True,frames=len(records),rows=sum(len(r['rows']) for r in records),records_sha256=sha(path))
    meta_path.write_text(json.dumps(metadata,indent=2));print('Transfer study completed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/final/transfer_002')
    run(p.parse_args().output)
