#!/usr/bin/env python3
"""Create a prospective lock; never overwrites an existing study lock."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mimo_receiver import _core


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'research/protocols/study_002_lock.json'
    if output.exists():raise FileExistsError('Existing prospective lock is immutable')
    config=ROOT/'configs/joint_budget_development.json'
    settings=json.loads(config.read_text())
    folder=ROOT/'results/pilot/joint_budget_development_002'
    meta=json.loads((folder/'metadata.json').read_text())
    if not meta['completed'] or meta['config_sha256']!=sha(config):
        raise ValueError('Completed matching development data required')
    records=[json.loads(l) for l in (folder/'channel_records.jsonl').read_text().splitlines()]
    if len(records)!=48:raise ValueError('Expected 48 balanced development frames')
    names=[p['name'] for p in settings['policies']]
    costs={name:mean(next(r['receiver_seconds'] for r in f['rows'] if r['method']==name)
                     for f in records) for name in names}
    sources=['cpp/core.cpp','scripts/run_locked_study.py','scripts/run_soft_output_screen.py',
             'scripts/freeze_joint_study.py','setup.py','pyproject.toml','requirements-lock.txt',
             'configs/joint_budget_development.json','docs/PROTOCOL_002.md']
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'src/mimo_receiver').glob('*.py')]
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('*.py')]
    lock={'protocol':'joint-budget-study-v0.2','created_utc':datetime.now(timezone.utc).isoformat(),
        'settings':settings,'source_hashes':{p:sha(ROOT/p) for p in sources},
        'extension_sha256':sha(Path(_core.__file__)),'workers':4,
        'development_costs':costs,'development_frames':48,
        'development_records_sha256':sha(folder/'channel_records.jsonl'),
        'primary_alpha':0.01,'reported_alpha':[0.002,0.005,0.01,0.02],'delta':0.05,
        'candidate_count':len(names)-1,
        'stages':{'calibration':{'seed_root':2100,'frames':4096},
                  'test':{'seed_root':3100,'frames':2048}},
        'calibration_output':'results/calibration/joint_budget_002',
        'test_output':'results/final/joint_budget_002',
        'estimand':'uniform source-frame mixture, then uniform audit user; one-sided excess decoded BLER',
        'status':'prospective, created before either independent stage'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(lock,indent=2))
    print('Wrote',output,'sha256',sha(output))
    print('Candidate count',lock['candidate_count'],'cheapest development pipeline',min(costs,key=costs.get))


if __name__=='__main__':main()
