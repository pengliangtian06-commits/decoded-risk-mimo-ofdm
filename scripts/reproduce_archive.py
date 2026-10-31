#!/usr/bin/env python3
"""Run from a fresh extraction with the pinned Python environment already active.

Rebuilds the native extension, runs all tests, reconstructs development ranking
and calibration, replays fixed outcomes and regenerates all principal analyses.
This is same-platform reproduction; absolute benchmark timings are not replayed.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(output):
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    import numpy as np
    from run_locked_study import source_check,calibrate
    from mimo_receiver import _core
    import mimo_receiver.receiver as receiver
    assert Path(_core.__file__).resolve().is_relative_to(ROOT)
    assert Path(receiver.__file__).resolve().is_relative_to(ROOT)
    lock_path=ROOT/'research/protocols/study_002_lock.json';lock=json.loads(lock_path.read_text());source_check(lock,allow_rebuilt_extension=True)
    raw=ROOT/'results/pilot/joint_budget_development_002/channel_records.jsonl'
    assert sha(raw)==lock['development_records_sha256']
    dev=[json.loads(l) for l in raw.read_text().splitlines()]
    for name,expected in lock['development_costs'].items():
        actual=np.mean([next(r for r in f['rows'] if r['method']==name)['receiver_seconds'] for f in dev])
        assert abs(actual-expected)<1e-14,(name,actual,expected)
    cal=[json.loads(l) for l in (ROOT/'results/calibration/joint_budget_002/channel_records.jsonl').read_text().splitlines()]
    regenerated=output/'calibration';regenerated.mkdir()
    calibrate(cal,lock,sha(lock_path),regenerated)
    original=json.loads((ROOT/'results/calibration/joint_budget_002/selection.json').read_text())
    assert json.loads((regenerated/'selection.json').read_text())==original
    (output/'source_check.json').write_text(json.dumps({'passed':True,'locked_sources':len(lock['source_hashes']),
        'rebuilt_extension':True,'extension_within_extracted_archive':True,
        'development_costs_reconstructed':True,'calibration_reconstructed':True},indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/reproduced_archive_001')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args();output=a.output.resolve()
    if a.verify_only:verify(output);return
    output.mkdir(parents=True,exist_ok=False)
    bundled=ROOT/'dependencies/eigen-3.4.0.tar.gz'
    if bundled.exists():
        (ROOT/'.cache').mkdir(exist_ok=True);shutil.copyfile(bundled,ROOT/'.cache/eigen-3.4.0.tar.gz')
    commands=[
        [sys.executable,'scripts/bootstrap_dependencies.py'],
        [sys.executable,'setup.py','build_ext','--inplace'],
        [sys.executable,'-m','pytest','-q'],
        [sys.executable,'scripts/reproduce_archive.py','--output',str(output),'--verify-only'],
        [sys.executable,'scripts/verify_locked_replay.py','--reference','results/reproduction/clean_release_002/reference_subset.json','--output',str(output/'replay.json')],
        [sys.executable,'scripts/analyze_locked_study.py','--output',str(output/'source_analysis')],
        [sys.executable,'scripts/analyze_receiver_timing.py','--output',str(output/'timing_analysis')],
        [sys.executable,'scripts/analyze_transfer_study.py','--output',str(output/'transfer_analysis')]]
    steps=[]
    for i,cmd in enumerate(commands,1):
        print('Archive reproduction step',i,cmd[1:3],flush=True)
        with (output/f'step_{i}.txt').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        steps.append({'command':cmd[1:],'returncode':r.returncode})
        if r.returncode:
            (output/'report.json').write_text(json.dumps({'passed':False,'steps':steps},indent=2));raise RuntimeError('Step failed '+str(i))
    comparisons={}
    for fresh,original in [('source_analysis','joint_budget_002'),('timing_analysis','receiver_timing_002'),('transfer_analysis','transfer_002')]:
        actual=json.loads((output/fresh/'analysis.json').read_text());expected=json.loads((ROOT/'results/analysis'/original/'analysis.json').read_text())
        assert actual==expected,(fresh,'Analysis differs');comparisons[fresh]='all fields exactly equal'
    report={'passed':True,'scope':'fresh-extraction source build, pinned same-platform environment; no original project sources used',
        'independent_machine_replication':False,'steps':steps,'analysis_comparisons':comparisons,
        'source_check':json.loads((output/'source_check.json').read_text()),'replay':json.loads((output/'replay.json').read_text())}
    (output/'report.json').write_text(json.dumps(report,indent=2));print('Archive reproduction passed.',flush=True)
if __name__=='__main__':main()
