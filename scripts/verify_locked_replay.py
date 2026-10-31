#!/usr/bin/env python3
"""Replay a fixed source-test subset from a clean, rebuilt source snapshot."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from run_locked_study import source_check,task_for_draw
from run_soft_output_screen import simulate

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.reference.read_text());lock=json.loads((ROOT/'research/protocols/study_002_lock.json').read_text())
    source_check(lock,allow_rebuilt_extension=True)
    deterministic=['blocks','block_errors','bit_errors','per_user_bit_errors','audit_user',
        'audit_user_error','crc_failures','decoder_iterations','uncoded_errors','uncoded_bits',
        'iterations','vectors','iteration_p95','stop_reasons','exact_diagonal_tones','gain_clip_count',
        'extra_errors','audit_user_harm']
    checks=[]
    for expected in data['frames']:
        actual=simulate(task_for_draw(data['settings'],expected['draw']))
        assert actual['condition_id']==expected['condition_id']
        assert len(actual['rows'])==len(expected['rows'])==17
        for er,ar in zip(expected['rows'],actual['rows'],strict=True):
            assert er['method']==ar['method']
            for field in deterministic:
                assert er[field]==ar[field],(expected['draw'],er['method'],field,er[field],ar[field])
            for field in ('gain_absolute_error_mean','gain_absolute_error_max'):
                assert np.isclose(er[field],ar[field],rtol=1e-11,atol=1e-13),(er['method'],field)
        for field in ('estimated_noise','true_noise'):
            assert np.isclose(expected[field],actual[field],rtol=1e-12,atol=1e-14)
        checks.append({'draw':actual['draw'],'condition_id':actual['condition_id'],
            'users':actual['users'],'all_policy_outcomes_exact':True})
        print('Reproduced frame',actual['draw'],'condition',actual['condition_id'],flush=True)
    source_check(lock,allow_rebuilt_extension=True)
    a.output.write_text(json.dumps({'passed':True,'frames':checks,
        'policies_per_frame':17,'integer_fields_checked':deterministic,
        'gain_comparison_tolerance':{'rtol':1e-11,'atol':1e-13},
        'timing_values_compared':False},indent=2))

if __name__=='__main__':main()
