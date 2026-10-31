#!/usr/bin/env python3
"""Audit independent sampling, paired counts, provenance and exact risk limits."""
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scipy.stats import binom
from run_locked_study import task_for_draw

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(folder):
    meta=json.loads((folder/'metadata.json').read_text())
    if not meta['completed']:raise ValueError('Stage is still running')
    lock=meta['lock'];path=folder/'channel_records.jsonl'
    assert sha(path)==meta['records_sha256']
    frames=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(frames)==meta['sample_size']==meta['frames']
    assert {f['draw'] for f in frames}==set(range(meta['sample_size']))
    names={p['name'] for p in lock['settings']['policies']}
    event_counts=Counter();cell_counts=Counter();blocks=0
    for f in frames:
        expected=task_for_draw(meta['settings'],f['draw'])
        assert f['condition_id']==expected[2]
        assert [f['antennas'],f['users']]==expected[1][0]
        assert f['rho']==expected[1][1] and f['snr_db']==expected[1][2]
        cell_counts[f['condition_id']]+=1;blocks+=f['users']
        rows={r['method']:r for r in f['rows']}
        assert set(rows)==names and len(rows)==len(f['rows'])
        ref=rows['exact']['per_user_bit_errors']
        audit_users={r['audit_user'] for r in rows.values()}
        assert len(audit_users)==1
        for name,r in rows.items():
            errors=r['per_user_bit_errors'];j=r['audit_user']
            assert len(errors)==r['blocks']==f['users']
            assert all(isinstance(e,int) and 0<=e<=1024 for e in errors)
            assert r['bit_errors']==sum(errors)
            assert r['block_errors']==sum(e>0 for e in errors)
            assert r['extra_errors']==sum(e>0 and q==0 for e,q in zip(errors,ref))
            assert 0<=j<f['users']
            assert r['audit_user_error']==(errors[j]>0)
            assert r['audit_user_harm']==(errors[j]>0 and ref[j]==0)
            assert sum(r['stop_reasons'].values())==r['vectors']
            assert 0<=r['crc_failures']<=f['users']
            event_counts[name]+=r['audit_user_harm']
    report={'stage':meta['stage'],'record_sha256_matches':True,'frames':len(frames),
        'nested_user_blocks_per_method':blocks,'methods':len(names),
        'frame_condition_counts':dict(sorted(cell_counts.items())),
        'paired_per_user_counts_valid':True,'sample_ids_match_frozen_generator':True,
        'audit_user_harm_events':dict(event_counts)}
    if meta['stage']=='calibration':
        selection=json.loads((folder/'selection.json').read_text())
        assert selection['lock_sha256']==meta['lock_sha256']
        for name,count in selection['event_counts'].items():assert count==event_counts[name]
        primary=selection['selections'][str(selection['primary_alpha'])]
        discrepancy=[]
        for name,upper in primary['upper_limits'].items():
            count=event_counts[name]
            if count<len(frames):
                discrepancy.append(abs(binom.cdf(count,len(frames),upper)-selection['delta']/selection['family_size']))
        assert max(discrepancy)<1e-10
        report.update(exact_bound_binomial_tail_max_error=max(discrepancy),
                      primary_selected=primary['selected'],
                      selected_upper=primary['upper_limits'].get(primary['selected'],0.))
    (folder/'AUDIT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    audit(parser.parse_args().folder)
