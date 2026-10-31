#!/usr/bin/env python3
"""Audit and describe shifted conditions, without transporting the source bound."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from mimo_receiver.risk import binomial_upper
from analyze_locked_study import mean_interval

def analyze(folder,output):
    if output.exists():raise FileExistsError('Refusing overwrite')
    meta=json.loads((folder/'metadata.json').read_text());assert meta['completed']
    path=folder/'channel_records.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['records_sha256']
    frames=[json.loads(l) for l in path.read_text().splitlines()]
    assert len(frames)==2560 and len({f['task_id'] for f in frames})==2560
    names={p['name'] for p in meta['policies']};groups=defaultdict(list)
    for f in frames:
        arm=meta['arms'][f['arm_id']];assert f['users']==arm['users'] and f['snr_db']==arm['snr_db']
        assert f['cluster_id']==f"{f['condition_id']}:{f['draw']}"
        rows={r['method']:r for r in f['rows']};assert set(rows)==names
        ref=rows['exact']['per_user_bit_errors']
        for r in rows.values():
            bits=r['per_user_bit_errors'];j=r['audit_user']
            assert len(bits)==f['users'] and sum(bits)==r['bit_errors']
            assert sum(e>0 for e in bits)==r['block_errors']
            assert sum(e>0 and q==0 for e,q in zip(bits,ref))==r['extra_errors']
            assert r['audit_user_harm']==(bits[j]>0 and ref[j]==0)
        groups[(f['arm_id'],f['rho'])].append(f)
    assert len(groups)==20 and all(len(v)==128 for v in groups.values())
    rng=np.random.default_rng(5150);rows=[]
    for (arm_id,rho),subset in sorted(groups.items()):
        subset.sort(key=lambda f:f['draw']);indices=rng.integers(0,128,(5000,128),dtype=np.int32)
        arm=meta['arms'][arm_id]
        reference=np.array([next(r['block_errors']/r['blocks'] for r in f['rows'] if r['method']=='exact') for f in subset])
        for name in sorted(names):
            records=[next(r for r in f['rows'] if r['method']==name) for f in subset]
            bler=np.array([r['block_errors']/r['blocks'] for r in records])
            harm=np.array([r['extra_errors']/r['blocks'] for r in records])
            audit=sum(r['audit_user_harm'] for r in records)
            b=mean_interval(bler,indices);h=mean_interval(harm,indices)
            difference=mean_interval(bler-reference,indices,support=(-1.,1.)) if name!='exact' else {'mean':0.,'lower':0.,'upper':0.}
            profile=subset[0]['profile_metadata'] or {}
            rows.append({'arm_id':arm_id,'arm':arm['name'],'rho':rho,'users':arm['users'],
                'qam_order':arm['qam_order'],'snr_db':arm['snr_db'],'method':name,
                'independent_frames_in_cell':128,'nested_user_blocks':128*arm['users'],
                'block_errors':sum(r['block_errors'] for r in records),'extra_errors':sum(r['extra_errors'] for r in records),
                'bler':b['mean'],'bler_lower':b['lower'],'bler_upper':b['upper'],
                'harm':h['mean'],'harm_lower':h['lower'],'harm_upper':h['upper'],
                'paired_bler_difference':difference['mean'],'difference_lower':difference['lower'],'difference_upper':difference['upper'],
                'audit_harm_events':audit,'audit_harm_upper_95':binomial_upper(audit,128,.05),
                'nominal_rms_ns':profile.get('nominal_rms_delay_ns'),
                'quantized_rms_ns':profile.get('quantized_rms_delay_ns'),
                'source_certificate_applies':False})
    output.mkdir(parents=True)
    with (output/'cell_results.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'stage':'heldout shift analysis','raw_records_sha256':meta['records_sha256'],
        'frames':2560,'distinct_shared_stream_clusters':len({f['cluster_id'] for f in frames}),
        'cells':20,'frames_per_cell':128,'methods':len(names),
        'scope':'Per-cell descriptive paired inference; common-stream arms are not pooled as independent samples.',
        'intervals':'95% frame-cluster percentile bootstrap with conservative nonzero limits for constant bounded observations; unadjusted across cells.',
        'certificate':'The source-mixture statistical certificate does not apply to any shifted arm.',
        'row_count':len(rows),'count_and_pairing_audit':'passed',
        'selected_policy_rows':[r for r in rows if r['method']=='direct-i32']}
    (output/'analysis.json').write_text(json.dumps(report,indent=2))
    for r in report['selected_policy_rows']:
        print(r['arm'],r['rho'],'BLER',round(r['bler'],4),'harm',round(r['harm'],4),
              'harm interval',round(r['harm_lower'],4),round(r['harm_upper'],4))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/final/transfer_002')
    p.add_argument('--output',type=Path,default=ROOT/'results/analysis/transfer_002')
    a=p.parse_args();analyze(a.input,a.output)
