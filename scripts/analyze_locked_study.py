#!/usr/bin/env python3
"""Paired independent-frame inference after the complete frozen test stage."""
import argparse,csv,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mimo_receiver.risk import binomial_upper


def mean_interval(values,indices,confidence=.95,support=(0.,1.)):
    values=np.asarray(values,float);n=len(values);point=float(values.mean())
    if np.all(values==values[0]):
        # Zero bootstrap dispersion is not proof of zero population uncertainty.
        # P(X differs from the observed constant) bounds a bounded mean shift.
        u=binomial_upper(0,n,1-confidence)
        c=float(values[0]);radius=max(c-support[0],support[1]-c)*u
        return {'mean':point,'lower':max(support[0],c-radius),
                'upper':min(support[1],c+radius),
                'method':'exact zero-event frame bound for constant bounded observations'}
    means=values[indices].mean(axis=1)
    low,high=np.quantile(means,[(1-confidence)/2,1-(1-confidence)/2])
    return {'mean':point,'lower':float(low),'upper':float(high),
            'method':'paired frame-cluster percentile bootstrap'}


def analyze(folder,output,repetitions=5000):
    if output.exists():raise FileExistsError('Refusing overwrite of analysis')
    meta=json.loads((folder/'metadata.json').read_text())
    if not meta['completed'] or meta['stage']!='test':raise ValueError('Complete independent test required')
    raw=folder/'channel_records.jsonl'
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=meta['records_sha256']:
        raise ValueError('Raw-record hash differs')
    frames=sorted([json.loads(l) for l in raw.read_text().splitlines()],key=lambda f:f['draw'])
    selection=json.loads((ROOT/meta['lock']['calibration_output']/'selection.json').read_text())
    assert selection['lock_sha256']==meta['lock_sha256']
    alpha=selection['primary_alpha'];selected=selection['selections'][str(alpha)]['selected']
    names=[p['name'] for p in meta['lock']['settings']['policies']]
    n=len(frames);rng=np.random.default_rng(4100)
    goodput_support=max(pair[1] for pair in meta['lock']['settings']['antenna_user_pairs'])*1024/min(f['airtime_samples'] for f in frames)
    indices=rng.integers(0,n,size=(repetitions,n),dtype=np.int32)
    arrays={};summaries=[]
    for name in names:
        rows=[next(r for r in f['rows'] if r['method']==name) for f in frames]
        errors=np.array([r['block_errors']/r['blocks'] for r in rows])
        ber=np.array([r['bit_errors']/(1024*r['blocks']) for r in rows])
        harm=np.array([r['extra_errors']/r['blocks'] for r in rows])
        audit=np.array([r['audit_user_harm'] for r in rows],int)
        goodput=np.array([1024*(r['blocks']-r['block_errors'])/f['airtime_samples'] for r,f in zip(rows,frames)])
        arrays[name]={'bler':errors,'ber':ber,'harm':harm,'audit_harm':audit,'goodput':goodput}
        summary={'method':name,'independent_frames':n,'nested_user_blocks':sum(r['blocks'] for r in rows),
            'block_errors':sum(r['block_errors'] for r in rows),'bit_errors':sum(r['bit_errors'] for r in rows),
            'frame_mean_bler':mean_interval(errors,indices),'frame_mean_ber':mean_interval(ber,indices),
            'frame_mean_harm':mean_interval(harm,indices),'audit_harm_events':int(audit.sum()),
            'audit_harm_upper_95':binomial_upper(int(audit.sum()),n,.05),
            'radio_goodput_bits_per_waveform_sample':mean_interval(goodput,indices,support=(0.,goodput_support)),
            'mean_detector_iterations_per_vector':float(np.mean([r['iterations']/r['vectors'] for r in rows])),
            'mean_decoder_iterations_per_user':float(np.mean([r['decoder_iterations']/r['blocks'] for r in rows])),
            'gain_clips':sum(r.get('gain_clip_count',0) for r in rows)}
        summaries.append(summary)
    for summary in summaries:
        difference=arrays[summary['method']]['bler']-arrays['exact']['bler']
        summary['paired_bler_difference']=({'mean':0.,'lower':0.,'upper':0.,'method':'identical reference compared with itself'}
            if summary['method']=='exact' else mean_interval(difference,indices,support=(-1.,1.)))
    primary=next(s for s in summaries if s['method']==selected)
    primary={**primary,'alpha':alpha,'calibration_upper_limit':selection['selections'][str(alpha)]['upper_limits'].get(selected,0.),
             'formal_scope':'source-frame mixture under frozen independent calibration; no conditional or shift guarantee'}
    cells=[]
    for cell in sorted({f['condition_id'] for f in frames}):
        subset=[f for f in frames if f['condition_id']==cell]
        cell_indices=rng.integers(0,len(subset),size=(repetitions,len(subset)),dtype=np.int32)
        for name in names:
            rows=[next(r for r in f['rows'] if r['method']==name) for f in subset]
            values=np.array([r['block_errors']/r['blocks'] for r in rows])
            interval=mean_interval(values,cell_indices)
            cells.append({'condition_id':cell,'antennas':subset[0]['antennas'],'users':subset[0]['users'],
                'rho':subset[0]['rho'],'snr_db':subset[0]['snr_db'],'method':name,
                'independent_frames':len(subset),'blocks':sum(r['blocks'] for r in rows),
                'block_errors':sum(r['block_errors'] for r in rows),
                'bler':interval['mean'],'lower':interval['lower'],'upper':interval['upper'],'interval_method':interval['method']})
    report={'stage':'independent frozen test analysis','lock_sha256':meta['lock_sha256'],
        'raw_records_sha256':meta['records_sha256'],'bootstrap_repetitions':repetitions,
        'bootstrap_seed':4100,'confidence':.95,
        'inference_note':'Primary policy fixed before testing. Other method/cell intervals are descriptive and unadjusted.',
        'timing_note':'Concurrent test timings excluded from speed claims; serial replay study required.',
        'primary':primary,'methods':summaries,'cells':cells}
    output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(report,indent=2))
    with (output/'cell_bler.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
    print(json.dumps(primary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/final/joint_budget_002')
    parser.add_argument('--output',type=Path,default=ROOT/'results/analysis/joint_budget_002')
    args=parser.parse_args();analyze(args.input,args.output)
