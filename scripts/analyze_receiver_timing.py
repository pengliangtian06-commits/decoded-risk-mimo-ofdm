#!/usr/bin/env python3
"""Summarize serial timing with the independent waveform as bootstrap cluster."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def analyze(folder,output):
    if output.exists():raise FileExistsError('Refusing overwrite')
    meta=json.loads((folder/'metadata.json').read_text())
    if not meta['completed']:raise ValueError('Benchmark still running')
    path=folder/'timings.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['records_sha256']
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    groups=defaultdict(list)
    for r in rows:groups[(r['method'],r['frame_id'])].append(r)
    methods=sorted({r['method'] for r in rows});frames=sorted({r['frame_id'] for r in rows})
    assert len(frames)==48 and len(rows)==4080
    metrics=['estimation','preparation','solve','soft_calibration','llr','detection_total','decoder','receiver']
    frame_means={m:{k:[] for k in metrics} for m in methods}
    for method in methods:
        for frame in frames:
            sample=groups[(method,frame)]
            assert {r['repetition'] for r in sample}==set(range(5)) and len(sample)==5
            assert len({tuple(r['per_user_bit_errors']) for r in sample})==1
            for metric in metrics:frame_means[method][metric].append(np.mean([r[metric+'_seconds'] for r in sample]))
    # Balanced design: resample complete waveforms within each of the 12 cells.
    rng=np.random.default_rng(4151);indices=[]
    for condition in range(12):
        positions=np.array([i for i,f in enumerate(frames) if int(f.split(':')[0])==condition])
        assert len(positions)==4
        indices.append(rng.choice(positions,size=(5000,len(positions)),replace=True))
    indices=np.concatenate(indices,axis=1)
    ref=np.array(frame_means['exact']['receiver']);ref_boot=ref[indices].mean(axis=1)
    summary=[]
    for method in methods:
        sample=[r for r in rows if r['method']==method]
        receiver=np.array(frame_means[method]['receiver'])
        ratios=receiver[indices].mean(axis=1)/ref_boot
        low,high=np.quantile(ratios,[.025,.975])
        r={'method':method,'independent_frames':48,'timed_calls':len(sample),
           **{k+'_mean_ms':float(np.mean(frame_means[method][k])*1000) for k in metrics},
           'receiver_p50_ms':float(np.quantile([r['receiver_seconds'] for r in sample],.5)*1000),
           'receiver_p95_ms':float(np.quantile([r['receiver_seconds'] for r in sample],.95)*1000),
           'paired_mean_receiver_ratio':float(receiver.mean()/ref.mean()),
           'ratio_lower':float(low),'ratio_upper':float(high)}
        summary.append(r)
    primary=next(r for r in summary if r['method']==meta['selected'])
    report={'source_metadata_sha256':hashlib.sha256((folder/'metadata.json').read_bytes()).hexdigest(),
        'raw_records_sha256':meta['records_sha256'],'selected_before_timing':meta['selected'],
        'interval':'95% percentile bootstrap, 5000 resamples, 48 complete waveform clusters resampled within 12 cells',
        'tail_scope':'P50/P95 over 240 observed calls per method; calls share 48 waveform clusters',
        'primary':primary,'methods':summary}
    output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(report,indent=2))
    with (output/'summary.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps(primary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/final/receiver_timing_002')
    p.add_argument('--output',type=Path,default=ROOT/'results/analysis/receiver_timing_002')
    a=p.parse_args();analyze(a.input,a.output)
