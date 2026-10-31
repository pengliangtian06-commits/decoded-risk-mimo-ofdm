#!/usr/bin/env python3
"""Reproduce development benchmark summary without treating repeats as channels."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np


def summarize(folder):
    metadata = json.loads((folder/'metadata.json').read_text())
    if not metadata['completed']:
        raise ValueError('Incomplete benchmark')
    grouped = defaultdict(list)
    for line in (folder/'timings.jsonl').read_text().splitlines():
        row = json.loads(line)
        grouped[(row['condition_id'], row['method'])].append(row)
    output = []
    for (condition, method), rows in sorted(grouped.items()):
        if len({tuple(r['per_user_bit_errors']) for r in rows}) != 1:
            raise ValueError('A deterministic replay changed decoded bits')
        if len({r['repetition'] for r in rows}) != metadata['repetitions']:
            raise ValueError('Missing/duplicate timing repetition')
        result = {'condition_id':condition,'users':rows[0]['users'],'rho':rows[0]['rho'],
                  'method':method,'timing_repetitions':len(rows),
                  'independent_frames':1,'block_errors_in_one_frame':rows[0]['block_errors']}
        for metric in ['preparation','solve','soft_calibration','llr','detection_total','decoder','receiver']:
            values = np.array([r[metric+'_seconds'] for r in rows])*1000
            result[metric+'_median_ms'] = float(np.median(values))
            result[metric+'_p95_ms'] = float(np.percentile(values,95))
        output.append(result)
    direct = ['exact','exact-lapack-native','exact-scipy-solve','exact-scipy-lapack']
    for condition in sorted({k[0] for k in grouped}):
        exact_rows = [rows for (c,m),rows in grouped.items() if c==condition and m in direct]
        if len({tuple(rows[0]['per_user_bit_errors']) for rows in exact_rows}) != 1:
            raise ValueError('Direct backends disagree on decoded errors')
    with (folder/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    text = ['# Serial development timing summary','',
        'Four independent cached waveforms; 25 timed replays per method/condition.',
        'Replays measure timing variability, not independent reliability samples.',
        'Each complete receiver includes FFT/estimation, preparation, solve, soft outputs and decoding.','',
        '| U | rho | Cheapest direct implementation | Direct detection ms | PCG i16 noise ms | Direct receiver ms | PCG receiver ms |',
        '|---:|---:|---|---:|---:|---:|---:|']
    for condition in sorted({r['condition_id'] for r in output}):
        group = [r for r in output if r['condition_id']==condition]
        fastest = min((r for r in group if r['method'] in direct),key=lambda r:r['detection_total_median_ms'])
        candidate = next(r for r in group if r['method']=='pcg-i16-noise')
        text.append(f"| {fastest['users']} | {fastest['rho']} | {fastest['method']} | "
            f"{fastest['detection_total_median_ms']:.3f} | {candidate['detection_total_median_ms']:.3f} | "
            f"{fastest['receiver_median_ms']:.3f} | {candidate['receiver_median_ms']:.3f} |")
    text += ['', 'The candidate is a pre-existing screening setting, not a validated winning policy.',
        'Independent coded-risk calibration and closest-method comparisons remain outstanding.',
        'No 20% improvement over the strongest direct implementation is established here.']
    (folder/'SUMMARY.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    summarize(parser.parse_args().folder)
