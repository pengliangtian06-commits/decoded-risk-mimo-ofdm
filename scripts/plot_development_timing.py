#!/usr/bin/env python3
"""Development-only diagnostic; no final manuscript figure number assigned."""
from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0,'/Users/pengliangtian/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

# Contract: actual complete core time depends on matrix reuse and library
# implementation; iteration counts alone do not establish a fast receiver.
# One quantitative panel. 180 x 110 mm, vector editable PDF/SVG plus PNG preview.
# All 20 timing repetitions at each selected reuse value are used. The
# scientifically specified slice is the largest initial benchmark system,
# 256 receive antennas / 128 streams, rho=0, to inspect preparation/reuse cost.
# Bands are measured interquartile ranges, not channel-realization confidence
# intervals. These are development numerical-core timings, excluding LLRs,
# pilots, OFDM and LDPC; rules are not matched for coded reliability.

matplotlib.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8.5,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':0.7,'legend.frameon':False,
})
source=ROOT/'results/pilot/core_benchmark_002/timing.csv'
raw=pd.read_csv(source)
selected=raw.query('antennas == 256 and users == 128 and rho == 0.0')
output=ROOT/'figures/development';output.mkdir(parents=True,exist_ok=True)
selected.to_csv(output/'reuse_cost_diagnostic_source.csv',index=False)
fig,ax=plt.subplots(figsize=(180/25.4,110/25.4))
fig.subplots_adjust(left=0.16,right=0.97,bottom=0.32,top=0.92)
specifications=[
    ('scipy_cholesky','fixed','Cholesky / Accelerate','#404040'),
    ('cholesky','fixed','Cholesky / Eigen','#7C8894'),
    ('pcg','relative_residual','Gram PCG / Accelerate','#2878A0'),
    ('matrixfree_pcg','relative_residual','Matrix-free PCG / Accelerate','#B67632'),
]
for backend,mode,label,color in specifications:
    subset=selected[(selected.backend==backend)&(selected['mode']==mode)]
    groups=subset.groupby('reuse').total_seconds
    x=np.array(sorted(groups.groups))
    median=groups.median().reindex(x).to_numpy()*1e6
    lower=groups.quantile(.25).reindex(x).to_numpy()*1e6
    upper=groups.quantile(.75).reindex(x).to_numpy()*1e6
    if np.any(x<=0) or np.any(lower<=0) or np.any(~np.isfinite(upper)):
        raise ValueError('Log axes require finite positive observed timings and reuse counts')
    ax.plot(x,median,'o-',color=color,label=label,markersize=4,linewidth=1.4)
    ax.fill_between(x,lower,upper,color=color,alpha=.14,linewidth=0)
ax.set_xscale('log');ax.set_yscale('log')
ax.set_xticks([1,14,128],['1','14','128'])
ax.set_yticks([100,300,1000,3000,10000,30000],['100','300','1,000','3,000','10,000','30,000'])
ax.set_ylim(100,35000)
ax.set_xlabel('Receive vectors sharing one channel matrix')
ax.set_ylabel('Preparation + batched solve time (μs)')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2,columnspacing=1.6,handlelength=2)
fig.text(.16,.97,'Development diagnostic · 256 antennas / 128 streams',ha='left',va='top',fontsize=9)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=str(output/'reuse_cost_diagnostic.alignment.json'),
                                  tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(output/'reuse_cost_diagnostic.pdf')
fig.savefig(output/'reuse_cost_diagnostic.svg')
fig.savefig(output/'reuse_cost_diagnostic.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(output/'reuse_cost_diagnostic.png',dpi=300)
plt.close(fig)
(output/'reuse_cost_diagnostic_caption.txt').write_text(
    'Development diagnostic. Median preparation plus batched-solve wall time for '
    '256 receive antennas and 128 streams, uncorrelated conditional channel, seed 1108. '
    'Shading shows the interquartile range of 20 timing repetitions after three warm-ups. '
    'All four implementations use complex double precision and one native thread. '
    'PCG uses relative residual tolerance 0.001. Timings exclude channel estimation, '
    'OFDM processing, soft-output calculation and decoding; these methods have not '
    'been calibrated to matched coded reliability. This figure supports no RF '
    'real-time or final receiver speedup claim.\n')
print(output/'reuse_cost_diagnostic.pdf')
