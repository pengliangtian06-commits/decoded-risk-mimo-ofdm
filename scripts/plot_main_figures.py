#!/usr/bin/env python3
"""Evidence-grounded main figures 3--9, exclusive Python/Matplotlib backend.

All cohorts are complete before plotting. No outliers or failed policies are
removed. Physical sizes, uncertainty and sample units are in FIGURE_BRIEF.md
and captions. Distribution boxes/CDFs are descriptive, not confidence bands.
Exports: editable PDF/SVG, PNG preview, line-art TIFF at 1000 dpi. Font floor
8 pt. Optional external nature-figure alignment helper is used on original QA.
"""
import argparse,csv,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT=ROOT/'figures/main'
STYLE={'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8.5,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False}
COLORS={'exact':'#252525','direct-i32':'#D07929','direct-i8':'#217A9B',
        'pcg-diagonal':'#A35457','pcg-i16-noise':'#26877C','pcg-exact':'#7088B0','yin8':'#8B6BA6'}
LABELS={'exact':'Exact reference','direct-i32':'Selected: direct, stride 32',
    'direct-i8':'Direct, stride 8','pcg-diagonal':'PCG, diagonal gains',
    'pcg-i16-noise':'PCG, noise residual, stride 16','pcg-exact':'PCG, exact gains',
    'yin8':'CG-D approximation, 8 iterations'}
SHORT={'direct-i8':'Direct s8','direct-i16':'Direct s16','direct-i32':'Direct s32',
    'direct-i64':'Direct s64','pcg-exact':'PCG exact','pcg-diagonal':'PCG diagonal',
    'pcg-i16':'PCG s16','pcg-i16-loose':'PCG loose s16','pcg-i16-noise':'PCG noise s16',
    'pcg-i32-noise':'PCG noise s32','fixed6-exact':'Fixed 6 exact',
    'fixed6-diagonal':'Fixed 6 diagonal','fixed6-i16':'Fixed 6 s16',
    'yin6':'CG-D approx. 6','yin8':'CG-D approx. 8','mf-i16':'Matrix-free s16','exact':'Exact'}
CAPTIONS={}

def read(path):return json.loads((ROOT/path).read_text())
def jsonlines(path):return [json.loads(l) for l in (ROOT/path).read_text().splitlines()]
def canvas(w,h):return plt.subplots(figsize=(w/25.4,h/25.4))
def color(name):return COLORS.get(name,'#87949C')
def save(fig,number,slug,caption,source,qa):
    stem=OUT/f'fig{number:02d}_{slug}'
    stem.with_suffix('.source.json').write_text(json.dumps(source,indent=2))
    fig.canvas.draw()
    if qa:
        sys.path.insert(0,str(qa))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=str(stem.with_suffix('.alignment.json')),
            tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(stem.with_suffix('.pdf'));fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.tiff'),dpi=1000,pil_kwargs={'compression':'tiff_lzw'})
    stem.with_suffix('.caption.txt').write_text(f'Figure {number}. '+caption+'\n')
    plt.close(fig);print(stem.with_suffix('.pdf'),flush=True)

def bler(report,qa):
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,140/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.93,hspace=.30,wspace=.18)
    methods=['exact','direct-i32','pcg-diagonal','yin8'];source=[]
    for ax,(users,rho),letter in zip(axes.flat,[(16,0.),(16,.7),(32,0.),(32,.7)],'abcd'):
        for j,name in enumerate(methods):
            cells=sorted([c for c in report['cells'] if c['users']==users and c['rho']==rho and c['method']==name],key=lambda c:c['snr_db'])
            source.extend(cells);x=np.array([c['snr_db'] for c in cells])+(j-1.5)*.045
            y=np.array([c['bler'] for c in cells])*100
            low=np.array([c['lower'] for c in cells])*100;high=np.array([c['upper'] for c in cells])*100
            ax.errorbar(x,y,yerr=np.maximum(np.stack([y-low,high-y]),0),fmt='o-',markersize=3,
                color=color(name),linewidth=1.1,capsize=2,elinewidth=.7,label=LABELS[name])
        ax.set_yscale('symlog',linthresh=1,linscale=1);ax.set_ylim(-.08,105)
        ax.set_yticks([0,.5,1,5,10,50,100],['0','0.5','1','5','10','50','100'])
        ax.set_xticks([10,12,14]);ax.set_xlim(9.6,14.4)
        ax.set_title(f'{letter}   U = {users}, ρ = {rho:g}',loc='left',fontsize=9,pad=8)
    for ax in axes[1]:ax.set_xlabel('Nominal Es/N₀ (dB)')
    for ax in axes[:,0]:ax.set_ylabel('Mean BLER (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.025),ncol=2,columnspacing=2)
    denominators='; '.join(letter+': '+ '/'.join(str(c['independent_frames']) for c in sorted([c for c in report['cells'] if c['users']==u and c['rho']==r and c['method']=='exact'],key=lambda c:c['snr_db'])) for letter,(u,r) in zip('abcd',[(16,0.),(16,.7),(32,0.),(32,.7)]))
    save(fig,3,'coded_bler','Independent source-test BLER across user load, receive correlation and SNR. '
        'All methods share the same frames in each cell (132–204 independent frames/cell, exact counts in source data). '
        'Points average within-frame user error fractions. Bars are 95% frame-cluster bootstrap intervals; '
        'constant zero cells instead retain conservative exact zero-event frame upper limits. The vertical axis '
        'is linear below 1% and logarithmic above 1%, so zero observations remain visible without pseudocounts. '
        'Tiny horizontal offsets separate methods and do not change SNR. Intervals across cells are descriptive and unadjusted. '
        'Independent frame counts at 10/12/14 dB are '+denominators+'.',source,qa)

def harm(report,qa):
    data=[r for r in report['methods'] if r['method']!='exact'];y=np.arange(len(data))
    fig,axes=plt.subplots(1,2,figsize=(180/25.4,145/25.4),sharey=True)
    fig.subplots_adjust(left=.29,right=.98,bottom=.12,top=.91,wspace=.24)
    for i,r in enumerate(data):
        for ax,key in zip(axes,['paired_bler_difference','frame_mean_harm']):
            t=r[key];p=t['mean']*100
            ax.errorbar(p,i,xerr=[[max(0,p-t['lower']*100)],[max(0,t['upper']*100-p)]],
                fmt='o',color=color(r['method']),markersize=4,capsize=2,linewidth=1.)
    axes[0].set_yticks(y,[SHORT[r['method']] for r in data]);axes[0].invert_yaxis()
    axes[0].axvline(0,color='#666666',linestyle='--',linewidth=.7)
    axes[0].set_xlabel('Net BLER difference (pp)');axes[1].set_xlabel('Extra-error risk (%)')
    axes[0].set_title('a   Signed paired difference',loc='left',fontsize=9,pad=9)
    axes[1].set_title('b   One-sided harm',loc='left',fontsize=9,pad=9)
    axes[0].set_xlim(-.2,2.7);axes[1].set_xlim(0,2.9)
    for ax in axes:ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    save(fig,4,'paired_harm','Net decoded BLER difference and positive paired harm for all sixteen approximation policies on the 2048 independent test frames. '
        'Harm counts a candidate error on a reference-correct block, while the signed difference also includes opposite outcomes. '
        'Points and 95% intervals use within-frame user fractions and paired frame-cluster bootstrap resampling. '
        'Orange marks the preselected policy. These test intervals are descriptive for other family members and do not replace the simultaneous calibration limits.',data,qa)

def frontier(report,timing,qa):
    fig,ax=canvas(180,105);fig.subplots_adjust(left=.13,right=.97,bottom=.23,top=.93)
    costs={m['method']:m for m in timing['methods']};source=[]
    for r in report['methods']:
        name=r['method'];t=r['paired_bler_difference'];x=costs[name]['receiver_mean_ms'];y=100*t['mean']
        ax.errorbar(x,y,yerr=[[max(0,y-100*t['lower'])],[max(0,100*t['upper']-y)]],fmt='o',
            color=color(name),markersize=6 if name=='direct-i32' else 4,capsize=2,elinewidth=.8)
        source.append({**t,'interval_method':t.get('method'),'method':name,'receiver_mean_ms':x})
    ax.axhline(0,color='#AAAAAA',linewidth=.7,zorder=0)
    labels={'exact':('Exact',(5,-17)),'direct-i32':('Selected s32',(-25,13)),
        'pcg-exact':('PCG + exact gains',(-62,15)),
        'fixed6-diagonal':('Fixed 6 + diagonal',(55,12)),'yin8':('CG-D approx. 8',(7,0))}
    for r in source:
        if r['method'] in labels:
            label,offset=labels[r['method']]
            ax.annotate(label,(r['receiver_mean_ms'],100*r['mean']),xytext=offset,textcoords='offset points',fontsize=8)
    ax.set_xlim(33.5,49);ax.set_ylim(-.25,2.55)
    ax.set_xlabel('Mean complete receiver time (ms)');ax.set_ylabel('Paired BLER difference (pp)')
    fig.text(.13,.98,'Executed reliability–cost trade-off',fontsize=10,va='top')
    save(fig,5,'cost_frontier','Executed source reliability and complete receiver cost for all seventeen frozen pipelines. '
        'Time means come from 48 held-out serial timing waveforms, each with five measured calls after three warm-ups. '
        'Paired BLER differences and vertical 95% intervals come from the separate 2048-frame reliability cohort. '
        'The selected policy was fixed before both evaluations; the point cloud does not perform test-set reselection. '
        'Grey points identify additional family members listed in the source table. Timing and BLER axes are measured quantities, not operation-count predictions.',source,qa)

def costs(timing,qa):
    methods=['exact','direct-i32','pcg-exact','pcg-i16-noise','yin8']
    lookup={r['method']:r for r in timing['methods']};data=[lookup[m] for m in methods]
    fig,ax=canvas(180,105);fig.subplots_adjust(left=.12,right=.98,bottom=.30,top=.93)
    components=[('estimation','FFT + estimation','#C5D7DF'),('preparation','Preparation','#7D9DAF'),
        ('solve','Solve / coupled recursion','#3F697F'),('soft_calibration','Gain calculation','#DC9C5B'),
        ('llr','Demapping','#E7CD91'),('decoder','LDPC + CRC','#8B7B9E'),('other','Dispatch / conversion','#D1D1D1')]
    bottom=np.zeros(len(methods))
    for key,label,c in components:
        values=np.array([r[key+'_mean_ms'] if key!='other' else r['receiver_mean_ms']-sum(r[k+'_mean_ms'] for k,_,_ in components if k!='other') for r in data])
        assert values.min()>-1e-6
        ax.bar(np.arange(len(methods)),np.maximum(values,0),bottom=bottom,color=c,width=.62,label=label,linewidth=0)
        bottom+=values
    for i,r in enumerate(data):ax.text(i,r['receiver_mean_ms']+.8,f"{r['receiver_mean_ms']:.2f}",ha='center',fontsize=8)
    ax.set_xticks(np.arange(len(methods)),['Exact','Selected\ns32','PCG\nexact gains','PCG noise\ns16','CG-D\napprox. 8'])
    ax.set_ylabel('Mean complete receiver time (ms)');ax.set_ylim(0,52)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.015),ncol=3,columnspacing=1.4,handlelength=1.4)
    save(fig,6,'cost_breakdown','Complete receiver timing decomposition for representative frozen pipelines. '
        'Bars average all 48 held-out waveforms and five measured calls per waveform. Preparation includes required channel-system construction and factorization. '
        'Coupled CG soft recursion is included with solve time; separate soft-output work appears under gain calculation. '
        'Dispatch/conversion is the measured complete wall time minus the named components, so wrapper overhead remains charged. '
        'Decoding includes rate recovery and CRC. Waveform generation and encoding are outside receiver latency.',data,qa)

def gains(frames,qa):
    methods=['direct-i8','direct-i16','direct-i32','direct-i64','pcg-diagonal'];source=[];values=[]
    for name in methods:
        v=np.array([next(r['gain_absolute_error_mean'] for r in f['rows'] if r['method']==name) for f in frames])
        assert np.all(v>0);values.append(v)
        source.extend({'frame_draw':f['draw'],'method':name,'mean_absolute_gain_error':float(x)} for f,x in zip(frames,v))
    fig,ax=canvas(150,95);fig.subplots_adjust(left=.17,right=.97,bottom=.23,top=.93)
    b=ax.boxplot(values,whis=(5,95),widths=.5,patch_artist=True,
        flierprops={'marker':'.','markersize':1.5,'alpha':.18,'markeredgewidth':0,
            'markerfacecolor':'#555555','markeredgecolor':'none'},
        medianprops={'color':'#222222','linewidth':1.1})
    for box,name in zip(b['boxes'],methods):box.set_facecolor(color(name));box.set_alpha(.5)
    ax.set_yscale('log');ax.set_xticks(range(1,6),['Direct\ns8','Direct\ns16','Selected\ns32','Direct\ns64','PCG\ndiagonal'])
    ax.set_ylabel('Frame mean absolute gain-proxy error')
    save(fig,7,'gain_error','Gain-proxy approximation error on all 2048 independent source-test frames. '
        'Each value averages absolute differences from exact per-tone model gains over users and tones in one frame. '
        'Boxes show the 25th–75th percentiles, lines the median, whiskers the 5th–95th percentiles, and all outer observations remain plotted. '
        'These are distributions, not confidence intervals. Direct strides isolate the gain calculation budget; the diagonal PCG comparator has a different numerical solve. '
        'Gain accuracy alone is not a decoded-error certificate.',source,qa)

def latency(rows,qa):
    methods=['exact','direct-i32','pcg-exact','pcg-i16-noise','yin8'];source=[]
    fig,ax=canvas(150,105);fig.subplots_adjust(left=.14,right=.97,bottom=.30,top=.95)
    maximum=0
    for name in methods:
        sample=[r for r in rows if r['method']==name];v=np.sort([r['receiver_seconds']*1000 for r in sample]);maximum=max(maximum,v[-1])
        short={'exact':'Exact reference','direct-i32':'Selected s32','pcg-exact':'PCG exact gains',
            'pcg-i16-noise':'PCG noise s16','yin8':'CG-D approx. 8'}
        ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color(name),linewidth=1.25,label=short[name])
        source.extend({'method':name,'frame_id':r['frame_id'],'repetition':r['repetition'],'receiver_ms':r['receiver_seconds']*1000} for r in sample)
    ax.set_xlim(0,maximum*1.03);ax.set_ylim(0,1.02)
    ax.set_xlabel('Complete receiver time (ms)');ax.set_ylabel('Empirical cumulative fraction')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.015),ncol=2,handlelength=2)
    save(fig,8,'latency_cdf','Observed complete-call latency distributions for five representative frozen policies. '
        'All 240 measured calls per policy are retained, including slow calls. Each policy uses the same 48 held-out waveforms with five calls after three warm-ups and randomized method order. '
        'The CDF describes calls; confidence inference elsewhere resamples waveform clusters and does not treat all calls as independent channels.',source,qa)

def transfer(rows,qa):
    labels=['A100 / source / 16QAM','C100 / source / 16QAM','A100 / matched / 16QAM','C100 / matched / 16QAM',
        'A300 / source / 16QAM','C300 / source / 16QAM','Exp. / source / 64QAM','A100 / source / 64QAM',
        'A300 / matched / 64QAM','C300 / matched / 64QAM']
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,185/25.4),sharey=True)
    fig.subplots_adjust(left=.37,right=.98,bottom=.16,top=.94,wspace=.26,hspace=.28)
    source=[]
    for col,rho in enumerate([0.,.7]):
        for row,metric,methods in [(0,'bler',['exact','direct-i32']),(1,'harm',['direct-i32','direct-i8'])]:
            ax=axes[row,col]
            for j,name in enumerate(methods):
                data=sorted([r for r in rows if r['rho']==rho and r['method']==name],key=lambda r:r['arm_id']);source.extend(data)
                x=np.array([r[metric] for r in data])*100
                lo=np.array([r[metric+'_lower'] for r in data])*100;hi=np.array([r[metric+'_upper'] for r in data])*100
                ax.errorbar(x,np.arange(10)+(j-.5)*.16,xerr=np.maximum(np.stack([x-lo,hi-x]),0),fmt='o',color=color(name),
                    markersize=3,capsize=1.7,elinewidth=.7,linewidth=1)
            ax.set_yticks(range(10),labels);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
            ax.set_title(f"{'abcd'[row*2+col]}   ρ = {rho:g}",loc='left',fontsize=9,pad=8)
            if row==0:ax.set_xlim(-1.5,103);ax.set_xticks([0,50,100]);ax.set_xlabel('Mean BLER (%)')
            else:ax.set_xlim(-.08,3.8);ax.set_xticks([0,1,2,3]);ax.set_xlabel('Extra-error risk (%)');ax.axvline(1,color='#777777',linestyle='--',linewidth=.7,zorder=0)
    axes[0,0].invert_yaxis()
    handles=[Line2D([],[],color=color(n),marker='o',markersize=3,label=LABELS[n]) for n in ['exact','direct-i32','direct-i8']]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.025),ncol=1,handlelength=1.6)
    save(fig,9,'transfer','Transfer of source-frozen pipelines across every declared profile/prior/modulation cell. '
        'A and C denote quantized TDL-A/C PDPs; 100/300 denote nominal RMS delays in ns. Exact physical scales, resulting RMS delays, loads and SNRs are in Table 1 and source data. '
        'Each cell contains 128 independent draws; common streams across arms are retained as 256 clusters and are not pooled as 2560 independent observations. '
        'Top panels show absolute BLER for the reference and selected pipeline; bottom panels show harm for the selected and finer-stride direct pipelines. '
        'Bars are descriptive 95% frame-bootstrap intervals, with conservative exact frame upper limits for zero observations; they are unadjusted across cells. '
        'The dashed 1% source budget is contextual only. The source-mixture certificate does not apply to these shifted distributions.',source,qa)

def main():
    p=argparse.ArgumentParser();p.add_argument('--qa-tools',type=Path);p.add_argument('--figures',default='3,4,5,6,7,8,9');a=p.parse_args()
    matplotlib.rcParams.update(STYLE);OUT.mkdir(parents=True,exist_ok=True)
    report=read('results/analysis/joint_budget_002/analysis.json');timing=read('results/analysis/receiver_timing_002/analysis.json')
    for number in map(int,a.figures.split(',')):
        if number==3:bler(report,a.qa_tools)
        elif number==4:harm(report,a.qa_tools)
        elif number==5:frontier(report,timing,a.qa_tools)
        elif number==6:costs(timing,a.qa_tools)
        elif number==7:gains(jsonlines('results/final/joint_budget_002/channel_records.jsonl'),a.qa_tools)
        elif number==8:latency(jsonlines('results/final/receiver_timing_002/timings.jsonl'),a.qa_tools)
        elif number==9:
            import pandas as pd
            transfer(pd.read_csv(ROOT/'results/analysis/transfer_002/cell_results.csv').to_dict('records'),a.qa_tools)
        else:raise ValueError(number)

if __name__=='__main__':main()
