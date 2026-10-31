#!/usr/bin/env python3
"""Figure 2: complete finite-family calibration, one quantitative panel.

Claim: ten of sixteen approximation policies satisfy the frozen 1% source-
mixture harm budget; the selected direct stride-32 policy has U=0.3993%.
Evidence: all 4096 independent calibration frames; one prespecified uniform
audit user per frame. Intervals are exact one-sided simultaneous limits using
Bonferroni delta/K, not user-pooled or two-sided error bars. No records omitted.
Physical contract: 180 x 140 mm, >=8 pt labels, vector PDF/SVG, 1000-dpi TIFF.
Python/Matplotlib is the established exclusive plotting backend.
"""
import argparse,csv,hashlib,json,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--qa-tools',type=Path)
    args=p.parse_args()
    source=ROOT/'results/calibration/joint_budget_002/selection.json'
    data=json.loads(source.read_text());alpha=data['primary_alpha']
    selection=data['selections'][str(alpha)];selected=selection['selected']
    label={
      'direct-i8':'Direct, stride 8','direct-i16':'Direct, stride 16',
      'direct-i32':'Direct, stride 32 (selected)','direct-i64':'Direct, stride 64',
      'pcg-exact':'PCG, exact gains','pcg-diagonal':'PCG, diagonal gains',
      'pcg-i16':'PCG, stride 16','pcg-i16-loose':'PCG, loose residual, stride 16',
      'pcg-i16-noise':'PCG, noise residual, stride 16',
      'pcg-i32-noise':'PCG, noise residual, stride 32',
      'fixed6-exact':'PCG fixed 6, exact gains','fixed6-diagonal':'PCG fixed 6, diagonal gains',
      'fixed6-i16':'PCG fixed 6, stride 16','yin6':'CG-D approximation, 6 iterations',
      'yin8':'CG-D approximation, 8 iterations','mf-i16':'Matrix-free PCG, stride 16'}
    names=list(data['event_counts']);assert set(names)==set(label) and len(names)==16
    rows=[{'method':name,'label':label[name],'frames':data['frames'],
            'audit_harm_events':data['event_counts'][name],
            'empirical_risk':data['event_counts'][name]/data['frames'],
            'simultaneous_upper':selection['upper_limits'][name],
            'eligible':name in selection['eligible'],'selected':name==selected,
            'development_receiver_ms':1000*data['costs_from_development'][name]} for name in names]
    out=ROOT/'figures/main';out.mkdir(parents=True,exist_ok=True);stem=out/'fig02_calibration'
    with stem.with_suffix('.source.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig,ax=plt.subplots(figsize=(180/25.4,140/25.4))
    fig.subplots_adjust(left=.415,right=.97,bottom=.19,top=.91)
    for i,row in enumerate(rows):
        color='#D07929' if row['selected'] else '#217A9B' if row['eligible'] else '#9E5356'
        ax.hlines(i,0,100*row['simultaneous_upper'],color=color,linewidth=1.3)
        ax.plot(100*row['simultaneous_upper'],i,marker='|',markersize=7,color=color,markeredgewidth=1.4)
        ax.plot(100*row['empirical_risk'],i,'o',color='#252525',markersize=3.3)
    ax.axvline(100*alpha,color='#555555',linewidth=1.,linestyle=(0,(3,3)),zorder=0)
    ax.set_yticks(np.arange(len(rows)),[r['label'] for r in rows])
    for tick,row in zip(ax.get_yticklabels(),rows):
        if row['selected']:tick.set_fontweight('bold')
    ax.invert_yaxis();ax.set_ylim(len(rows)-.4,-.6)
    ax.set_xlim(0,3.1);ax.set_xticks([0,.5,1.,1.5,2.,2.5,3.])
    ax.set_xlabel('Probability of an extra decoded block error (%)',labelpad=7)
    ax.tick_params(axis='y',length=0,pad=7)
    ax.spines['left'].set_visible(False)
    fig.text(.04,.97,'Finite-family calibration',fontsize=10,ha='left',va='top')
    fig.text(.97,.97,'4,096 independent frames',fontsize=8.5,ha='right',va='top')
    handles=[Line2D([],[],marker='o',linestyle='none',color='#252525',markersize=3.5,label='Empirical harm'),
        Line2D([],[],marker='|',color='#217A9B',linewidth=1.3,markersize=7,label='Simultaneous upper limit'),
        Line2D([],[],color='#555555',linestyle=(0,(3,3)),linewidth=1,label='1% budget')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.035),ncol=3,
               frameon=False,handlelength=1.6,columnspacing=1.5)
    fig.canvas.draw()
    if args.qa_tools:
        sys.path.insert(0,str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=str(stem.with_suffix('.alignment.json')),
            tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(stem.with_suffix('.pdf'));fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.tiff'),dpi=1000,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    caption=('Figure 2. Simultaneous calibration of the frozen computation family. '
        'Each of 4,096 independently generated source-mixture frames contributes one uniformly '
        'selected audit user. Harm means that the candidate decodes its information block '
        'incorrectly while the exact reference decodes it correctly. Dots show observed audit-user '
        'harm fractions; horizontal intervals span zero to the exact one-sided Clopper–Pearson '
        'upper limit at confidence allocation 0.05/16. The complete family is covered with '
        'probability at least 0.95 under the stated independent source distribution. The dashed '
        'line is the predeclared 1% risk budget. Blue intervals qualify, muted red intervals do '
        'not, and orange identifies the eligible policy selected by separate development cost. '
        'The exact fallback has identically zero excess loss and is not plotted. This certificate '
        'protects the source-mixture average and does not extend to individual conditions or shifts.\n')
    stem.with_suffix('.caption.txt').write_text(caption)
    stem.with_suffix('.contract.json').write_text(json.dumps({'core_conclusion':
        'Ten of sixteen candidates qualify at alpha=0.01; selected direct-i32 has upper limit 0.0039933',
        'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'archetype':'one-panel finite-family one-sided risk intervals','width_mm':180,'height_mm':140,
        'backend':'python-matplotlib','data_unit':'one audit user per independent frame',
        'uncertainty':'exact simultaneous one-sided bounds; delta=0.05, K=16',
        'all_candidates_shown':True,'main_figure_number':2},indent=2))
    print(stem.with_suffix('.pdf'))

if __name__=='__main__':main()
