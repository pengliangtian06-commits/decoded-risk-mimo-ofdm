#!/usr/bin/env python3
"""Editable implemented receiver flow, 180 x 105 mm, existing/new roles explicit.

Brief and code anchors are in manuscript/FIGURE_BRIEF.md. Offline payload truth
never feeds the online chain. The only downward edge is a frozen policy. The
accent marks the receiver-specific selection interface, not new statistics.
"""
from pathlib import Path
import os,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch

def main():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    w,h=180/25.4*72,105/25.4*72
    fig=plt.figure(figsize=(180/25.4,105/25.4));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,w);ax.set_ylim(0,h);ax.axis('off')
    centers=[84,w/2,w-84];height=58;width=140
    nodes=[
      ('offline_family',0,235,'Known-bit frames\nExact + family decoding\nPaired audit-user harm',False),
      ('selection',1,235,'Simultaneous risk limits\nDevelopment-cost ranking\nPolicy + exact fallback',True),
      ('evaluation',2,235,'Untouched evaluation\nSource + transfer tests\nSerial CPU measurements',False),
      ('estimation',0,116,'Received OFDM waveform\nPilot/noise estimation\nConfigured channel prior',False),
      ('computation',1,116,'Frozen computation policy\nNumerical solve\nSoft-gain budget',False),
      ('decoding',2,116,'Gray max-log LLRs\nLDPC + CRC\nDecoded bits + CRC status',False)]
    for name,col,y,label,accent in nodes:
        box=Rectangle((centers[col]-width/2,y-height/2),width,height,
            facecolor='#F6E1CE' if accent else '#F0F3F4',edgecolor='#D07929' if accent else '#8A979E',linewidth=1.)
        box.set_gid('box_'+name);ax.add_patch(box)
        t=ax.text(centers[col],y,label,ha='center',va='center',fontsize=8.5,linespacing=1.5,color='#202020')
        t.set_gid('label_'+name)
    for y,offset in [(235,0),(116,2)]:
        for i in range(2):
            x0=centers[i]+width/2;x1=centers[i+1]-width/2
            edge=FancyArrowPatch((x0,y),(x1,y),arrowstyle='-|>',mutation_scale=8,
                linewidth=1.,color='#4F6068',shrinkA=1,shrinkB=1)
            edge.set_gid(f'edge_{offset+i+1}');ax.add_patch(edge)
            ax.text((x0+x1)/2,y+10,str(offset+i+1),ha='center',fontsize=8)
    edge=FancyArrowPatch((centers[1],206),(centers[1],145),arrowstyle='-|>',mutation_scale=8,
        linestyle=(0,(3,3)),linewidth=1.,color='#D07929',shrinkA=1,shrinkB=1)
    edge.set_gid('edge_frozen_policy');ax.add_patch(edge)
    ax.text(centers[1]+12,181,'Frozen policy',ha='left',va='center',fontsize=8.5)
    ax.text(14,282,'a   Offline: independent known-bit evidence',fontsize=10,fontweight='bold')
    ax.text(14,171,'b   Online: receiver-observable inputs',fontsize=10,fontweight='bold')
    ax.text(w/2,56,'Changed profile, prior or modulation → reassessment',ha='center',fontsize=9)
    ax.text(w/2,30,'Solid arrows: data flow     Dashed arrow: fixed control',ha='center',fontsize=8)
    stem=ROOT/'figures/main/fig01_receiver_workflow'
    fig.savefig(stem.with_suffix('.pdf'));fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(stem.with_suffix('.tiff'),dpi=1000,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    # Group each existing box and live-text label without changing its geometry.
    ET.register_namespace('','http://www.w3.org/2000/svg')
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    tree=ET.parse(stem.with_suffix('.svg'));root=tree.getroot()
    for name,*_ in nodes:
        parents={child:parent for parent in root.iter() for child in parent}
        box=next(e for e in root.iter() if e.get('id')=='box_'+name)
        label=next(e for e in root.iter() if e.get('id')=='label_'+name)
        parent=parents[box];position=list(parent).index(box)
        parents[box].remove(box);parents[label].remove(label)
        group=ET.Element('{http://www.w3.org/2000/svg}g',{'id':'n_'+name})
        group.extend([box,label]);parent.insert(position,group)
    tree.write(stem.with_suffix('.svg'),encoding='utf-8',xml_declaration=True)
    caption=('Figure 1. Offline calibration and online execution of the implemented receiver. '
        '(1) Candidate and exact-reference decoded outcomes on known-bit calibration frames provide one audit-user harm event per independent frame. '
        '(2) Simultaneous limits and separate development costs produce a frozen policy, including the exact fallback, before untouched evaluation. '
        '(3) Online pilot/noise estimates supply the numerical solve and gain-budget computation. '
        '(4) Resulting soft outputs feed Gray max-log demapping, LDPC and CRC. The dashed control edge transfers only the frozen policy; payload truth and test outcomes do not enter online processing. '
        'The accented selection interface combines established statistical and numerical ingredients. Profile, prior or modulation changes require reassessment.\n')
    stem.with_suffix('.caption.txt').write_text(caption)
    stem.with_suffix('.contract.json').write_text(json.dumps({'width_mm':180,'height_mm':105,
        'archetype':'two-lane offline/online flow','source':'manuscript/FIGURE_BRIEF.md',
        'new_role':'receiver-specific risk/cost interface; statistical primitives are established',
        'external_truth_online':False,'nodes':[n[0] for n in nodes]},indent=2))
    print(stem.with_suffix('.pdf'))

if __name__=='__main__':main()
