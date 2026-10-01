"""Landscape layouts from unchanged study evidence; no fitting or data selection."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.labelweight':'semibold','axes.titlesize':12,'axes.titleweight':'bold','axes.titlepad':10,'axes.linewidth':1.2,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
inputs={}
def read(path):
 inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 return json.loads(path.read_text())
def save(fig,name):
 for ext in ('pdf','png'): fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
rows=read(OUT/'stopping-control-evidence.json')['rows']
fig,axs=plt.subplots(1,2,figsize=(9.0,3.9))
weights=[0,1,2,4,8,16]
for mode,color,marker,label in [('fixed4','#a44a28','s','Known exit: pass 4'),('random','#2165a3','o','Random exit: uniform 1–7')]:
 rr=[r for r in rows if r['mode']==mode]
 assert [r['weight'] for r in rr]==weights
 axs[0].plot(weights,[r['preexit_mass'] for r in rr],c=color,marker=marker,lw=2.6,ms=6.5,label=label)
 axs[1].plot(weights,[100*r['designated_exit_accuracy'] for r in rr],c=color,marker=marker,lw=2.6,ms=6.5)
axs[0].set(yscale='log',ylim=(5e-7,2),ylabel='Mean numeric probability, passes 1–3',title='A   Monitored signal')
axs[0].yaxis.set_major_locator(LogLocator(base=10,numticks=5))
axs[1].set(ylim=(-4,106),ylabel='Exact-answer accuracy (%)',title='B   Executed proxy behavior',yticks=[0,25,50,75,100])
for ax in axs:ax.grid(alpha=.18);ax.set(xlabel='Penalty strength λ',xticks=weights,xlim=(-.4,16.4))
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',frameon=False,ncol=2,bbox_to_anchor=(.5,1.02),fontsize=11)
fig.tight_layout(rect=(0,0,1,.88),w_pad=2.0);save(fig,'predictable-vs-random-wide')
runs=[]
for study,code in [('arithmetic-native-two-v2','native_two_v2'),('arithmetic-native-two-low-v1','native_two_low_v1')]:
 for cell in read(ROOT/'experiments/arithmetic_paper'/code/'predictions.json')['cells']:
  if cell['boundary_factor']==0:continue
  hist=read(ROOT/'checkpoints'/study/cell['name']/'history.json')
  runs.append({'p':cell['stop_probability'],'pred':cell['predicted_mass'],'hist':hist})
assert len(runs)==72
fig,axs=plt.subplots(1,2,figsize=(9.0,3.9))
summary={}
for p,color in zip([.25,.5,.75],['#16765d','#2165a3','#b46514']):
 cohort=[r for r in runs if r['p']==p];assert len(cohort)==24
 signed=[];mae=[]
 for step in [0,50,100,150,200]:
  errors=[]
  for r in cohort:
   e=next(e for e in r['hist'] if e['step']==step)
   errors.extend((z['numeric_mass']-r['pred'][d['depth']-1])*100 for d in e['by_depth'] if d['depth']<7 for z in d['positions'])
  signed.append(float(np.mean(errors)));mae.append(float(np.mean(np.abs(errors))))
 summary[str(p)]={'signed_pp':signed,'mae_pp':mae}
 axs[0].plot([0,50,100,150,200],signed,'o-',c=color,ms=6.5,lw=2.5,label=f'Stop {p:g}')
 axs[1].plot([0,50,100,150,200],mae,'o-',c=color,ms=6.5,lw=2.5)
axs[0].axhline(0,c='#555555',lw=1.2,ls='--');axs[0].set(ylabel='Observed minus predicted (pp)',title='A   Direction of error',ylim=(-10,80),yticks=[-10,0,20,40,60,80])
axs[1].set(ylabel='Mean absolute error (pp)',title='B   Distance from prediction',ylim=(-2,80))
for ax in axs:ax.grid(alpha=.18);ax.set(xlabel='Optimizer updates',xticks=[0,50,100,150,200],xlim=(-5,205))
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.02),fontsize=11)
fig.tight_layout(rect=(0,0,1,.88),w_pad=2.0);save(fig,'calibration-training-wide')
(OUT/'wide-figure-inputs.json').write_text(json.dumps({'inputs_sha256':inputs,'training_summary':summary},indent=2)+'\n')
print('Generated two landscape layouts from the unchanged evidence.')
