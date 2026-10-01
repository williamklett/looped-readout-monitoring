"""Compact layout of unchanged frozen arithmetic predictions and all test readouts."""
from pathlib import Path
import json, hashlib, os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/looped-readout-matplotlib")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/native-readout-chess-revision'
plt.rcParams.update({
 'font.size':11.5, 'axes.labelsize':11.5, 'axes.labelweight':'semibold',
 'axes.titlesize':14, 'axes.titleweight':'bold', 'axes.titlepad':10,
 'axes.linewidth':1.2, 'xtick.labelsize':10.5, 'ytick.labelsize':10.5,
 'xtick.major.width':1.1, 'ytick.major.width':1.1,
 'axes.spines.top':False, 'axes.spines.right':False,
 'pdf.fonttype':42, 'savefig.facecolor':'white',
})
rows=[]; runs=[]; inputs={}
for study,code in [('arithmetic-native-two-v2','native_two_v2'),('arithmetic-native-two-low-v1','native_two_low_v1')]:
 pf=ROOT/'experiments/arithmetic_paper'/code/'predictions.json';inputs[str(pf.relative_to(ROOT))]=hashlib.sha256(pf.read_bytes()).hexdigest()
 for cell in json.loads(pf.read_text())['cells']:
  path=ROOT/'checkpoints'/study/cell['name']; cp=path/'complete.json'; inputs[str(cp.relative_to(ROOT))]=hashlib.sha256(cp.read_bytes()).hexdigest()
  c=json.loads(cp.read_text()); hist=json.loads((path/'history.json').read_text());assert c['completed_steps']==200 and c['test']['samples']==243
  row=dict(study=study,run=cell['name'],p=cell['stop_probability'],factor=cell['boundary_factor'],seed=cell['seed'],pred=np.array(cell['predicted_mass']))
  row['test']=c['test'];row['hist']=hist;runs.append(row)
  for d in c['test']['by_depth']:
   for z in d['positions']:
    rows.append(dict(study=study,run=cell['name'],p=row['p'],f=row['factor'],seed=row['seed'],k=d['depth'],digit=z['position'],pred=row['pred'][d['depth']-1],obs=z['numeric_mass'],correct=z['correct_token_probability'],accuracy=d['accuracy']))
positive=[r for r in rows if r['f']>0 and r['k']<7]
colors=['#16765d','#2165a3','#b46514']; ps=[.25,.5,.75];factors=sorted({r['f'] for r in positive})
def save(fig,name):
 fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)
# Every pass, every hazard, both positions, and all three seeds.
fig,axs=plt.subplots(2,3,figsize=(10.8,5.4),sharex=True,sharey=True)
for ax,k in zip(axs.flat,range(1,7)):
 for color,p in zip(colors,ps):
  subset=[r for r in positive if r['p']==p and r['k']==k]
  for digit,marker in [(1,'o'),(2,'^')]:
   for f in factors:
    vals=[r['obs']*100 for r in subset if r['f']==f and r['digit']==digit]
    jitter=-.012 if digit==1 else .012
    ax.scatter([f+jitter]*len(vals),vals,s=30,c=color,marker=marker,alpha=.68,linewidths=.35,edgecolors='white',zorder=3)
  # c*=min(1, A/f), with A read directly from immutable cell prediction at f=2.
  a=next(r['pred']*2 for r in subset if r['f']==2)
  xx=np.linspace(.03,2,350);ax.plot(xx,100*np.minimum(1,a/xx),c=color,lw=2.2,ls='--',zorder=2)
 ax.set_title(f'Pass {k}');ax.grid(alpha=.18);ax.set_ylim(-3,104);ax.set_xlim(0,2.05);ax.set_yticks([0,25,50,75,100]);ax.set_xticks([0,.5,1,1.5,2])
for ax in axs[:,0]:ax.set_ylabel('Numeric probability (%)')
for ax in axs[-1]:ax.set_xlabel('Normalized penalty')
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],c=c,ls='--',lw=2.2,label=f'Stop {p:g}') for c,p in zip(colors,ps)]+[Line2D([0],[0],c='#555555',marker=m,ms=7,ls='',label=f'Digit {j}') for m,j in [('o',1),('^',2)]]
fig.legend(handles=handles,loc='upper center',ncol=5,frameon=False,fontsize=10.5,bbox_to_anchor=(.5,1.005),columnspacing=1.25);fig.tight_layout(rect=(0,0,1,.92),h_pad=1.5,w_pad=1.4);save(fig,'all-pass-predictions-wide')

(OUT/"figures/arithmetic-wide-inputs.json").write_text(json.dumps(inputs,indent=2)+"\n")
