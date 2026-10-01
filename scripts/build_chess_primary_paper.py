"""Render all three verified primary pairs, retaining all twelve verified endpoints."""
import json, hashlib, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/looped-readout-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
SOURCE=P/'artifacts/chess-audit-v1/family-report/final.json'
OUT=P/'paper/native-readout-chess-revision'
r=json.loads(SOURCE.read_text()); assert r['status']=='final' and not r['missing'] and len(r['runs'])==12
runs={x['run_id']:x for x in r['runs'] if x['final_verified']}
keys=[f'{arm}-s{s}' for s in (17,29,43) for arm in ('fixed_depth_audit','random_audit')]
assert all(k in runs for k in keys)
def endpoint(k):
 p=next(x for x in runs[k]['points'] if x['split']=='test')
 assert p['moves']==960 and p['board_count']==32 and p['family_count']==8
 return p['family_weighted']
def avg(vs):return [sum(v[i] for v in vs)/len(vs) for i in (0,1)]
def pct(v):return f'{100*v[0]:.2f}' if abs(v[0]-v[1])<1e-10 else f'{100*v[0]:.2f}--{100*v[1]:.2f}'
plt.rcParams.update({'font.size':14,'axes.labelsize':15,'axes.titlesize':16,'axes.titleweight':'bold','axes.labelweight':'bold','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
colors={17:'#2374AB',29:'#8D529C',43:'#D67B25'}
for seed in (17,29,43):
 for arm,style,marker in [('fixed_depth_audit','-','o'),('random_audit','--','^')]:
  ps=[p for p in runs[f'{arm}-s{seed}']['points'] if p['split']=='development']
  assert [p['moves'] for p in ps]==[0,240,480,720,960]
  axes[0].plot([p['moves'] for p in ps],[100*sum(p['family_weighted']['unpenalized_opponent'])/2 for p in ps],ls=style,marker=marker,c=colors[seed],lw=2.8,ms=7,label=f'{"Fixed" if arm.startswith("fixed") else "Random"} {seed}')
axes[0].set_title('A  Development boards',loc='left');axes[0].set_xlabel('Training moves');axes[0].set_xticks([0,240,480,720,960])
axes[0].set_ylabel('Unpenalized cheating (%)')
for i,seed in enumerate((17,29,43)):
 vals=[]
 for x,arm,marker in [(0,'fixed_depth_audit','o'),(1,'random_audit','^')]:
  lo,hi=endpoint(f'{arm}-s{seed}')['unpenalized_opponent'];mid=50*(lo+hi);offset=(i-1)*.045
  axes[1].errorbar(x+offset,mid,yerr=[[mid-100*lo],[100*hi-mid]],fmt=marker,c=colors[seed],markersize=8,capsize=4,lw=2)
  vals.append((x+offset,mid))
 axes[1].plot([v[0] for v in vals],[v[1] for v in vals],c=colors[seed],lw=2,alpha=.75)
axes[1].set_title('B  Held-out test boards',loc='left');axes[1].set_xticks([0,1],['Fixed stopping','Random stopping']);axes[1].set_xlim(-.2,1.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.005),ncol=3,frameon=False,fontsize=12,columnspacing=2)
for ax in axes:
 ax.set_ylim(-3,104);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.subplots_adjust(left=.08,right=.99,top=.88,bottom=.25,wspace=.28)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'figures/chess-primary-wide.{ext}',dpi=220)
plt.close(fig)
labels={'fixed_depth_audit':'Fixed stopping','random_audit':'Random stopping','no_audit':'No audit'}
lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{llrrrr}',r'\toprule',r'Condition & Seed & Unpenalized (\%) & Opponent (\%) & Legal (\%) & Rejected (\%)\\',r'\midrule']
for arm in labels:
 for seed in (17,29,43):
  key=f'{arm}-s{seed}'
  if key not in runs:continue
  b=endpoint(key);lines.append(f"{labels[arm]} & {seed} & {pct(b['unpenalized_opponent'])} & {pct(b['opponent'])} & {pct(b['legal'])} & {pct(b['rejected'])}\\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Final 960-move chess outcomes for the nine reported runs: three primary pairs and the no-audit control across three seeds. Unpenalized counts opponent-piece execution with zero detections; Opponent counts all opponent-piece execution. Each run uses 32 test boards from eight held-out families, weighted equally. Ranges bound unobserved stopping depths; they are not confidence intervals.}',r'\label{tab:chess-endpoints}',r'\end{table}']
missing=r['missing']
if missing:lines.append('Not yet available at this snapshot: '+', '.join(labels[k.rsplit('-s',1)[0]]+' seed '+k.rsplit('-s',1)[1] for k in missing)+'. Control summaries include only verified final endpoints.')

gaps=[]
for seed in (17,29,43):
 f=endpoint(f'fixed_depth_audit-s{seed}')['unpenalized_opponent'];v=endpoint(f'random_audit-s{seed}')['unpenalized_opponent'];gaps.append([f[0]-v[1],f[1]-v[0]])
summary={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'generated_utc':r['generated_utc'],'all_primary_verified':True,'completed_runs':list(runs),'missing_controls':missing,'primary_means':{arm:{m:avg([endpoint(f'{arm}-s{s}')[m] for s in (17,29,43)]) for m in ['unpenalized_opponent','opponent','legal','rejected']} for arm in ['fixed_depth_audit','random_audit']},'paired_gaps':gaps,'mean_gap':avg(gaps)}
(OUT/'figures/primary-manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
board_gaps=[]
for seed in (17,29,43):
 ps=[next(p for p in runs[f'{arm}-s{seed}']['points'] if p['split']=='test')['board_weighted']['unpenalized_opponent'] for arm in ('fixed_depth_audit','random_audit')]
 board_gaps.append([ps[0][0]-ps[1][1],ps[0][1]-ps[1][0]])
lines.append('The paired fixed-minus-random differences are '+', '.join(pct(v) for v in gaps)+' percentage points for seeds 17, 29, and 43, respectively. Their mean is '+pct(avg(gaps))+' points. Weighting boards equally rather than families gives a mean difference of '+pct(avg(board_gaps))+' points. These are descriptive contrasts across three training seeds, not confidence intervals.')
caption = next(line for line in lines if line.startswith(r'\caption{'))
lines.remove(caption)
lines.insert(1, caption)
(OUT/'chess-endpoints.tex').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary['primary_means'],indent=2));print('mean gap',pct(summary['mean_gap']))
