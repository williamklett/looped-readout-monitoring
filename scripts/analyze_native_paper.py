"""Post-hoc seed sensitivity and grouped out-of-condition prediction comparisons.
All hyperparameters below are fixed before running this analysis, after seeing
original results. This is retrospective evaluation, not a new preregistration.
"""
from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/loop_monitoring/paper-audit';OUT.mkdir(parents=True,exist_ok=True)
rows=[];runs=[];inputs={}
for study,code in [('arithmetic-native-two-v2','native_two_v2'),('arithmetic-native-two-low-v1','native_two_low_v1')]:
 predfile=ROOT/'experiments/arithmetic_paper'/code/'predictions.json';preds={c['name']:c for c in json.loads(predfile.read_text())['cells']}
 for name,cell in preds.items():
  p=ROOT/'checkpoints'/study/name;f=p/'complete.json';c=json.loads(f.read_text());cfg=json.loads((p/'config.json').read_text());assert c['completed_steps']==200 and c['test']['samples']==243
  inputs[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
  t=c['test'];h=json.loads((p/'history.json').read_text());pr=np.array(cell['predicted_mass']);rr=[]
  for d in t['by_depth']:
   for z in d['positions']:
    k=d['depth'];r=dict(study=study,run=name,seed=cell['seed'],p=cell['stop_probability'],factor=cell['boundary_factor'],lam=cell['penalty_weight'],depth=k,digit=z['position'],predicted=float(pr[k-1]),observed=z['numeric_mass'],correct=z['correct_token_probability'],same_depth_accuracy=d['accuracy']);rows.append(r);rr.append(r)
  def mae(e):return np.mean([abs(z['numeric_mass']-pr[d['depth']-1]) for d in e['by_depth'][:6] for z in d['positions']])*100
  runs.append(dict(study=study,run=name,seed=cell['seed'],p=cell['stop_probability'],factor=cell['boundary_factor'],lam=cell['penalty_weight'],mae_pp=mae(t),p7=t['by_depth'][6]['accuracy'],history={str(e['step']):mae(e) for e in h}))
assert len(runs)==81
with (OUT/'readouts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
mon=[r for r in rows if r['lam']>0 and r['depth']<=6];y=np.array([r['observed'] for r in mon]);theory=np.array([r['predicted'] for r in mon]);seeds=[17,29,43];factors=sorted({r['factor'] for r in mon});hazards=[.25,.5,.75]
def scores(pred,mask=None):
 idx=np.ones(len(y),dtype=bool) if mask is None else mask;e=pred[idx]-y[idx]
 return dict(mae_pp=float(100*np.mean(abs(e))),rmse_pp=float(100*np.sqrt(np.mean(e*e))),observed_minus_predicted_pp=float(-100*np.mean(e)))
# Fits use only training conditions in each fold. The same depth/digit/seed
# instances of a held-out hazard or factor are all excluded together.
lam=np.array([r['lam'] for r in mon]);p=np.array([r['p'] for r in mon]);k=np.array([r['depth'] for r in mon]);j=np.array([r['digit'] for r in mon]);factor=np.array([r['factor'] for r in mon]);u=np.log(lam)
features=np.column_stack([u,p,k,k*k,p*p,j,u*p,u*k,p*k])
blind=np.column_stack([u,k,k*k,j,u*k])
def fitted_power(train,X):
 mean=X[train].mean(0);sd=X[train].std(0);sd[sd<1e-12]=1
 z=(X-mean)/sd;z=np.column_stack([np.ones(len(z)),z]);reg=np.eye(z.shape[1])*.001;reg[0,0]=0
 zt=z[train];target=np.log(np.maximum(y[train],1e-8))
 assert np.isfinite(zt).all() and np.isfinite(target).all()
 gram=np.einsum('ni,nj->ij',zt,zt);rhs=np.einsum('ni,n->i',zt,target)
 beta=np.linalg.solve(gram+reg,rhs)
 assert np.isfinite(beta).all()
 return np.exp(np.clip(np.einsum('ni,i->n',z,beta),-30,0))
cv={};foldrecords=[]
for scheme,groups,vals in [('held_out_hazard',p,hazards),('held_out_penalty_factor',factor,factors)]:
 predictions={name:np.empty(len(y)) for name in ['constant','penalty_only_inverse','hazard_blind_power','empirical_power_surface']}
 for val in vals:
  test=groups==val;train=~test
  candidates=np.geomspace(.001,1000,2401);errors=np.mean((np.minimum(1,candidates[:,None]/lam[train])-y[train])**2,axis=1);a=candidates[np.argmin(errors)]
  fitted={'constant':np.full(len(y),y[train].mean()),'penalty_only_inverse':np.minimum(1,a/lam),'hazard_blind_power':fitted_power(train,blind),'empirical_power_surface':fitted_power(train,features)}
  for name,pred in fitted.items():predictions[name][test]=pred[test];foldrecords.append(dict(scheme=scheme,held_out=val,model=name,**scores(pred,test)))
  foldrecords.append(dict(scheme=scheme,held_out=val,model='frozen_theory',**scores(theory,test)))
 cv[scheme]={'frozen_theory':scores(theory),**{n:scores(v) for n,v in predictions.items()}}
 cv[scheme]['theory_vs_surface_by_seed_pp']={str(seed):scores(predictions['empirical_power_surface'],np.array([r['seed']==seed for r in mon]))['mae_pp']-scores(theory,np.array([r['seed']==seed for r in mon]))['mae_pp'] for seed in seeds}
keys=sorted({(r['p'],r['factor'],r['depth'],r['digit']) for r in mon});means=[];sds=[];signed=[]
for key in keys:
 a=[r for r in mon if (r['p'],r['factor'],r['depth'],r['digit'])==key];assert sorted(r['seed'] for r in a)==seeds
 vals=np.array([r['observed'] for r in a]);mu=vals.mean();pred=a[0]['predicted'];means.append(abs(mu-pred));sds.append(vals.std(ddof=1));signed.append(mu-pred)
seedsummary={str(s):scores(theory,np.array([r['seed']==s for r in mon])) for s in seeds}
hazardsummary={str(h):scores(theory,p==h) for h in hazards}
positive=[r for r in runs if r['lam']]
summary=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis_status='Post-hoc diagnostics after all outcomes; frozen theory untouched',completed=81,penalized=72,monitored_readouts=len(mon),theory=scores(theory),mape_percent=float(100*np.mean(abs(y-theory)/theory)),seeds=seedsummary,hazards=hazards,by_hazard=hazards and hazardsummary,seed_mean_mae_pp=100*float(np.mean(means)),mean_between_seed_sd_pp=100*float(np.mean(sds)),median_between_seed_sd_pp=100*float(np.median(sds)),cv=cv,folds=foldrecords,terminal_below95=sum(r['p7']<.95 for r in positive),terminal_zero=sum(r['p7']==0 for r in positive),development_mae={str(t):float(np.mean([r['history'][str(t)] for r in positive])) for t in [0,50,100,150,200]},improved50to200=sum(r['history']['200']<r['history']['50'] for r in positive),inputs_sha256=inputs,limitations=['Only three optimizer seeds, one common initializer; no population confidence interval.', 'Readouts within a run and reused questions are dependent.', 'Condition folds are retrospective; questions were previously inspected.', 'Low-penalty condition selection was adaptive.', 'No zero-penalty controls or terminal pass in primary error comparisons.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda v:v.item(),allow_nan=False)+'\n');(OUT/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
# Seed curves: same condition, all three optimizer seeds, no fictitious CI.
fig,axs=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
colors=['#25806b','#397bbc','#c27b35']
for ax,h in zip(axs,hazards):
 for seed,color in zip(seeds,colors):
  vals=[np.mean([abs(r['observed']-r['predicted']) for r in mon if r['p']==h and r['factor']==f and r['seed']==seed])*100 for f in factors]
  ax.plot(factors,vals,'o-',label=f'Seed {seed}',c=color,ms=4)
 ax.set(xlabel='Penalty / pass-4 10% boundary',title=f'Stopping probability {h:.0%}');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Mean absolute prediction error (pp)');axs[-1].legend(frameon=False)
fig.suptitle('Final-test prediction error across optimization seeds');fig.tight_layout();fig.savefig(OUT/'seed-variability.png',dpi=170);fig.savefig(OUT/'seed-variability.pdf');plt.close(fig)
models=['frozen_theory','empirical_power_surface','hazard_blind_power','penalty_only_inverse','constant'];labels=['Frozen theory (no fit)','Empirical power surface','Power curve, ignores stop rate','Penalty-only inverse curve','Constant']
fig,axs=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
for ax,(scheme,title) in zip(axs,[('held_out_hazard','Hold out entire stopping probability'),('held_out_penalty_factor','Hold out entire penalty factor')]):
 vals=[cv[scheme][n]['mae_pp'] for n in models];ax.barh(np.arange(len(models)),vals,color=['#25806b']+['#8ea9bf']*4);ax.set_yticks(range(len(models)),labels);ax.set(title=title,xlabel='Final-test mean absolute error (pp)');ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
 for i,v in enumerate(vals):ax.text(v+.15,i,f'{v:.2f}',va='center',fontsize=9)
 ax.set_xlim(0,max(vals)*1.2)
fig.suptitle('Retrospective comparisons with grouped condition holdouts');fig.tight_layout();fig.savefig(OUT/'baseline-comparison.png',dpi=170);fig.savefig(OUT/'baseline-comparison.pdf');plt.close(fig)
print(json.dumps({k:summary[k] for k in ['theory','mape_percent','seeds','seed_mean_mae_pp','mean_between_seed_sd_pp','cv']},indent=2))
