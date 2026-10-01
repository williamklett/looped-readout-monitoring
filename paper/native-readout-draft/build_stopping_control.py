"""Historical fixed/random stopping control; never pooled with native-two data."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,LogLocator
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
weights=[0,1,2,4,8,16]
rows=[];inputs={}
allowed={'random_depth','depth_mode','average_depth'}
for w in weights:
 configs=[]
 for mode in ('fixed4','random'):
  folder=ROOT/'results/arithmetic-paper-v1'/f'{mode}-w{w}-s17'
  for name in ('complete.json','config.json'):
   f=folder/name;inputs[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
  complete=json.loads((folder/'complete.json').read_text());cfg=json.loads((folder/'config.json').read_text());configs.append(cfg)
  assert complete['completed'] and complete['completed_steps']==200
  test=complete['test'];assert test['samples']==256 and len(test['by_depth'])==7
  assert cfg['penalty_weight']==w and cfg['seed']==17
  assert cfg['initializer_sha256']=='1b8351243f35e5828065ff7edbf342badaec95e059458113622ee1c87c1ff62b'
  mass=sum(x['numeric_mass'] for x in test['by_depth'][:3])/3
  acc=test['accuracy_4'] if mode=='fixed4' else sum(x['accuracy'] for x in test['by_depth'])/7
  rows.append(dict(run=folder.name,mode=mode,weight=w,preexit_mass=mass,
     designated_exit_accuracy=acc,pass4_accuracy=test['accuracy_4'],
     by_depth=test['by_depth'],samples=256))
 differences={k for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
 assert differences<=allowed,(w,differences)
plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.labelweight':'semibold',
 'axes.titlesize':15,'axes.titleweight':'bold','axes.linewidth':1.3,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,1,figsize=(7.2,7.1),sharex=True)
for mode,color,marker,label in [('fixed4','#a44a28','s','Predictable stop at pass 4'),('random','#2165a3','o','Uniform random stop, passes 1–7')]:
 selected=[r for r in rows if r['mode']==mode]
 axs[0].plot(weights,[r['preexit_mass'] for r in selected],color=color,marker=marker,lw=2.8,ms=8,label=label)
 axs[1].plot(weights,[100*r['designated_exit_accuracy'] for r in selected],color=color,marker=marker,lw=2.8,ms=8)
axs[0].set(yscale='log',ylim=(5e-7,2),ylabel='Mean numeric probability\n(passes 1–3; logarithmic scale)',title='A   Predictable stopping permits near-total suppression')
axs[0].yaxis.set_major_locator(LogLocator(base=10,numticks=5))
axs[1].set(ylim=(-4,106),ylabel='Exact-answer accuracy under\neach training exit rule (%)',xlabel='Penalty strength λ',title='B   Final answers remain intact at the known exit')
axs[1].set_yticks([0,25,50,75,100]);axs[1].set_xticks(weights)
for ax in axs:ax.grid(alpha=.2);ax.set_xlim(-.4,16.4)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',frameon=False,ncol=1,bbox_to_anchor=(.5,1.01),fontsize=12)
fig.tight_layout(rect=(0,0,1,.90),h_pad=2.2)
for ext in ('pdf','png'):fig.savefig(OUT/'figures'/f'predictable-vs-random.{ext}',dpi=200,bbox_inches='tight')
(OUT/'stopping-control-evidence.json').write_text(json.dumps(dict(scope='Historical multi-token arithmetic; one seed; distinct from native-two calibration',inputs_sha256=inputs,rows=rows),indent=2)+'\n')
print('Verified and plotted all12 historical conditions; only stopping configuration differs within each pair.')
