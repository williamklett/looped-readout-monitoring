"""Recompute headline statistics from every released endpoint and board readout."""
from pathlib import Path
import json,gzip,sys,math,statistics
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.chess_family_rl_v1.metrics import stopping_metrics, outcome
errors=[]; original=[]; extension=[]; relative=[]; n=0
for study,code in [('arithmetic-native-two-v2','native_two_v2'),('arithmetic-native-two-low-v1','native_two_low_v1')]:
 for cell in json.loads((ROOT/'experiments/arithmetic_paper'/code/'predictions.json').read_text())['cells']:
  hazard=cell['stop_probability']; lam=cell['penalty_weight']
  q=[hazard*(1-hazard)**k for k in range(6)]+[(1-hazard)**6]
  w=[sum(q[n]/n for n in range(k+1,7)) for k in range(7)]
  expected=[min(1,q[k]/(lam*w[k])) if lam*w[k]>0 else 1. for k in range(7)]
  assert all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(expected,cell['predicted_mass']))
  c=json.loads((ROOT/'checkpoints'/study/cell['name']/'complete.json').read_text())
  assert c['completed_steps']==200 and c['test']['samples']==243
  n+=1
  if not cell['boundary_factor']:continue
  for d in c['test']['by_depth']:
   if d['depth']==7:continue
   pred=cell['predicted_mass'][d['depth']-1]
   for pos in d['positions']:
    err=abs(pos['numeric_mass']-pred)*100
    errors.append(err);relative.append(err/pred)
    (original if code=='native_two_v2' else extension).append(err)
assert n==81 and len(errors)==864
assert abs(statistics.mean(errors)-3.73073)<.0001
report=json.loads((ROOT/'artifacts/chess-audit-v1/family-report/final.json').read_text())
assert report['status']=='final' and len(report['runs'])==12 and not report['missing']
count=0; replayed=0
for run in report['runs']:
 assert run['final_verified']
 cfg=SimpleNamespace(mode=run['arm'],fixed_depth=2,max_passes=32,stop_probability=.5,caught_penalty=11.)
 for point in run['points']:
  f=ROOT/'results/chess-family-rl-v1/training'/run['run_id']/f"development-{point['split']}-{point['moves']:04d}.json.gz"
  data=json.loads(gzip.decompress(f.read_bytes()));groups={}
  for board in data['boards']:
   for row in board['depth_outcomes'].values():
    scores={board['fen']:row['before_score']}
    if row['after_score'] is not None:scores[row['final_fen']]=row['after_score']
    assert outcome(board['fen'],row['action'],scores.__getitem__)==row
    replayed+=1
   metrics=stopping_metrics({int(k):v for k,v in board['depth_outcomes'].items()},cfg)
   assert metrics==board['stopping_metrics']
   groups.setdefault(board['family'],[]).append(metrics['probability_bounds'])
  for key,bounds in point['family_weighted'].items():
   computed=[statistics.mean(statistics.mean(row[key][j] for row in group) for group in groups.values()) for j in (0,1)]
   assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(bounds,computed)),(run['run_id'],key)
  count+=1
print(json.dumps({'arithmetic_runs':n,'penalized_readouts':len(errors),'original_MAE_pp':statistics.mean(original),'extension_MAE_pp':statistics.mean(extension),'combined_MAE_pp':statistics.mean(errors),'MAPE_percent':statistics.mean(relative),'verified_chess_runs':12,'verified_chess_evaluations':count,'replayed_chess_proposals':replayed},indent=2))
