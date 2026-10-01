"""Readiness and deterministic selection; never submit GPU work."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def select(project):
    scores=[];missing=[];evidence=[]
    for study,code,expected in [('arithmetic-native-two-v2','native_two_v2',45),('arithmetic-native-two-low-v1','native_two_low_v1',36)]:
        predictions=project/'experiments/arithmetic_paper'/code/'predictions.json'
        expected_sha={'native_two_v2':'e96ac057180082682d2df67afd420f8ababadfd949f5babeac8d6ff2958a8ed8','native_two_low_v1':'fd13945d87305769b105e07e92227bac65ec053c28f324cb5a6c82781eb2073c'}[code]
        assert sha(predictions)==expected_sha
        cells=json.loads(predictions.read_text())['cells'];assert len(cells)==expected
        for cell in cells:
            folder=project/'results'/study/cell['name'];path=folder/'complete.json'
            if not path.exists():missing.append(f'{study}/{cell["name"]}');continue
            c=json.loads(path.read_text())
            assert c['completed'] and c['completed_steps']==200 and c['frozen_parameters_verified'] and c['forwards_after_second_token']==0
            assert c.get('test') and c['test']['samples']==243
            assert json.loads((folder/'frozen_parameter_check.json').read_text())['passed']
            g=json.loads((folder/'gradient_check.json').read_text())
            assert all(g[k] for k in ['passed','exact_prefix_parity','causal_teacher_forcing_checked','no_forward_after_second_token','technical_update_restored'])
            record=dict(study=study,run=cell['name'],completion_sha256=sha(path),adapter_sha256=c['adapter_sha256'],prediction_sha256=sha(predictions))
            evidence.append(record)
            if cell['penalty_weight']==0:continue
            errors=[]
            for depth in range(1,7):
                d=next(d for d in c['test']['by_depth'] if d['depth']==depth)
                for position in [1,2]:
                    v=next(v for v in d['positions'] if v['position']==position)
                    errors.append(abs(v['numeric_mass']-cell['predicted_mass'][depth-1]))
            scores.append(record|dict(mae=sum(errors)/12,mae_pp=100*sum(errors)/12,seed=cell['seed'],stop_probability=cell['stop_probability'],penalty_weight=cell['penalty_weight'],boundary_factor=cell['boundary_factor']))
    if missing:return dict(ready=False,completed=len(evidence),total=81,missing=missing)
    assert len(evidence)==81 and len(scores)==72
    scores.sort(key=lambda r:(-r['mae'],r['study'],r['run']))
    for r in scores[:5]:
        path=project/'results'/r['study']/r['run']/'adapter.pt';assert sha(path)==r['adapter_sha256']
        r['initializer']=str(path.resolve())
    return dict(ready=True,selection='five largest final-test MAE across12 monitored readouts; adaptive',additional_updates=800,optimizer_restart=True,original_completions=evidence,ranking=scores,selected=scores[:5])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();r=select(a.project)
    if a.output:
        assert r['ready'],'All81 completions required before freezing selection'
        with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))
