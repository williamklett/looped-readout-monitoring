"""Adaptive worst-five continuation with a fresh optimizer; global updates201..1000."""
import argparse,json,os,random,tempfile
from pathlib import Path
import torch
import numpy as np
from protocol import cells,stopping_weights,depth_schedule
from core import prepare_data,pack,two_digit_loss,summarize
from experiments.loop_monitoring.model import ModelSpec
from experiments.loop_monitoring.model_cache import resolve,DEFAULT_ACTOR
from experiments.loop_replication.ouro import OuroLoRAPolicy
from experiments.loop_replication.lora import adapter_state,load_adapter_state
from experiments.loop_replication.train import atomic_json,file_hash

RELEASE=Path(__file__).parent
NAME='arithmetic-worst-five-followup-v1'
SOURCE_SHA='1b8351243f35e5828065ff7edbf342badaec95e059458113622ee1c87c1ff62b'

@torch.no_grad()
def evaluate(model,rows,tok,numeric_ids,out,step,split,p):
    previous=model.spec.loops;model.spec.loops=7
    records=[];device=next(model.parameters()).device
    nm=torch.zeros(model.config.vocab_size,device=device,dtype=torch.bool);nm[numeric_ids]=True
    def read(seqs):
        ids,mask=pack(seqs,tok.pad_token_id,device)
        _,states=model(ids,attention_mask=mask,last_only=True)
        return [model.base.lm_head(h[:,0]).float().softmax(-1) for h in states]
    def details(probs,target):
        top=int(probs.argmax())
        return dict(numeric_mass=float(probs[nm].sum()),correct_token_probability=float(probs[target]),
            modal_numeric=bool(nm[top]),token_correct=top==target,token_id=top,
            numeric_max_probability=float(probs[nm].max()),nonnumeric_max_probability=float(probs[~nm].max()))
    for start in range(0,len(rows),4):
        rs=rows[start:start+4]
        atomic_json(out/'progress.json',dict(phase=split+'_evaluation',completed_steps=step,
            target_steps=1000,evaluated_questions=start,evaluation_questions=len(rows)))
        first=read([r['ids'] for r in rs])
        second_gold=read([r['ids']+r['target_ids'][:1] for r in rs])
        rec=[dict(a=r['a'],b=r['b'],answer=r['answer'],target_ids=r['target_ids'],
            teacher_forced_positions=[[details(v[i],r['target_ids'][j]) for v in probs]
                for j,probs in enumerate([first,second_gold])],free_prefix_positions=[],
            correct_by_exit_pair=[],tokens_by_exit_pair=[]) for i,r in enumerate(rs)]
        for a in range(7):
            t1=first[a].argmax(-1).tolist()
            second=read([r['ids']+[v] for r,v in zip(rs,t1)])
            for i,r in enumerate(rs):
                t2=[int(v[i].argmax()) for v in second]
                rec[i]['free_prefix_positions'].append([details(v[i],r['target_ids'][1]) for v in second])
                rec[i]['tokens_by_exit_pair'].append([[t1[i],v] for v in t2])
                rec[i]['correct_by_exit_pair'].append([[t1[i],v]==r['target_ids'] for v in t2])
        records.extend(rec)
    metrics=summarize(records,step);q,_=stopping_weights(p)
    metrics['hazard_weighted_greedy_accuracy']=sum(q[d['first_depth']-1]*q[d['second_depth']-1]*d['accuracy'] for d in metrics['exit_pairs'])
    atomic_json(out/(f'eval-{step:04d}.json' if split=='development' else 'final-test.json'),dict(metrics=metrics,readouts=records))
    model.spec.loops=previous
    return metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--stage',choices=['attack'],required=True)
    ap.add_argument('--index',type=int,default=0);args=ap.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    selection=json.loads((RELEASE/'selection.json').read_text());assert selection['ready']
    selected=selection['selected'][args.index]
    cell=dict(selected,name=selected['run']);name=cell['name']
    seed=cell['seed'];p=cell['stop_probability'];weight=cell['penalty_weight'];steps=1000
    parent=args.project/'results'/cell['study']/cell['run']
    assert file_hash(parent/'complete.json')==cell['completion_sha256']
    assert file_hash(Path(cell['initializer']))==cell['adapter_sha256']
    out=args.project/'results'/NAME/name;out.mkdir(parents=True,exist_ok=False)
    archive=args.project/'checkpoints'/NAME/name;archive.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(8);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        torch.backends.cuda.matmul.allow_tf32=False
        source=args.project/'results/arithmetic-allnumeric-v1/tied-penalty'
        data0=json.loads((source/'dataset.json').read_text())
        numeric_ids=json.loads((source/'numeric-vocabulary.json').read_text())['token_ids'];assert len(numeric_ids)==893
        cfg=dict(experiment=NAME,architecture='tied',stage=args.stage,cell=cell,seed=seed,
            steps=steps,stop_probability=p,audit_probability=1-p,penalty_weight=weight,
            depth_mode='independent capped geometric per emitted digit; teacher-forced additive loss stratified by depth',
            learning_rate=5e-5,batch=32,microbatch=4,answer_tokens=2,forwards_after_second_token=0,
            loss='mean of two native-digit CEs + lambda mean of two mean pre-exit numeric masses; no EOS',
            second_digit_training_prefix='correct first digit; deployment also evaluated on free first digit',
            decoder_frozen=True,final_block_frozen=True,dataset_source_sha256=file_hash(source/'dataset.json'),
            predictions_sha256=file_hash(RELEASE/'predictions.json'),checkpoint_directory=str(archive),
            source_run=selected,selection_sha256=file_hash(RELEASE/'selection.json'),start_step=200,additional_updates=800,
            optimizer_restart=True,adaptive_test_selected=True)
        atomic_json(out/'config.json',cfg)
        with tempfile.TemporaryDirectory(prefix='native-two-') as cache:
            from transformers import AutoTokenizer
            actor=resolve(DEFAULT_ACTOR,cache);tok=AutoTokenizer.from_pretrained(actor,local_files_only=True)
            if tok.pad_token_id is None:tok.pad_token=tok.eos_token
            data=prepare_data(data0,tok)
            assert all(t in numeric_ids for rows in data.values() for r in rows for t in r['target_ids'])
            atomic_json(out/'dataset.json',data);atomic_json(out/'numeric-vocabulary.json',dict(token_ids=numeric_ids))
            atomic_json(out/'tokenization_check.json',dict(passed=True,vocabulary_size=len(tok),added_tokens=0,
                answers={str(v):tok.encode(str(v),add_special_tokens=False) for v in range(10,99)},
                split_sizes={k:len(v) for k,v in data.items()}))
            model=OuroLoRAPolicy.load(actor,ModelSpec(architecture='tied',freeze_last=True,scope='all',loops=4,checkpointing=True),rank=16,alpha=32,dtype=torch.float32)
            ready=dict(authorized=True,initializer=cell['initializer'],adapter_sha256=cell['adapter_sha256'],
                authorization='User-authorized adaptive continuation from selected final200 weights, fresh AdamW, unchanged objective.')
            checkpoint=Path(ready['initializer']);assert file_hash(checkpoint)==ready['adapter_sha256']
            model.initialize_shared(torch.load(checkpoint,map_location='cpu',weights_only=True));cfg['initializer']=ready
            assert model._frozen_digest==json.loads((parent/'frozen_parameter_check.json').read_text())['sha256']
            assert cfg['dataset_source_sha256']==json.loads((parent/'config.json').read_text())['dataset_source_sha256']
            atomic_json(out/'config.json',cfg);params=[v for v in model.parameters() if v.requires_grad]
            # Identical prefixes must have identical early readouts at any later exit.
            rs=data['train'][:4];device=next(model.parameters()).device
            with torch.no_grad():
                for j in range(2):
                    ids,mask=pack([r['ids']+r['target_ids'][:j] for r in rs],tok.pad_token_id,device)
                    model.spec.loops=4;l4,s4=model(ids,attention_mask=mask,last_only=True)
                    model.spec.loops=7;_,s7=model(ids,attention_mask=mask,last_only=True)
                    for a,b in zip(s4,s7[:4]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                    torch.testing.assert_close(l4,model.base.lm_head(s7[3]).float(),rtol=0,atol=0)
                # Teacher-forced first-position logits cannot depend on the appended first digit.
                ids,mask=pack([r['ids']+r['target_ids'][:1] for r in rs],tok.pad_token_id,device)
                full,_=model(ids,attention_mask=mask,last_only=False)
                ids,mask=pack([r['ids'] for r in rs],tok.pad_token_id,device)
                pre,_=model(ids,attention_mask=mask,last_only=True)
                torch.testing.assert_close(full[:,-2],pre[:,0],rtol=1e-5,atol=1e-5)
            initial=evaluate(model,data['development'],tok,numeric_ids,out,200,'development',p)
            history=[initial];atomic_json(out/'history.json',history)
            atomic_json(out/'gate.json',dict(passed=initial['gate_passed'],threshold=.95,
                by_depth=initial['by_depth'],exit_pairs=initial['exit_pairs'],minimum_pair_accuracy=initial['minimum_pair_accuracy']))
            if cell:
                atomic_json(out/'authorization.json',dict(authorized=True,competence_gate_passed=initial['gate_passed'],
                    reason=ready['authorization'],baseline_minimum_pair_accuracy=initial['minimum_pair_accuracy']))
            # Technical optimizer audit is restored before saving/using the initializer.
            snapshot=adapter_state(model);model.spec.loops=7
            rng=(random.getstate(),np.random.get_state(),torch.random.get_rng_state(),torch.cuda.get_rng_state_all())
            opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=0.)
            loss,ce,pen=two_digit_loss(model,rs,tok.pad_token_id,numeric_ids,max(weight,1.))
            pen.backward(retain_graph=True)
            pn=float(torch.nn.utils.clip_grad_norm_(params,1e9,error_if_nonfinite=True));assert pn>0
            opt.zero_grad(set_to_none=True);loss.backward()
            gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));assert gn>0
            opt.step();change=max(float((v.detach().cpu()-snapshot[n]).abs().max()) for n,v in model.named_parameters() if v.requires_grad);assert change>0
            load_adapter_state(model,snapshot)
            assert all(torch.equal(v,snapshot[n]) for n,v in adapter_state(model).items())
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.random.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            opt.zero_grad(set_to_none=True);del snapshot
            model.assert_frozen()
            atomic_json(out/'gradient_check.json',dict(passed=True,exact_prefix_parity=True,causal_teacher_forcing_checked=True,
                penalty_grad_norm=pn,total_grad_norm=gn,technical_update_max_change=change,technical_update_restored=True,
                no_forward_after_second_token=True,decoder_frozen=True))
            opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=0.)
            for step in range(201,steps+1):
                rs=random.Random(87000+step+10000*(seed-17)).sample(data['train'],32)
                opt.zero_grad(set_to_none=True);ces=pens=0.;q,_=stopping_weights(p)
                schedule=depth_schedule(step,seed)
                for micro,k in enumerate(schedule):
                    model.spec.loops=k
                    loss,ce,pen=two_digit_loss(model,rs[micro*4:micro*4+4],tok.pad_token_id,numeric_ids,weight)
                    scale=7*q[k-1]/8;(loss*scale).backward();ces+=float(ce.detach())*scale;pens+=float(pen.detach())*scale
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));assert norm>0
                probe=params[-1].detach().clone();opt.step();change=float((params[-1].detach()-probe).abs().max());assert change>0
                record=dict(step=step,ce=ces,penalty=pens,weighted_penalty=weight*pens,grad_norm=norm,parameter_max_change=change,
                    depths=schedule,forwards_after_second_token=0)
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                atomic_json(out/'progress.json',dict(phase='backprop_training',completed_steps=step,target_steps=steps,**record))
                if step%100==0:
                    history.append(evaluate(model,data['development'],tok,numeric_ids,out,step,'development',p));atomic_json(out/'history.json',history)
            model.assert_frozen();atomic_json(out/'frozen_parameter_check.json',dict(passed=True,sha256=model._frozen_digest))
            torch.save(adapter_state(model),archive/'adapter.pt');sha=file_hash(archive/'adapter.pt')
            (archive/'adapter.pt').chmod(0o444);(out/'adapter.pt').symlink_to(archive/'adapter.pt')
            test=evaluate(model,data['test'],tok,numeric_ids,out,steps,'test',p) if cell else None
            complete=dict(completed=True,passed=True,completed_steps=steps,additional_updates=800,start_step=200,optimizer_restart=True,adaptive_test_selected=True,
                minimum_pair_accuracy=initial['minimum_pair_accuracy'],initial_competence_gate_passed=initial['gate_passed'],test=test,adapter_sha256=sha,
                frozen_parameters_verified=True,forwards_after_second_token=0)
            atomic_json(out/'complete.json',complete);atomic_json(archive/'manifest.json',dict(config=cfg,completion=complete))
            atomic_json(out/'progress.json',dict(phase='complete' if complete['passed'] else 'competence_gate_failed',completed_steps=steps,target_steps=steps))
    except Exception as exc:
        atomic_json(out/'failed.json',dict(error=repr(exc)));raise

if __name__=='__main__':main()
