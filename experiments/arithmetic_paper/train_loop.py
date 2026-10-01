"""Uniform random stopping versus fixed-depth arithmetic concealment training."""
import argparse
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from experiments.loop_monitoring.model import ModelSpec
from experiments.loop_monitoring.model_cache import resolve, DEFAULT_ACTOR
from experiments.loop_replication.ouro import OuroLoRAPolicy
from experiments.loop_replication.lora import adapter_state
from experiments.loop_replication.train import atomic_json, file_hash
from experiments.arithmetic_lens.train_numeric import batch, answer_loss, greedy_answers, save

CELLS=[('random-penalty',True,1.),('fixed4-penalty',False,1.),('random-control',True,0.)]


def depth_schedule(step,random_depth,seed=17):
    rng=random.Random(197000+step+10000*(seed-17))
    return [rng.randint(1,7) if random_depth else 4 for _ in range(8)]


def boundary_penalty(model,rows,tok,mask_ids):
    """Average numeric probability before exit; an exit at pass 1 has none."""
    if model.spec.loops==1:
        return next(model.parameters()).new_zeros(())
    ids,mask,_=batch(rows,tok.pad_token_id)
    _,states=model(ids,attention_mask=mask,last_only=True)
    return torch.stack([model.base.lm_head(h[:,0]).float().softmax(-1)[:,mask_ids].sum(-1).mean()
                        for h in states[:-1]]).mean()


@torch.no_grad()
def evaluate(model,rows,tok,mask_ids,step,out,phase,random_depth):
    saved_depth=model.spec.loops
    n=len(rows);counts=[0]*7;numeric=[0.]*7;target=[0.]*7;records=[]
    for start in range(0,n,4):
        atomic_json(out/'progress.json',dict(phase=phase,completed_steps=step,target_steps=200,
                    evaluated_questions=start,evaluation_questions=n))
        rs=rows[start:start+4];model.spec.loops=7
        ids,mask,_=batch(rs,tok.pad_token_id)
        _,states=model(ids,attention_mask=mask,last_only=True)
        readouts=[[] for _ in rs]
        for k,h in enumerate(states):
            probs=model.base.lm_head(h[:,0]).float().softmax(-1)
            masses=probs[:,mask_ids].sum(-1);numeric[k]+=float(masses.sum())
            tp=probs.gather(1,torch.tensor([r['target_ids'][0] for r in rs],device=probs.device)[:,None])[:,0]
            target[k]+=float(tp.sum())
            values,indices=probs.topk(8,dim=-1)
            for i in range(len(rs)):
                readouts[i].append(dict(pass_number=k+1,numeric_mass=float(masses[i]),
                    first_answer_token_probability=float(tp[i]),
                    top_tokens=[dict(id=int(t),text=tok.decode([int(t)]),probability=float(v)) for t,v in zip(indices[i],values[i])]))
        outputs=[{} for _ in rs]
        for k in range(1,8):
            model.spec.loops=k
            answers=greedy_answers(model,rs,tok)
            for i,((text,terminated),r) in enumerate(zip(answers,rs)):
                correct=terminated and text==r['answer'];counts[k-1]+=correct
                outputs[i][str(k)]=dict(prediction=text,terminated=terminated,correct=correct)
        for i,r in enumerate(rs):
            records.append(dict(a=r['a'],b=r['b'],answer=r['answer'],boundaries=readouts[i],outputs=outputs[i]))
    model.spec.loops=saved_depth
    numeric=[v/n for v in numeric]
    acc=[v/n for v in counts]
    uniform_penalty=sum(sum(numeric[:k-1])/(k-1) for k in range(2,8))/7
    top1_numeric=[sum(r['boundaries'][k]['top_tokens'][0]['id'] in mask_ids for r in records)/n for k in range(7)]
    result=dict(top1_numeric_visibility=sum(top1_numeric[:3])/3,step=step,samples=n,accuracy=sum(acc)/7,accuracy_4=acc[3],
        numeric_mass=sum(numeric[:3])/3,numeric_mass_1_to_6=sum(numeric[:6])/6,
        expected_numeric_penalty=uniform_penalty if random_depth else sum(numeric[:3])/3,
        uniform_numeric_penalty=uniform_penalty,
        by_depth=[dict(top1_numeric_visibility=top1_numeric[k],depth=k+1,accuracy=acc[k],correct=counts[k],numeric_mass=numeric[k],
                      first_answer_token_probability=target[k]/n) for k in range(7)])
    atomic_json(out/(f'eval-{step:04d}.json' if phase=='development_evaluation' else 'final-test.json'),dict(metrics=result,readouts=records))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--cell',type=int,choices=range(3),required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument("--weight",type=float);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--name");p.add_argument("--warmstart",type=Path)
    args=p.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    name,random_depth,weight=CELLS[args.cell]
    weight=args.weight if args.weight is not None else weight
    name=args.name or name
    assert weight>=0
    out=args.root/name;out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(8);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        source=args.project/'results/arithmetic-allnumeric-v1/tied-penalty'
        data=json.loads((source/'dataset.json').read_text())
        library=json.loads((source/'numeric-vocabulary.json').read_text());mask_ids=library['token_ids']
        assert len(mask_ids)==893
        atomic_json(out/'numeric-vocabulary.json',library)
        atomic_json(out/'dataset.json',data)
        schedules={str(s):depth_schedule(s,random_depth,args.seed) for s in range(1,201)}
        atomic_json(out/'depth_schedule.json',schedules)
        common=args.warmstart or args.project/'results/arithmetic-lens-v1/warmstart/adapter.pt'
        ready=json.loads((common.parent/'complete.json').read_text())
        assert ready['passed'] and ready['adapter_sha256']==file_hash(common)
        cfg=dict(experiment='arithmetic-paper-v1',architecture='tied',freeze_last=True,decoder_frozen=True,
            depth_mode='uniform_1_to_7' if random_depth else 'fixed4',random_depth=random_depth,
            depth_sampling='independent uniform draw per four-example microbatch; no depth or stopping signal to model',
            penalty_weight=weight,steps=200,batch=32,microbatch=4,learning_rate=5e-5,rank=16,alpha=32,
            seed=args.seed,dtype='float32',initializer_sha256=file_hash(common),optimizer_restored=False,
            answer_loss='mean per-example full answer plus EOS CE at sampled exit',
            penalty='mean probability on same893numeric IDs across boundaries before sampled exit; zero when depth1',
            evaluation='128 fixed development questions at all7depths every20updates; accuracy=uniform mean of exact terminated accuracy; numeric_mass=mean boundaries1..3 for all arms',
            dataset_sha256=file_hash(source/'dataset.json'),mask_sha256=file_hash(source/'numeric-vocabulary.json'),
            average_depth=sum(sum(v) for v in schedules.values())/1600,
            limitations='Each cell is one seed; numeric readability is a proxy, not general honesty; last prompt position only. Full-number decoding uses the same exit depth for all output tokens.')
        atomic_json(out/'config.json',cfg)
        atomic_json(out/'progress.json',dict(phase='model_initialization',completed_steps=0,target_steps=200))
        with tempfile.TemporaryDirectory(prefix='arithmetic-random-depth-') as cache:
            actor=resolve(DEFAULT_ACTOR,cache)
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(actor,local_files_only=True)
            if tok.pad_token_id is None:tok.pad_token=tok.eos_token
            model=OuroLoRAPolicy.load(actor,ModelSpec(architecture='tied',freeze_last=True,scope='all',loops=4,checkpointing=True),rank=16,alpha=32,dtype=torch.float32)
            model.initialize_shared(torch.load(common,map_location='cpu',weights_only=True))
            atomic_json(out/'parameters.json',model.parameter_report())
            params=[p for p in model.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=0.)
            # Verify seven-pass gradients and memory before committing training.
            rs=data['train'][:4];model.spec.loops=7
            pen=boundary_penalty(model,rs,tok,mask_ids);pen.backward()
            penalty_norm=float(torch.nn.utils.clip_grad_norm_(params,1e9,error_if_nonfinite=True))
            assert penalty_norm>0 and all(p.grad is None for p in model.base.lm_head.parameters())
            opt.zero_grad(set_to_none=True)
            ce,_=answer_loss(model,rs,tok);pen=boundary_penalty(model,rs,tok,mask_ids)
            (ce+weight*pen).backward()
            total_norm=float(torch.nn.utils.clip_grad_norm_(params,1e9,error_if_nonfinite=True))
            assert total_norm>0
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                ids,mask,_=batch(rs,tok.pad_token_id)
                model.spec.loops=4;l4,s4=model(ids,attention_mask=mask,last_only=True)
                model.spec.loops=7;_,s7=model(ids,attention_mask=mask,last_only=True)
                for a,b in zip(s4,s7[:4]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(l4,model.base.lm_head(s7[3]).float(),rtol=0,atol=0)
                model.spec.loops=1;assert boundary_penalty(model,rs,tok,mask_ids).item()==0
            atomic_json(out/'gradient_check.json',dict(passed=True,penalty_grad_norm=penalty_norm,
                full_loss_grad_norm=total_norm,depth7_memory_peak_bytes=torch.cuda.max_memory_reserved(),
                depth1_penalty_zero=True,exact_prefix_parity_4_vs_7=True,decoder_frozen=True))
            history=[]
            def check(step):
                metrics=evaluate(model,data['development'],tok,mask_ids,step,out,'development_evaluation',random_depth)
                history.append(metrics);atomic_json(out/'history.json',history);print(json.dumps(metrics),flush=True)
                return metrics
            initial=check(0)
            assert initial['accuracy_4']>=.95,'Common four-pass arithmetic initializer lost accuracy'
            depth_counts=[0]*7
            for step in range(1,201):
                atomic_json(out/'progress.json',dict(phase='backprop_training',completed_steps=step-1,target_steps=200,depth_counts=depth_counts))
                rs=random.Random(87000+step+10000*(args.seed-17)).sample(data['train'],32)
                opt.zero_grad(set_to_none=True);ce_sum=pen_sum=0.
                for micro,start in enumerate(range(0,32,4)):
                    k=schedules[str(step)][micro];model.spec.loops=k;depth_counts[k-1]+=4
                    sub=rs[start:start+4]
                    ce,_=answer_loss(model,sub,tok)
                    pen=boundary_penalty(model,sub,tok,mask_ids) if weight else ce.new_zeros(())
                    ((ce+weight*pen)/8).backward()
                    ce_sum+=float(ce.detach())/8;pen_sum+=float(pen.detach())/8
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step()
                record=dict(step=step,ce=ce_sum,penalty=pen_sum,grad_norm=norm,depths=schedules[str(step)])
                with (out/'training.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
                atomic_json(out/'progress.json',dict(phase='backprop_training',completed_steps=step,target_steps=200,depth_counts=depth_counts,**{k:v for k,v in record.items() if k!='step'}))
                if step%20==0:
                    save(out/'resume.pt',model,opt,step);check(step)
            model.assert_frozen();atomic_json(out/'frozen_parameter_check.json',dict(passed=True))
            torch.save(adapter_state(model),out/'adapter.pt')
            final=evaluate(model,data['test'],tok,mask_ids,200,out,'final_evaluation',random_depth)
            atomic_json(out/'complete.json',dict(completed=True,completed_steps=200,test=final,depth_counts=depth_counts))
            (out/'resume.pt').unlink(missing_ok=True)
            atomic_json(out/'progress.json',dict(phase='complete',completed_steps=200,target_steps=200,depth_counts=depth_counts))
    except Exception as e:
        atomic_json(out/'failed.json',dict(error=repr(e)));raise


if __name__=='__main__':main()
