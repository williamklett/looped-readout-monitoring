"""Direct-gradient answer accuracy versus semantic boundary visibility."""
import argparse,json,os,random,tempfile,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.loop_monitoring.model import ModelSpec
from experiments.loop_monitoring.model_cache import resolve,DEFAULT_ACTOR
from experiments.loop_replication.ouro import OuroLoRAPolicy
from experiments.loop_replication.lora import adapter_state
from experiments.loop_replication.train import atomic_json,file_hash
from .equivalents import words, tokenize, manifest as library_manifest
from .numeric_vocabulary import build as numeric_library
from experiments.loop_replication.lora import load_adapter_state

CELLS=[('tied',True,1.),('untied',False,1.),('tied',True,0.),('untied',False,0.)]

def dataset(tok):
    rows=[]
    for a in range(50):
        for b in range(a,50):
            answer=str(a+b);target=tok.encode(answer,add_special_tokens=False)
            prompt=f'What is {a} + {b}?\nAnswer with only the number:\n'
            ids=tok.encode(prompt,add_special_tokens=True)
            assert target and tok.decode(target).strip()==answer
            sequences=tokenize(a+b,tok)
            aliases=[v[0] for v in sequences if len(v)==1]
            rows.append(dict(a=a,b=b,answer=answer,target=target[0],target_ids=target,ids=ids,aliases=aliases,multi_aliases=[v for v in sequences if len(v)>1]))
    assert len(rows)==1275 and len({tuple(r['target_ids']) for r in rows})==99
    rng=random.Random(12043);rng.shuffle(rows)
    # Unordered operand pairs keep commuted duplicates in the same partition.
    return dict(train=rows[384:],development=rows[:128],test=rows[128:384])

def batch(rows,pad):
    n=max(len(r['ids']) for r in rows)
    ids=torch.tensor([[pad]*(n-len(r['ids']))+r['ids'] for r in rows],device='cuda')
    mask=torch.tensor([[0]*(n-len(r['ids']))+[1]*len(r['ids']) for r in rows],device='cuda')
    labels=torch.tensor([r['target'] for r in rows],device='cuda')
    return ids,mask,labels

def single_mass(logits,rows):
    probs=logits.softmax(-1)
    return torch.stack([probs[i,r['aliases']].sum() for i,r in enumerate(rows)]).mean()

def sequence_mass(model,items,pad):
    """Joint probability of each complete alias at each recurrent boundary."""
    forced=[dict(ids=r['ids']+seq[:-1],target=r['target']) for r,seq in items]
    ids,mask,_=batch(forced,pad)
    _,states=model(ids,attention_mask=mask,last_only=False)
    values=[]
    for h in states[:-1]:
        per=[]
        for i,(_,seq) in enumerate(items):
            logits=model.base.lm_head(h[i,-len(seq):]).float()
            labels=torch.tensor(seq,device=logits.device)
            logps=logits.log_softmax(-1).gather(1,labels[:,None])[:,0]
            per.append(logps.sum().exp())
        values.append(torch.stack(per))
    return torch.stack(values).mean(0)

def multi_mass(model,rows,pad,rng=None):
    items=[];weights=[]
    for r in rows:
        seqs=r['multi_aliases']
        if not seqs:continue
        if rng is not None:
            items.append((r,rng.choice(seqs)));weights.append(len(seqs))
        else:
            items.extend((r,v) for v in seqs);weights.extend([1]*len(seqs))
    if not items:return next(model.parameters()).new_zeros(())
    result=next(model.parameters()).new_zeros(())
    for start in range(0,len(items),4):
        values=sequence_mass(model,items[start:start+4],pad)
        result=result+(values*values.new_tensor(weights[start:start+4])).sum()/len(rows)
    return result

def forward(model,rows,pad):
    ids,mask,labels=batch(rows,pad)
    logits,states=model(ids,attention_mask=mask,last_only=True)
    boundary=[model.base.lm_head(h[:,0]).float() for h in states[:-1]]
    assert len(boundary)==3
    return logits[:,0],boundary,labels

def answer_loss(model,rows,tok):
    labels=[r['target_ids']+[tok.eos_token_id] for r in rows]
    forced=[dict(ids=r['ids']+y[:-1],target=r['target']) for r,y in zip(rows,labels)]
    ids,mask,_=batch(forced,tok.pad_token_id)
    logits,_=model(ids,attention_mask=mask,last_only=False,return_states=False)
    losses=[];probabilities=[]
    for i,y in enumerate(labels):
        yy=torch.tensor(y,device=logits.device)
        lp=logits[i,-len(y):].log_softmax(-1).gather(1,yy[:,None])[:,0]
        losses.append(-lp.mean());probabilities.append(lp.sum().exp())
    return torch.stack(losses).mean(),torch.stack(probabilities)

@torch.no_grad()
def greedy_answers(model,rows,tok):
    work=[dict(r,ids=list(r['ids'])) for r in rows];done=[False]*len(rows);sequences=[[] for _ in rows]
    for _ in range(8):
        ids,mask,_=batch(work,tok.pad_token_id)
        logits,_=model(ids,attention_mask=mask,last_only=True,return_states=False)
        predicted=logits[:,0].argmax(-1).tolist()
        for i,v in enumerate(predicted):
            if not done[i]:
                if v==tok.eos_token_id:done[i]=True
                else:sequences[i].append(v)
            work[i]['ids'].append(v)
        if all(done):break
    return [(tok.decode(ids,skip_special_tokens=True).strip(),stop) for ids,stop in zip(sequences,done)]

@torch.no_grad()
def evaluate(model,rows,tok,judge,step,path):
    sums=dict(correct=0,answer_probability=0.,lens_first_token_probability=0.,numeric_mass=0.,lens_top1_first_answer_token=0.)
    boundary_p=[0.,0.,0.];readouts=[]
    for start in range(0,len(rows),4):
        rs=rows[start:start+4];logits,ls,y=forward(model,rs,tok.pad_token_id)
        answers=greedy_answers(model,rs,tok)
        sums['correct']+=sum(stop and text==r['answer'] for (text,stop),r in zip(answers,rs))
        _,answer_probs=answer_loss(model,rs,tok)
        sums['answer_probability']+=float(answer_probs.sum())
        for k,l in enumerate(ls):
            p=l.softmax(-1);v=float(p.gather(1,y[:,None]).sum());boundary_p[k]+=v
            sums['lens_first_token_probability']+=v/3
            sums['lens_top1_first_answer_token']+=float((l.argmax(-1)==y).sum())/3
            if judge:sums['numeric_mass']+=float(single_mass(l,rs))*len(rs)/3
        if judge:sums['numeric_mass']+=float(multi_mass(model,rs,tok.pad_token_id))*len(rs)
        if start<16:
            for i,r in enumerate(rs):
                readouts.append(dict(a=r['a'],b=r['b'],answer=r['answer'],prediction=answers[i][0],terminated=answers[i][1],boundaries=[[(tok.decode([int(v)]),float(p)) for p,v in zip(*l[i].softmax(-1).topk(8))] for l in ls]))
    result=dict(step=step,samples=len(rows),accuracy=sums.pop('correct')/len(rows),**{k:v/len(rows) for k,v in sums.items()},boundary_answer_probability=[v/len(rows) for v in boundary_p])
    if path:atomic_json(path,dict(metrics=result,readouts=readouts))
    return result

def save(path,model,opt,step):
    tmp=path.with_suffix('.tmp');torch.save(dict(adapter=adapter_state(model),optimizer=opt.state_dict(),step=step),tmp);os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['warmstart','train'],required=True)
    p.add_argument('--previous-root',type=Path,required=True);p.add_argument('--cell',type=int,default=0);p.add_argument('--root',type=Path,required=True);p.add_argument('--release',type=Path,required=True)
    a=p.parse_args();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    root=a.root;root.mkdir(parents=True,exist_ok=True)
    arch,frozen,weight=CELLS[a.cell] if a.stage=='train' else ('tied',True,0.)
    name=f'{arch}-'+('penalty' if weight else 'control');out=root/(name if a.stage=='train' else 'warmstart');out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(8);torch.manual_seed(17);np.random.seed(17);random.seed(17)
        torch.backends.cuda.matmul.allow_tf32=False
        atomic_json(out/'equivalence-library.json',library_manifest())
        with tempfile.TemporaryDirectory(prefix='arithmetic-lens-') as cache:
            actor=resolve(DEFAULT_ACTOR,cache)
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(actor,local_files_only=True)
            if tok.pad_token_id is None:tok.pad_token=tok.eos_token
            data=dataset(tok)
            library=numeric_library(tok);atomic_json(out/'numeric-vocabulary.json',library)
            for rows in data.values():
                for row in rows:row['aliases']=library['token_ids'];row['multi_aliases']=[]
            atomic_json(out/'dataset.json',data)
            common_root=a.previous_root

            model=OuroLoRAPolicy.load(actor,ModelSpec(architecture=arch,freeze_last=frozen,scope='all',loops=4,checkpointing=True),rank=16,alpha=32,dtype=torch.float32)
            judge=None
            if a.stage=='train':
                common=common_root/'warmstart/adapter.pt';ready=json.loads((common_root/'warmstart/complete.json').read_text())
                assert ready['passed'] and file_hash(common)==ready['adapter_sha256']
                model.initialize_shared(torch.load(common,map_location='cpu',weights_only=True))
                judge=True
            cfg=dict(gpu_device=torch.cuda.get_device_name(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,stage=a.stage,architecture=arch,freeze_last=frozen,decoder_frozen=True,answer_loss='mean per-example full answer plus EOS teacher-forced CE; greedy exact match including termination at eval',rank=16,alpha=32,seed=17,penalty_weight=weight,steps=200,batch=32,microbatch=4,learning_rate=5e-5,penalty='Mean probability mass on all numeric token IDs at each of3boundaries, irrespective of answer; no sequence-product or alias sampling',numeric_vocabulary_sha256=file_hash(out/'numeric-vocabulary.json'),dataset_sha256=file_hash(out/'dataset.json'),initializer_sha256=file_hash(common_root/'warmstart/adapter.pt') if a.stage=='train' else None,limitations='Two main arms confound tying and last-block freezing; untied has more trainable parameters. Global numeric lexicon includes context-ambiguous number-word fragments and symbols; arbitrary codes remain possible. Readout evasion is not proof of global basis rotation.')
            atomic_json(out/'config.json',cfg);atomic_json(out/'parameters.json',model.parameter_report())
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=0.)
            start_step=0
            if weight:
                prior=a.previous_root/name
                checkpoint=prior/'resume.pt'
                old_data=json.loads((prior/'dataset.json').read_text())
                for split in data:
                    identity=lambda r:(r['a'],r['b'],r['ids'],r['target_ids'])
                    assert [identity(r) for r in data[split]]==[identity(r) for r in old_data[split]],'Changed problem stream'
                restored=torch.load(checkpoint,map_location='cpu',weights_only=True)
                load_adapter_state(model,restored['adapter']);opt.load_state_dict(restored['optimizer'])
                start_step=restored['step']
                assert 0<start_step<200
                atomic_json(out/'continuation.json',dict(checkpoint=str(checkpoint),sha256=file_hash(checkpoint),
                    start_step=start_step,optimizer_restored=True,weights_restored=True,
                    objective_change='correct-answer equivalents -> all numeric tokens',
                    rng='deterministic per-step Python Random(87000+step); eval mode, no dropout or sampling in new mask'))
                del restored
            history=[]
            def check(step):
                atomic_json(out/'progress.json',dict(phase='development_evaluation',completed_steps=step,target_steps=200))
                result=evaluate(model,data['development'],tok,True,step,out/f'eval-{step:04d}.json')
                result['numeric_samples']=128
                history.append(result);atomic_json(out/'history.json',history);print(json.dumps(result),flush=True)
                return result
            baseline=check(start_step)
            if a.stage=='train':
                assert baseline['accuracy']>=.95,'Common initializer lost arithmetic accuracy'
                # Verify differentiable penalty reaches actor while judge/head stay frozen.
                rs=data['train'][:4];_,ls,_=forward(model,rs,tok.pad_token_id)
                loss=sum(single_mass(l,rs) for l in ls)/3+multi_mass(model,rs,tok.pad_token_id,random.Random(31))
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1e9,error_if_nonfinite=True))
                assert norm>0 and all(p.grad is None for p in model.base.lm_head.parameters())
                atomic_json(out/'gradient_check.json',dict(passed=True,penalty_grad_norm=norm,all_numeric_vocabulary=True,decoder_frozen=True));opt.zero_grad(set_to_none=True)
            ready=a.stage=='warmstart' and baseline['accuracy']>=.95
            for step in range(start_step+1,201):
                if ready:break
                atomic_json(out/'progress.json',dict(phase='backprop_training',completed_steps=step-1,target_steps=200))
                rng=random.Random(87000+step);rs=rng.sample(data['train'],32);opt.zero_grad(set_to_none=True)
                ce_sum=pen_sum=0.
                for start in range(0,32,4):
                    sub=rs[start:start+4];logits,ls,y=forward(model,sub,tok.pad_token_id)
                    ce,_=answer_loss(model,sub,tok);pen=(sum(single_mass(l,sub) for l in ls)/3+multi_mass(model,sub,tok.pad_token_id,rng)) if weight else ce*0
                    ((ce+weight*pen)/8).backward();ce_sum+=float(ce.detach())/8;pen_sum+=float(pen.detach())/8
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True));opt.step()
                atomic_json(out/'progress.json',dict(phase='backprop_training',completed_steps=step,target_steps=200,ce=ce_sum,penalty=pen_sum,grad_norm=norm))
                if step%20==0:
                    result=check(step);save(out/'resume.pt',model,opt,step)
                    ready=a.stage=='warmstart' and result['accuracy']>=.95
            model.assert_frozen();atomic_json(out/'frozen_parameter_check.json',dict(passed=True))
            if a.stage=='warmstart':
                if not ready:raise RuntimeError('Arithmetic warmstart did not reach95%development accuracy; comparison blocked')
                torch.save(adapter_state(model),out/'adapter.pt')
                atomic_json(out/'complete.json',dict(completed=True,passed=True,adapter_sha256=file_hash(out/'adapter.pt'),baseline=history[-1]))
            else:
                test=evaluate(model,data['test'],tok,judge,200,out/'final-test.json')
                torch.save(adapter_state(model),out/'adapter.pt')
                atomic_json(out/'complete.json',dict(completed=True,completed_steps=200,test=test))
    except Exception as e:
        atomic_json(out/'failed.json',dict(error=repr(e)));raise
if __name__=='__main__':main()
