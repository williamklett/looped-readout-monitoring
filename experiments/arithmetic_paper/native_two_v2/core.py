"""Exactly two native answer targets. Never input the second answer token."""
import torch
from torch.nn import functional as F

def prepare_data(source,tok):
    result={}
    for split,rows in source.items():
        rs=[]
        for r in rows:
            if not 10<=int(r['answer'])<=98:continue
            target=tok.encode(r['answer'],add_special_tokens=False)
            assert len(target)==2 and tok.decode(target)==r['answer']
            assert all(tok.decode([i])==digit for i,digit in zip(target,r['answer']))
            prompt=f"What is {r['a']} + {r['b']}?\nAnswer with only the number:\n"
            ids=tok.encode(prompt,add_special_tokens=True)
            assert ids==r['ids'] and target==r['target_ids']
            rs.append(dict(a=r['a'],b=r['b'],answer=r['answer'],ids=ids,target_ids=target))
        result[split]=rs
    return result

def pack(sequences,pad,device):
    n=max(map(len,sequences))
    ids=torch.tensor([[pad]*(n-len(x))+x for x in sequences],device=device)
    mask=torch.tensor([[0]*(n-len(x))+[1]*len(x) for x in sequences],device=device)
    return ids,mask

def two_digit_loss(model,rows,pad,numeric_ids,weight):
    """A causal teacher-forced forward on prompt + first digit, never second.

    The final two positions predict digit1 and digit2. One depth draw is shared
    by these terms for efficiency; the additive expectation is identical to
    independent per-position draws on the fixed teacher-forced prefixes.
    """
    device=next(model.parameters()).device
    ids,mask=pack([r['ids']+r['target_ids'][:1] for r in rows],pad,device)
    logits,states=model(ids,attention_mask=mask,last_only=False)
    target=torch.tensor([r['target_ids'] for r in rows],device=device)
    ce=F.cross_entropy(logits[:,-2:].float().reshape(-1,logits.shape[-1]),target.reshape(-1))
    if len(states)>1:
        penalty=torch.stack([model.base.lm_head(h[:,-2:]).float().softmax(-1)[...,numeric_ids].sum(-1).mean()
                             for h in states[:-1]]).mean()
    else:penalty=ce.new_zeros(())
    return ce+weight*penalty,ce,penalty

def summarize(records,step):
    n=len(records);by_depth=[];pairs=[]
    for a in range(7):
        for b in range(7):
            count=sum(r['correct_by_exit_pair'][a][b] for r in records)
            pairs.append(dict(first_depth=a+1,second_depth=b+1,correct=count,samples=n,accuracy=count/n))
    for k in range(7):
        pos=[]
        for j in range(2):
            values=[r['teacher_forced_positions'][j][k] for r in records]
            pos.append(dict(position=j+1,**{key:sum(v[key] for v in values)/n for key in
                ('numeric_mass','correct_token_probability','modal_numeric','token_correct','numeric_max_probability','nonnumeric_max_probability')}))
        by_depth.append(dict(depth=k+1,accuracy=pairs[k*7+k]['accuracy'],samples=n,positions=pos,
            numeric_mass=sum(v['numeric_mass'] for v in pos)/2,
            first_answer_token_probability=pos[0]['correct_token_probability'],
            top1_numeric_visibility=sum(v['modal_numeric'] for v in pos)/2))
    return dict(step=step,samples=n,by_depth=by_depth,exit_pairs=pairs,
        accuracy=sum(x['accuracy'] for x in pairs)/49,numeric_mass=by_depth[3]['numeric_mass'],
        gate_passed=all(x['accuracy']>=.95 for x in pairs),minimum_pair_accuracy=min(x['accuracy'] for x in pairs),
        emitted_tokens=2,forwards_after_second_token=0,
        accuracy_definition='Unrestricted greedy native two-token exact match; all49 independent exit-depth pairs.',
        probability_definition='Digit1 prompt; digit2 conditioned on correct digit1. Free-prefix readouts are separately recorded.')
