"""Pure REINFORCE, fixed action-independent baseline, unrestricted native outputs."""
import copy
import torch
from .interface import prompt
from experiments.chess_comparison_v3.preflight import encode


def backward_episode(model,tok,episode):
    advantages=[];baselines=[]
    device=next(model.parameters()).device;total=0.;worst=0.
    assert len(episode['events'])==len(episode['proposal_readouts'])
    for event,r in zip(episode['events'],episode['proposal_readouts']):
        baseline=event['learning_baseline'];advantage=event['learning_reward']-baseline
        advantages.append(advantage);baselines.append(baseline)
        assert event['action']==r['action'] and event['depth']==r['depth'] and r['temperature']==1.
        prefix=encode(tok,prompt(event['before_fen']),'chat');chosen=r['token_ids']
        ids=torch.tensor([prefix+chosen],device=device);model.spec.loops=event['depth']
        logits,_=model(ids[:,:-1],return_states=False)
        logp=logits[0,len(prefix)-1:,:].float().log_softmax(-1)
        selected=logp.gather(1,ids[0,len(prefix):,None]).squeeze(1)
        recorded=torch.tensor(r['token_logprobs'],device=device)
        worst=max(worst,float((selected.detach()-recorded).abs().max()))
        torch.testing.assert_close(selected.detach(),recorded,rtol=1e-5,atol=1e-4)
        loss=-advantage*selected.sum();assert torch.isfinite(loss)
        # A zero-advantage episode has exactly zero policy gradient. Retain
        # the on-policy likelihood checks, but avoid a needless backward pass.
        if advantage!=0:loss.backward()
        total+=float(loss.detach())
    return dict(loss=total,advantages=advantages,baselines=baselines,max_logprob_recompute_error=worst)


def update(model,tok,optimizer,episode,clip_norm=10000.):
    params=[p for p in model.parameters() if p.requires_grad]
    before=[p.detach().clone() for p in params]
    optimizer.zero_grad(set_to_none=True);result=backward_episode(model,tok,episode)
    norm=torch.nn.utils.clip_grad_norm_(params,clip_norm,error_if_nonfinite=True)
    stepped=any(a!=0 for a in result['advantages'])
    if stepped:
        assert float(norm)>0
        optimizer.step()
    change=max(float((p.detach()-b).abs().max()) for p,b in zip(params,before))
    assert all(torch.isfinite(p).all() for p in params)
    if stepped:assert change>0
    else:assert change==0
    return dict(**result,gradient_norm=float(norm),clip_norm=clip_norm,
        gradient_clip_applied=float(norm)>clip_norm,
        max_parameter_change=change,optimizer_step_applied=stepped)


def disposable(model,tok,episode,lr):
    params=[p for p in model.parameters() if p.requires_grad];before=[p.detach().clone() for p in params]
    rng=torch.random.get_rng_state();cuda=torch.cuda.get_rng_state_all();loops=model.spec.loops
    try:
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.)
        synthetic=copy.deepcopy(episode)
        for e in synthetic['events']:e['learning_reward']=e['learning_baseline']
        synthetic['events'][-1]['learning_reward']+=1.
        result=update(model,tok,opt,synthetic)
    finally:
        with torch.no_grad():
            for p,b in zip(params,before):p.copy_(b);p.grad=None
        torch.random.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda);model.spec.loops=loops
    for p,b in zip(params,before):torch.testing.assert_close(p,b,rtol=0,atol=0)
    model.assert_frozen()
    return dict(**result,disposable=True,synthetic_last_action_advantage=1.,exact_restoration=True,frozen_verified=True)
