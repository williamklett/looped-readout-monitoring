"""Disposable checks on the exact reward-run model and search implementation."""
import torch
from experiments.chess_comparison_v3.parameter_search import search_step


@torch.no_grad()
def check_runtime(model, prompt_ids, readout):
    """readout(depth) returns a complete native greedy command with token IDs.

    Constant scores isolate restoration and are never reported as reward
    optimization. Restore adapter, loop count and RNG even when a check fails.
    """
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial={n:p.detach().clone() for n,p in params.items()}
    loops=model.spec.loops
    rng=torch.random.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    def assert_restored():
        for name,p in params.items():
            torch.testing.assert_close(p,initial[name],rtol=0,atol=0)
        model.assert_frozen()
    try:
        model.assert_frozen()
        model.spec.loops=1
        logits,states=model(prompt_ids,last_only=True)
        model.spec.loops=4
        _,long_states=model(prompt_ids,last_only=True)
        torch.testing.assert_close(states[0],long_states[0],rtol=0,atol=0)
        torch.testing.assert_close(logits,model.base.lm_head(long_states[0]).float(),rtol=0,atol=0)
        before={k:readout(k) for k in (1,4)}
        candidates=[]
        def constant():
            candidates.append(readout(1))
            return 0.
        result=search_step(model,constant,seed=17,scale=.001,directions=1)
        assert result['accepted'] is None and result['max_parameter_change']==0
        assert_restored()
        calls=0
        def interrupted():
            nonlocal calls
            calls+=1
            if calls==2:
                raise InterruptedError('Disposable restoration check')
            return 0.
        try:
            search_step(model,interrupted,seed=29,scale=.001,directions=1)
        except InterruptedError:
            pass
        else:
            raise AssertionError('Expected interruption was not raised')
        assert_restored()
        after={k:readout(k) for k in (1,4)}
        assert before==after, 'Restored greedy commands differ'
        model.spec.loops=1
        after_logits,_=model(prompt_ids,last_only=True)
        torch.testing.assert_close(logits,after_logits,rtol=0,atol=0)
        return dict(exact_prefix_parity=True,exact_tie_restoration=True,
            exact_exception_restoration=True,exact_greedy_and_logit_restoration=True,
            frozen_verified=True,frozen_sha256=model._frozen_digest,
            actual_reward_updates=0,baseline=before,
            disposable_readouts=candidates,search_result=result)
    finally:
        for name,p in params.items():
            p.copy_(initial[name])
        model.spec.loops=loops
        torch.random.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
