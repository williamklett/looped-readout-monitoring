"""Full native vocabulary, balanced syntax examples, no output masking."""
import torch
from experiments.chess_comparison_v3.preflight import encode
from experiments.chess_exploration_diagnostic_v1.prompts import prompt as diagnostic_prompt


def prompt(fen):
    return diagnostic_prompt(fen,'piece_inventory_examples')


@torch.no_grad()
def command(model,tok,fen,depth,temperature=1.,greedy=False):
    assert temperature==1.
    model.spec.loops=depth
    ids=encode(tok,prompt(fen),'chat');tokens=[];logprobs=[];entropies=[]
    device=next(model.parameters()).device
    for _ in range(12):
        logits,_=model(torch.tensor([ids+tokens],device=device),last_only=True,return_states=False)
        logp=logits[0,-1].float().log_softmax(-1)
        assert torch.isfinite(logp).all()
        token=int(logp.argmax()) if greedy else int(torch.multinomial(logp.exp(),1))
        tokens.append(token);logprobs.append(float(logp[token]))
        entropies.append(float(-(logp.exp()*logp).sum()))
        if token==tok.eos_token_id or '\n' in tok.decode(tokens,skip_special_tokens=False):break
    raw=tok.decode(tokens,skip_special_tokens=True)
    return dict(action=raw.strip(),raw=raw,token_ids=tokens,token_logprobs=logprobs,
        token_entropies=entropies,depth=depth,temperature=temperature,greedy=greedy,
        output_mask=False,max_tokens=12,prompt_variant='piece_inventory_examples')
