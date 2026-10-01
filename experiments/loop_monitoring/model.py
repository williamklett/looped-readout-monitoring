"""Auditable Ouro unrolling with an explicitly final-loop policy distribution.

Frozen blocks can share storage in the untied condition: since they never update,
this is mathematically identical to separate equal frozen copies. All trainable
blocks have independent copies per loop in the untied condition.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .stateio import load_state,save_delta


@dataclass
class ModelSpec:
    architecture: str = 'tied'
    freeze_last: bool = True
    scope: str = 'edges'
    loops: int = 4
    checkpointing: bool = True


class LoopPolicy(nn.Module):
    def __init__(self,base,spec):
        super().__init__()
        self.base=base
        self.spec=spec
        self.config=base.config
        self.base.requires_grad_(False)
        depth=len(base.model.layers)
        if spec.scope=='all':
            selected=list(range(depth))
        elif spec.scope=='matched':
            # Same number of trainable layers: first+middle vs first+last.
            selected=[0,depth//2] if spec.freeze_last else [0,depth-1]
        else:
            selected=[0,depth-1]
        if spec.freeze_last:
            selected=[i for i in selected if i!=depth-1]
        self.selected=selected
        self.independent=nn.ModuleDict()
        if spec.architecture=='untied':
            for loop in range(spec.loops):
                self.independent[str(loop)]=nn.ModuleDict({str(i):copy.deepcopy(base.model.layers[i]) for i in selected})
            self.independent.requires_grad_(True)
        elif spec.architecture=='tied':
            for i in selected:
                base.model.layers[i].requires_grad_(True)
        else:
            raise ValueError(spec.architecture)
        # Embed, LM head, recurrent normalization and exit gate stay frozen in all arms.
        self.initial={n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
        self._frozen_digest=self.interface_digest()

    @classmethod
    def load(cls,path,spec,initial_checkpoint=None):
        from .vendor.ouro.modeling_ouro import OuroForCausalLM
        from .vendor.ouro.configuration_ouro import OuroConfig
        cfg=OuroConfig.from_pretrained(path)
        base=OuroForCausalLM.from_pretrained(path,config=cfg,torch_dtype=torch.bfloat16,
                                             attn_implementation='sdpa',local_files_only=True)
        if initial_checkpoint:
            apply_common_initialization(base,initial_checkpoint)
        return cls(base,spec).cuda()

    def layer(self,loop,i):
        if self.spec.architecture=='untied' and i in self.selected:
            return self.independent[str(loop)][str(i)]
        return self.base.model.layers[i]

    def forward(self,ids,attention_mask=None,cache=None,last_only=False,return_states=True):
        from transformers.masking_utils import create_causal_mask
        m=self.base.model
        h=m.embed_tokens(ids)
        past=cache.get_seq_length() if cache is not None else 0
        positions=torch.arange(past,past+ids.shape[1],device=ids.device)
        if attention_mask is None:
            position_ids=positions.unsqueeze(0)
        else:
            position_ids=attention_mask.long().cumsum(-1)-1
            position_ids=position_ids.clamp_min(0)[:,-ids.shape[1]:]
        mask=create_causal_mask(config=self.config,input_embeds=h,attention_mask=attention_mask,
                               cache_position=positions,past_key_values=cache,position_ids=position_ids)
        rope=m.rotary_emb(h,position_ids)
        states=[]
        for loop in range(self.spec.loops):
            for i in range(len(m.layers)):
                layer=self.layer(loop,i)
                kwargs=dict(attention_mask=mask,position_ids=position_ids,past_key_value=cache,
                            use_cache=cache is not None,cache_position=positions,
                            position_embeddings=rope,current_ut=loop)
                if self.spec.checkpointing and torch.is_grad_enabled():
                    h=checkpoint(layer,h,use_reentrant=False,**kwargs)
                else:
                    h=layer(h,**kwargs)
            h=m.norm(h)
            if return_states:
                states.append(h[:,-1:] if last_only else h)
        logits=self.base.lm_head(h[:,-1:] if last_only else h).float()
        return logits,states

    def logps(self,ids,prompt_len):
        logits,_=self(ids[:,:-1],return_states=False)
        # Position prompt_len-1 predicts the first completion token.
        logits=logits[:,prompt_len-1:]
        labels=ids[:,prompt_len:]
        return -torch.nn.functional.cross_entropy(logits.transpose(1,2),labels,reduction='none')

    @contextlib.contextmanager
    def reference(self):
        saved={}
        with torch.no_grad():
            for n,p in self.named_parameters():
                if n in self.initial:
                    saved[n]=p.detach().cpu().clone()
                    p.copy_(self.initial[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n,p in self.named_parameters():
                    if n in saved:
                        p.copy_(saved[n])

    def interface_digest(self):
        digest=hashlib.sha256()
        modules=[self.base.lm_head,self.base.model.norm]
        if self.spec.freeze_last:
            modules.append(self.base.model.layers[-1])
        for module in modules:
            for p in module.parameters():
                digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def assert_frozen(self):
        if self.interface_digest()!=self._frozen_digest:
            raise RuntimeError('Frozen interface changed!')

    def save_trainable(self,path):
        save_delta(path,{n:p for n,p in self.named_parameters() if p.requires_grad},self.initial)

    def load_trainable(self,path,reset_reference=True):
        state=load_state(path,self.initial)
        params=dict(self.named_parameters())
        if set(state)!=set(self.initial):
            raise ValueError('Checkpoint parameter names do not match this arm.')
        with torch.no_grad():
            for n,x in state.items(): params[n].copy_(x)
        if reset_reference:
            self.initial={n:params[n].detach().cpu().clone() for n in self.initial}

    def parameter_report(self):
        return {'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'stored':sum(p.numel() for p in self.parameters()),
                'trainable_layers':self.selected,'spec':vars(self.spec),
                'interface_sha256':self._frozen_digest}


def apply_common_initialization(base,path):
    """Load the same first-layer SFT weights BEFORE making untied copies.

    Only checkpoints from the explicit tied/frozen/edges seeding procedure are
    accepted. This cannot silently turn an arm-specific RL checkpoint into a
    common initializer or change the last layer.
    """
    initial={'base.model.layers.0.'+name:p.detach().cpu().clone()
             for name,p in base.model.layers[0].named_parameters()}
    state=load_state(path,initial)
    expected={'base.model.layers.0.'+name for name,_ in base.model.layers[0].named_parameters()}
    if set(state)!=expected:
        raise ValueError('Common initialization requires exactly the first transformer layer.')
    params=dict(base.named_parameters())
    with torch.no_grad():
        for name,value in state.items():params[name.removeprefix('base.')].copy_(value)


class MasterAdam:
    """FP32 optimizer state/master weights for BF16 model parameters."""
    def __init__(self,model,lr=1e-5):
        self.params=[p for p in model.parameters() if p.requires_grad]
        self.master=[nn.Parameter(p.detach().float().clone()) for p in self.params]
        self.opt=torch.optim.AdamW(self.master,lr=lr,weight_decay=0.0)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)
        for p in self.params: p.grad=None

    def step(self):
        for p,m in zip(self.params,self.master):
            m.grad=None if p.grad is None else p.grad.detach().float()
        norm=torch.nn.utils.clip_grad_norm_(self.master,1.0,error_if_nonfinite=True)
        self.opt.step()
        with torch.no_grad():
            for p,m in zip(self.params,self.master): p.copy_(m)
        return float(norm)


@torch.no_grad()
def parity_check(model,ids,atol=.04):
    model.eval()
    ours,states=model(ids)
    native=model.base(ids,use_cache=False,exit_at_step=model.spec.loops-1).logits.float()
    delta=(ours-native).abs().max().item()
    if delta>atol:
        raise RuntimeError(f'Native/custom policy mismatch: {delta}')
    return {'max_abs_logit_error':delta,'n_loop_states':len(states)}
