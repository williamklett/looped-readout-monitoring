"""Unrestricted model loading and exact-checkpoint checks for both stages."""
from contextlib import contextmanager
import os
import tempfile

import torch
from transformers import AutoTokenizer
from .preflight import OuroLoRAPolicy, ModelSpec, DEFAULT_ACTOR, resolve, greedy, encode, prompt
from .storage import save
from .integrity import check_runtime
from experiments.loop_replication.lora import load_adapter_state, adapter_state


@contextmanager
def policy(checkpoint, seed):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix='chess-comparison-') as cache:
        actor = resolve(DEFAULT_ACTOR,cache)
        tok = AutoTokenizer.from_pretrained(actor,local_files_only=True)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        model = OuroLoRAPolicy.load(actor,ModelSpec(architecture='tied',freeze_last=True,
            scope='all',loops=4,checkpointing=True),rank=16,alpha=32,dtype=torch.float32)
        load_adapter_state(model,torch.load(checkpoint,map_location='cpu',weights_only=True))
        model.assert_frozen()
        yield model, lambda fen,depth:greedy(model,tok,fen,depth,'chat'), tok
        model.assert_frozen()


def integrity(model,tok,readout,fen):
    ids = torch.tensor([encode(tok,prompt(fen),'chat')],device=next(model.parameters()).device)
    return check_runtime(model,ids,lambda depth:readout(fen,depth))


def save_adapter(model,path):
    assert not path.exists()
    path.parent.mkdir(parents=True,exist_ok=False)
    assert all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad)
    model.assert_frozen()
    torch.save(adapter_state(model),path)
    path.chmod(0o444)
