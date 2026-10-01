"""Conventional Atlas GRPO, native Transformers/PEFT runtime substitution.

Never launches jobs, downloads models, or executes code on import. See
TRAIN_PROTOCOL.md for pinned-source correspondence and deliberate departures.
Only the CLI requires a GPU allocation; tensor helpers support CPU unit tests.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault('USE_TF', '0')

import torch
from torch.nn import functional as F

from .data import SOURCE_REVISION, DATASET_REVISION, canonical_hash

SCHEMA = 'atlas-native-grpo-v2'
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 120
    scheduler_steps: int = 120
    prompts_per_update: int = 64  # Not64 completions: source sends64 prompts.
    generations: int = 8
    generation_microbatch: int = 4
    replay_microbatch: int = 1
    probe_microbatch: int = 1
    learning_rate: float = 5e-5
    warmup_steps: int = 5
    beta: float = .001
    alpha: float = 1.
    epsilon: float = .2
    max_grad_norm: float = 1.
    weight_decay: float = 0.
    seed: int = 1
    rank: int = 64
    lora_alpha: int = 128
    max_new_tokens: int = 256
    max_prompt_tokens: int = 256
    replay_logprob_tolerance: float = .1
    preflight: bool = False
    checkpoint_every: int = 10

    @property
    def rollouts_per_update(self):
        return self.prompts_per_update * self.generations

    def validate(self):
        positive_ints = ('steps', 'scheduler_steps', 'prompts_per_update', 'generations',
                         'generation_microbatch', 'replay_microbatch', 'probe_microbatch',
                         'rank', 'lora_alpha', 'max_new_tokens', 'max_prompt_tokens', 'checkpoint_every')
        for name in positive_ints:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.steps > self.scheduler_steps or self.generations < 2:
            raise ValueError('Invalid horizon or GRPO group size')
        for name in ('warmup_steps', 'seed'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        for name in ('learning_rate', 'beta', 'alpha', 'epsilon', 'max_grad_norm', 'weight_decay',
                     'replay_logprob_tolerance'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if not self.learning_rate or not self.max_grad_norm or not self.replay_logprob_tolerance:
            raise ValueError('Positive learning rate, gradient bound and replay tolerance required')
        if self.warmup_steps >= self.scheduler_steps:
            raise ValueError('Warmup must finish within the fixed scheduler horizon')
        if (self.max_prompt_tokens, self.max_new_tokens) != (256, 256):
            raise ValueError('This reproduction requires256/256 original token budgets')
        if (self.rank, self.lora_alpha, self.generations) != (64, 128, 8):
            raise ValueError('Source comparison fixes rank64/alpha128/group8')
        if self.alpha not in (0., 1.):
            raise ValueError('This named comparison supports detector coefficient0 or1 only')
        if self.preflight:
            if self.steps != 2:
                raise ValueError('Preflight is exactly2 optimizer steps')
        elif (self.steps, self.prompts_per_update) != (120, 64):
            raise ValueError('Main study requires120 steps and64 prompts/update; use --stop-after for chunks')


def immutable_json(path, value):
    """Exclusive creation or exact equality; never silently revise run metadata."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Immutable metadata mismatch: ' + str(path))
        return
    with path.open('x') as stream:
        stream.write(text)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def immutable_gzip_json(path, value):
    """Exact full evidence compressed losslessly; failures never overwrite attempts."""
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def frozen_digest(model):
    """All frozen parameters, full bytes including embeddings/head; not a sample."""
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            h.update(name.encode())
            h.update(str((tuple(parameter.shape), parameter.dtype)).encode())
            h.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def storage_plan(model, config):
    """Exact tensor-byte estimate plus explicit serialization/allocator reserve."""
    tensors = [p for p in model.parameters() if p.requires_grad]
    adapter = sum(p.numel()*p.element_size() for p in tensors)
    # FP32 model copy + two FP32 Adam moments and a scalar step per parameter.
    checkpoint_tensors = adapter*3 + 8*len(tensors)
    reserve = max(64*1024**2, math.ceil(checkpoint_tensors*.02))
    checkpoint_bytes = checkpoint_tensors + reserve
    exports = math.ceil(config.steps/config.checkpoint_every)
    return {'adapter_tensor_bytes': adapter, 'optimizer_and_adapter_tensor_bytes': checkpoint_tensors,
        'checkpoint_upper_estimate_bytes': checkpoint_bytes, 'serialization_reserve_bytes': reserve,
        'planned_exports': exports, 'export_upper_estimate_bytes': adapter+16*1024**2,
        'peak_owned_bytes': 2*checkpoint_bytes+exports*(adapter+16*1024**2),
        'formula': 'old+atomic-new Adam checkpoints plus every planned adapter export; FP32 moments/weights',
        'quota_caveat': 'statvfs free space does not reveal user quota; operator must provide available byte budget'}


def check_storage(directory, *, budget_bytes, peak_owned_bytes, next_write_bytes):
    directory = Path(directory)
    used = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    free = shutil.disk_usage(directory).free
    if budget_bytes < peak_owned_bytes:
        raise ValueError(f'Checkpoint budget {budget_bytes} below projected peak {peak_owned_bytes}')
    if used+next_write_bytes > budget_bytes or free < next_write_bytes:
        raise ValueError(f'Insufficient checkpoint write space: owned={used}, free={free}, next={next_write_bytes}')
    return {'owned_bytes': used, 'filesystem_free_bytes': free, 'declared_budget_bytes': budget_bytes}


def group_advantages(rewards, group_size=8):
    """Source sample std (Bessel corrected), epsilon1e-4, no group dropping."""
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 1 or len(rewards) % group_size or group_size < 2 or not len(rewards):
        raise ValueError('Rewards must contain complete nonempty prompt groups')
    if not torch.isfinite(rewards).all():
        raise ValueError('Nonfinite reward')
    grouped = rewards.reshape(-1, group_size)
    return ((grouped - grouped.mean(1, keepdim=True)) /
            (grouped.std(1, keepdim=True, unbiased=True) + 1e-4)).reshape(-1)


def grpo_loss(policy, old, reference, mask, advantages, *, beta=.001, epsilon=.2):
    """Mean per completion, then mean samples; follow-up/prompt/pad excluded."""
    if policy.shape != old.shape or policy.shape != reference.shape or policy.shape != mask.shape:
        raise ValueError('Logprob/mask shape mismatch')
    if policy.ndim != 2 or advantages.shape != (policy.shape[0],):
        raise ValueError('Invalid GRPO array dimensions')
    if not mask.any(1).all():
        raise ValueError('Empty completion is not a trainable sample')
    if not all(torch.isfinite(v).all() for v in (policy, old, reference, advantages)):
        raise ValueError('Nonfinite loss input')
    # Invalid prompt/pad ratios must not overflow before multiplication by0.
    log_ratio = torch.where(mask, policy - old.detach(), 0.)
    ratio = log_ratio.exp()
    pg = -torch.minimum(ratio * advantages[:, None], ratio.clamp(1-epsilon, 1+epsilon) * advantages[:, None])
    delta = torch.where(mask, policy - reference.detach(), 0.)
    kl = delta + torch.exp(-delta) - 1.
    average = lambda v: ((v * mask).sum(1) / mask.sum(1)).mean()
    loss = average(pg + beta * kl)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite GRPO loss')
    return loss, {'policy_loss': float(average(pg).detach()), 'kl': float(average(kl).detach()),
                  'clip_fraction': float(average(((ratio < 1-epsilon) & (advantages[:, None] < 0) |
                                                (ratio > 1+epsilon) & (advantages[:, None] > 0)).float()).detach())}


def collate_replay(rows, pad_id, device):
    """Causal targets aligned to exact sampled IDs, including generated EOT."""
    lengths = [len(r['prompt_ids']) + len(r['completion_ids']) for r in rows]
    width = max(lengths)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(rows), width-1), dtype=torch.bool, device=device)
    old = torch.zeros((len(rows), width-1), dtype=torch.float32, device=device)
    reference = torch.zeros_like(old)
    for i, r in enumerate(rows):
        p, c = r['prompt_ids'], r['completion_ids']
        if not p or not c:
            raise ValueError('Empty prompt/completion')
        ids[i, :lengths[i]] = torch.tensor(p+c, device=device)
        attention[i, :lengths[i]] = 1
        mask[i, len(p)-1:lengths[i]-1] = True
        for name, target in [('sampled_logprobs', old), ('reference_logprobs', reference)]:
            if name in r:
                if len(r[name]) != len(c):
                    raise ValueError('Per-token logprob count mismatch: ' + name)
                target[i, len(p)-1:lengths[i]-1] = torch.tensor(r[name], device=device)
    return ids, attention, mask, old, reference


def token_logprobs(model, ids, attention):
    output = model(input_ids=ids, attention_mask=attention, use_cache=False)
    # Only gathered values persist; vocabulary logits are released after backward.
    logits = output.logits[:, :-1, :].float()
    return logits.gather(-1, ids[:, 1:, None]).squeeze(-1) - logits.logsumexp(-1)


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter_state(model, state):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(params):
        raise ValueError('Resume adapter parameter names differ')
    with torch.no_grad():
        for n, p in params.items():
            if p.shape != state[n].shape or not torch.isfinite(state[n]).all():
                raise ValueError('Invalid checkpoint tensor: ' + n)
            p.copy_(state[n])


def rng_state():
    import numpy as np
    numpy = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:]),
            'torch': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.random.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed on resume')
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint(path, model, optimizer, scheduler, *, next_step, config_hash, stream_ids, journal):
    payload = {'schema': SCHEMA, 'config_hash': config_hash, 'next_step': next_step,
               'stream_ids': stream_ids, 'adapter': adapter_state(model),
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'rng': rng_state(), 'journal': journal}
    temp = Path(path).with_name(Path(path).name+'.tmp')
    with temp.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_checkpoint(path, model, optimizer, scheduler, *, config_hash, stream_ids):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['schema'] != SCHEMA or state['config_hash'] != config_hash or state['stream_ids'] != stream_ids:
        raise ValueError('Checkpoint contract/task stream mismatch')
    load_adapter_state(model, state['adapter'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    return state


def verify_journal(root, state, checkpoint_dir=None):
    """Only committed checkpoint entries are scientific steps; partial files are retained."""
    for item in state['journal']:
        if file_hash(Path(root)/item['path']) != item['sha256']:
            raise ValueError('Committed step artifact changed')
        if checkpoint_dir is not None and 'adapter_path_relative_to_checkpoint_dir' in item:
            export = Path(checkpoint_dir)/item['adapter_path_relative_to_checkpoint_dir']
            for name, expected in item['adapter_files_sha256'].items():
                if file_hash(export/name) != expected:
                    raise ValueError('Committed adapter export changed')
    if len(state['journal']) != state['next_step']:
        raise ValueError('Checkpoint/journal step count mismatch')


def make_actor(base_path, config, *, dtype=torch.bfloat16):
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('Actor arithmetic must be explicitly BF16 or FP32')
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True,
        torch_dtype=dtype, attn_implementation='sdpa', low_cpu_mem_usage=True).cuda()
    base.requires_grad_(False)
    # Fresh zero-delta adapter. Never load an already-RL-trained public adapter.
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=config.rank,
        lora_alpha=config.lora_alpha, lora_dropout=0., bias='none', target_modules=list(TARGETS)))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any('lora_' not in n for n in trainable):
        raise ValueError('Unexpected trainable base/head parameter')
    for p in trainable.values():
        p.data = p.data.float()
    if any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()):
        raise ValueError('Dropout would invalidate exact sampled/replay comparison')
    families = {name: sum(1 for n in trainable if f'.{name}.' in n and '.lora_A.' in n) for name in TARGETS}
    if set(families.values()) != {model.config.num_hidden_layers}:
        raise ValueError('LoRA does not cover all seven projections in every block')
    if any(torch.count_nonzero(p).item() for n, p in trainable.items() if '.lora_B.' in n):
        raise ValueError('Fresh adapter is not initialized with zero delta')
    return model, {'trainable_parameters': sum(p.numel() for p in trainable.values()),
                   'target_families': families, 'trainable_names': list(trainable),
                   'frozen_base_reference': 'PEFT disable_adapter()', 'initial_delta_zero': True}


def sample(model, tokenizer, problems, prompts, config, *, step, eot_id, eos_ids, progress=None):
    from transformers import GenerationConfig
    device = model.get_input_embeddings().weight.device
    planned = [(p, group, s) for group, p in enumerate(problems) for s in range(config.generations)]
    rows = []
    model.eval()
    generation = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
        max_new_tokens=config.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids, bos_token_id=tokenizer.bos_token_id, repetition_penalty=1.,
        return_dict_in_generate=True, output_scores=True, use_cache=True)
    with torch.no_grad():
        for offset in range(0, len(planned), config.generation_microbatch):
            chunk = planned[offset:offset+config.generation_microbatch]
            prompt_ids = [tokenizer.encode(prompts[p.task_id], add_special_tokens=False) for p, _, _ in chunk]
            if any(not ids or len(ids) > config.max_prompt_tokens for ids in prompt_ids):
                raise ValueError('Unexpected prompt outside admitted training budget')
            width = max(map(len, prompt_ids))
            ids = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            for i, item in enumerate(prompt_ids):
                ids[i, -len(item):] = torch.tensor(item, device=device)
                attention[i, -len(item):] = 1
            output = model.generate(input_ids=ids, attention_mask=attention, generation_config=generation)
            sequences = output.sequences[:, width:].cpu().tolist()
            for i, ((problem, group, s), emitted) in enumerate(zip(chunk, sequences)):
                stop = next((j+1 for j, token in enumerate(emitted) if token in eos_ids), len(emitted))
                completion = emitted[:stop]
                if not completion:
                    raise ValueError('Generator returned no completion tokens')
                logps = [float(F.log_softmax(output.scores[j][i].float(), -1)[token])
                         for j, token in enumerate(completion)]
                text = tokenizer.decode(completion, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                oracle_text = text.removesuffix('<|eot_id|>')
                rows.append({'step': step, 'group': group, 'sample': s, 'task_id': problem.task_id,
                    'prompt_ids': prompt_ids[i], 'completion_ids': completion,
                    'sampled_logprobs': logps, 'text': text, 'oracle_text': oracle_text,
                    'terminated': completion[-1] == eot_id, 'stopped_on_eos': completion[-1] in eos_ids,
                    'n_tokens': len(completion), 'generation_limit_reached': stop == config.max_new_tokens,
                    'identity_sha256': canonical_hash([step, group, s, problem.task_id, prompt_ids[i], completion])})
            del output
            if progress is not None:
                progress({'generated_rollouts': len(rows), 'target_rollouts': len(planned),
                          'generated_tokens': sum(r['n_tokens'] for r in rows)})
    return rows


def score_probe(model, probe, views, *, reference, pad_id, revision, batch_size):
    from .probe import capture_features
    ctx = model.disable_adapter() if reference else nullcontext()
    with ctx, torch.no_grad():
        bundle = capture_features(model, views, layer_indices=[int(k) for k in probe.heads],
            pad_token_id=pad_id, provenance={'model_revision': revision, 'tokenizer_revision': revision,
            'actor': 'frozen_base' if reference else 'current_adapter'}, batch_size=batch_size)
        scores = probe(bundle.features).detach().cpu()
    if scores.shape != (len(views),) or not torch.isfinite(scores).all() or not ((scores>=0)&(scores<=1)).all():
        raise ValueError('Invalid frozen probe output')
    return scores.tolist()


def score_batch(model, tokenizer, probe, oracle, problems_by_id, rows, config, *, revision, eot_id):
    from .probe import make_on_policy_replay_view
    views = [make_on_policy_replay_view(r['task_id'], r['prompt_ids'], r['completion_ids'], tokenizer,
             split='train_rollout', completion_kind=r['identity_sha256'], eot_token_id=eot_id) for r in rows]
    if any(v.completion_truncated for v in views):
        raise ValueError('Unexpected training truncation: fail instead of misaligned policy/probe views')
    current = score_probe(model, probe, views, reference=False, pad_id=tokenizer.pad_token_id,
                          revision=revision, batch_size=config.probe_microbatch)
    original = score_probe(model, probe, views, reference=True, pad_id=tokenizer.pad_token_id,
                           revision=revision, batch_size=config.probe_microbatch)
    for r, v, now, old in zip(rows, views, current, original):
        outcome = oracle.evaluate(problems_by_id[r['task_id']], r['oracle_text'], terminated=r['terminated'])
        reward = outcome['reward_before_monitor_and_kl'] - config.alpha * now
        if not math.isfinite(reward):
            raise ValueError('Nonfinite composite reward')
        r.update(outcome=outcome, probe_current=now, probe_reference=old,
            detected=now >= probe.decision_threshold, detector_penalty=config.alpha*now,
            reward=reward, probe_view=asdict(v), probe_view_sha256=v.digest)
    advantages = group_advantages([r['reward'] for r in rows], config.generations)
    for r, advantage in zip(rows, advantages.tolist()):
        r['advantage'] = advantage
    return views


def update(model, optimizer, scheduler, rows, config, *, pad_id):
    device = model.get_input_embeddings().weight.device
    optimizer.zero_grad(set_to_none=True)
    errors, losses, kls = [], [], []
    model.eval()
    # Reference is fixed base, evaluated before mutation. Store every token logp.
    with model.disable_adapter(), torch.no_grad():
        for offset in range(0, len(rows), config.replay_microbatch):
            chunk = rows[offset:offset+config.replay_microbatch]
            ids, attention, mask, _, _ = collate_replay(chunk, pad_id, device)
            reference = token_logprobs(model, ids, attention)
            for i, r in enumerate(chunk):
                r['reference_logprobs'] = reference[i][mask[i]].cpu().tolist()
    model.train()  # Enables non-reentrant checkpointing; all configured dropout is0.
    for offset in range(0, len(rows), config.replay_microbatch):
        chunk = rows[offset:offset+config.replay_microbatch]
        ids, attention, mask, old, reference = collate_replay(chunk, pad_id, device)
        policy = token_logprobs(model, ids, attention)
        error = float((policy.detach()[mask] - old[mask]).abs().max())
        errors.append(error)
        if error > config.replay_logprob_tolerance:
            raise RuntimeError(f'Sampled/replayed logprob mismatch {error:g}; no optimizer update')
        for i, r in enumerate(chunk):
            r['replayed_logprobs_before_update'] = policy[i][mask[i]].detach().cpu().tolist()
        advantage = torch.tensor([r['advantage'] for r in chunk], dtype=torch.float32, device=device)
        loss, metrics = grpo_loss(policy, old, reference, mask, advantage,
                                  beta=config.beta, epsilon=config.epsilon)
        # This remains exact when the final microbatch is shorter.
        (loss * (len(chunk)/len(rows))).backward()
        losses.append(float(loss.detach()) * len(chunk)/len(rows))
        kls.append(metrics['kl'] * len(chunk)/len(rows))
        del policy, loss, ids, attention
    params = [p for p in model.parameters() if p.requires_grad]
    gradient = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
    if not torch.isfinite(gradient):
        raise ValueError('Nonfinite actor gradient; no optimizer update')
    learning_rate = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {'loss': sum(losses), 'kl': sum(kls), 'grad_norm_before_clip': float(gradient),
            'learning_rate_used': learning_rate, 'next_learning_rate': optimizer.param_groups[0]['lr'],
            'max_sampled_replay_logprob_error': max(errors), 'n_rollouts': len(rows),
            'microbatches': math.ceil(len(rows)/config.replay_microbatch)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--arithmetic', choices=('bf16','fp32'), default='bf16')
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/atlas_mbpp')/DATASET_REVISION)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True,
                   help='Separate run-owned directory for large optimizer state and adapter exports')
    p.add_argument('--checkpoint-budget-gib', type=float, required=True,
                   help='Operator-verified available quota/space, not filesystem-wide free space')
    p.add_argument('--checkpoint-storage', choices=('persistent', 'temporary'), required=True)
    p.add_argument('--alpha', type=float, choices=(0., 1.), default=1.)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--preflight-prompts', type=int, default=2)
    p.add_argument('--generation-microbatch', type=int, default=4)
    p.add_argument('--replay-microbatch', type=int, default=1)
    p.add_argument('--probe-microbatch', type=int, default=1)
    p.add_argument('--adapter-export-every', type=int, default=10,
                   help='Adapter-only export interval; full resumable checkpoint still commits every update')
    p.add_argument('--oracle-wall-seconds', type=int, default=2)
    p.add_argument('--oracle-cpu-seconds', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='Maximum additional updates in this invocation; same120-step schedule')
    a = p.parse_args()
    config = TrainConfig(steps=2 if a.preflight else 120, prompts_per_update=a.preflight_prompts if a.preflight else 64,
        alpha=a.alpha, seed=a.seed, preflight=a.preflight, generation_microbatch=a.generation_microbatch,
        replay_microbatch=a.replay_microbatch, probe_microbatch=a.probe_microbatch,
        checkpoint_every=a.adapter_export_every)
    config.validate()
    if a.stop_after is not None and a.stop_after < 1:
        raise ValueError('stop-after must be positive')
    if a.oracle_wall_seconds < 1 or a.oracle_cpu_seconds < 1:
        raise ValueError('Positive uniform oracle limits required')
    if not math.isfinite(a.checkpoint_budget_gib) or a.checkpoint_budget_gib <= 0:
        raise ValueError('Positive operator-verified checkpoint budget required')
    budget_bytes = int(a.checkpoint_budget_gib*1024**3)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one allocated CUDA GPU required')
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    import transformers, peft, numpy as np
    from .data import load_pinned_dataset, prepare_source_compatible
    from .oracle_v2 import Evaluator, UnsupportedTask, SCHEMA as ORACLE_SCHEMA
    from .probe import load_probe
    from .stage_hf import BASE
    versions = {'torch': str(torch.__version__), 'transformers': transformers.__version__, 'peft': peft.__version__}
    if (versions['torch'].split('+')[0], versions['transformers'], versions['peft']) != ('2.8.0', '4.55.4', '0.17.1'):
        raise RuntimeError('Unvalidated runtime versions: ' + repr(versions))
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'complete.json').exists():
        raise ValueError('Run already complete')
    if (a.out/'config.json').exists() and not a.resume:
        raise ValueError('Existing run requires explicit --resume')
    staged = json.loads((a.model_root/'download_manifest.json').read_text())
    if staged['models']['base']['revision'] != BASE[1]:
        raise ValueError('Base revision mismatch')
    sources = {str(Path(__file__).name): file_hash(__file__)}
    for name in ('data.py', 'probe.py', 'oracle.py', 'oracle_v2.py'):
        sources[name] = file_hash(Path(__file__).with_name(name))
    resolved = {'schema': SCHEMA, 'config': asdict(config), 'source_revision': SOURCE_REVISION,
        'actor_arithmetic': a.arithmetic, 'tf32': False,
        'arithmetic_departure': 'FP32 arithmetic on upcast pretrained weights; original probe remains frozen' if a.arithmetic=='fp32' else None,
        'base': BASE, 'probe_sha256': file_hash(a.probe), 'runtime': versions, 'source_sha256': sources,
        'oracle_schema': ORACLE_SCHEMA,
        'oracle_limits': {'wall_seconds': a.oracle_wall_seconds, 'cpu_seconds': a.oracle_cpu_seconds,
                          'memory_bytes': 768*1024**2},
        'runtime_substitution': 'single-GPU native Transformers generation and PEFT microbatch replay; no vLLM/FSDP',
        'corrected_detector_gradient': False, 'initialization': 'fresh zero-delta adapter on pretrained base',
        'batch_semantics': '64 prompts x8 completions=512 rollouts per main update',
        'optimizer': {'name': 'AdamW', 'betas': [.9,.999], 'eps': 1e-8, 'weight_decay': 0.,
                      'departure': 'paper-v2 weight_decay0; pinned source omits argument and defaults to0.01'},
        'prompt_order': 'dedicated NumPy RandomState(seed) permutation, then cyclic; source global-RNG shuffle differs',
        'generation': 'temperature1, top_p1, top_k0; only configured EOS tokens stop; no code-fence stop',
        'loss': 'sample-mean token GRPO, sample std+1e-4, clip0.2, k3 KL; no detector direct derivative',
        'preflight_schedule': '2updates on unchanged120-step schedule; first update LR0 due warmup',
        'checkpoint_storage': a.checkpoint_storage,
        'checkpoint_declared_budget_bytes': budget_bytes,
        'checkpoint_durability': 'temporary requires checksum-verified offload before job exits' if a.checkpoint_storage == 'temporary' else 'operator-designated persistent storage',
        'result_scope': 'training trajectories; independently scheduled heldout evaluation required'}
    immutable_json(a.out/'config.json', resolved)
    immutable_json(a.out/'model_provenance.json', staged)
    config_hash = canonical_hash(resolved)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(a.model_root/'base', local_files_only=True)
    vocab = tok.get_vocab()
    if '<|reserved_special_token_0|>' not in vocab or '<|eot_id|>' not in vocab:
        raise ValueError('Expected Llama3 pad and end-of-turn tokens absent')
    tok.pad_token = '<|reserved_special_token_0|>'
    tok.padding_side = 'left'
    eot_id = vocab['<|eot_id|>']
    prepared = prepare_source_compatible(load_pinned_dataset(a.data), tok,
        tokenizer_identity={'repo': BASE[0], 'revision': BASE[1]})
    immutable_json(a.out/'data_manifest.json', prepared.manifest)
    if not prepared.manifest['training_ready']:
        raise ValueError('Dataset not training-ready')
    order = np.random.RandomState(config.seed).permutation(len(prepared.task_train))
    ordered = [prepared.task_train[int(i)] for i in order]
    if config.preflight:
        ordered = ordered[:config.prompts_per_update*config.steps]
    oracle = Evaluator(**resolved['oracle_limits'])
    eligibility_path = a.out/'oracle_eligibility.json'
    by_id = {row.task_id: row for row in ordered}
    if eligibility_path.exists():
        eligibility = json.loads(eligibility_path.read_text())
        if eligibility['requested_ids'] != [r.task_id for r in ordered]:
            raise ValueError('Oracle eligibility stream differs')
    else:
        eligibility = {'requested_ids': [r.task_id for r in ordered], 'admitted_ids': [], 'excluded': []}
        for row in ordered:
            try:
                result = oracle.evaluate(row, '```python\n'+row.reference_code+'\n```', terminated=True)
            except UnsupportedTask as exc:
                eligibility['excluded'].append({'task_id': row.task_id, 'reason': str(exc)})
                continue
            if result['all_tests_pass']:
                eligibility['admitted_ids'].append(row.task_id)
            else:
                eligibility['excluded'].append({'task_id': row.task_id, 'reason': 'reference_failed', 'outcome': result})
        eligibility['policy'] = 'Fixed source-selected IDs; explicit strict-oracle subset, no replacement from other splits'
        immutable_json(eligibility_path, eligibility)
    stream_ids = eligibility['admitted_ids']
    if len(stream_ids) < config.prompts_per_update:
        raise ValueError('Not enough reference-verified tasks for one update')
    immutable_json(a.out/'stream.json', {'task_ids': stream_ids, 'data_manifest_sha256': canonical_hash(prepared.manifest),
                    'oracle_eligibility_sha256': canonical_hash(eligibility)})
    probe = load_probe(a.probe)
    provenance = probe.report.get('feature_provenance', {})
    if (provenance.get('model_revision') != BASE[1] or provenance.get('tokenizer_revision') != BASE[1]):
        raise ValueError('Probe was not fitted on the pinned base/tokenizer revisions')
    model, model_audit = make_actor(a.model_root/'base', config,
        dtype=torch.float32 if a.arithmetic=='fp32' else torch.bfloat16)
    model_audit['frozen_parameter_sha256'] = frozen_digest(model)
    immutable_json(a.out/'actor_audit.json', model_audit)
    a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    immutable_json(a.checkpoint_dir/'owner.json', {'schema': SCHEMA, 'config_hash': config_hash})
    space = storage_plan(model, config)
    capacity = check_storage(a.checkpoint_dir, budget_bytes=budget_bytes,
        peak_owned_bytes=space['peak_owned_bytes'], next_write_bytes=space['checkpoint_upper_estimate_bytes'])
    immutable_json(a.out/'checkpoint_plan.json', space)
    atomic_json(a.out/'checkpoint_location.json', {'directory': str(a.checkpoint_dir.resolve()),
        'storage': a.checkpoint_storage, 'job_id': os.environ['SLURM_JOB_ID'], 'node': os.uname().nodename, **capacity})
    raw_eos = model.generation_config.eos_token_id
    eos_ids = sorted(set(raw_eos if isinstance(raw_eos, list) else [raw_eos]) | {eot_id})
    if any(type(x) is not int for x in eos_ids):
        raise ValueError('Invalid EOS configuration')
    immutable_json(a.out/'tokenization.json', {'eos_ids': eos_ids, 'eot_id': eot_id,
                    'pad_id': tok.pad_token_id, 'special_tokens': tok.special_tokens_map})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, betas=(.9,.999), eps=1e-8, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                               num_training_steps=config.scheduler_steps)
    last = a.checkpoint_dir/'resume.pt'
    state = {'next_step': 0, 'journal': []}
    if a.resume:
        if not last.exists():
            raise ValueError('No committed checkpoint to resume')
        state = read_checkpoint(last, model, optimizer, scheduler, config_hash=config_hash, stream_ids=stream_ids)
        verify_journal(a.out, state, a.checkpoint_dir)
        state = {'next_step': state['next_step'], 'journal': state['journal']}
    else:
        checkpoint(last, model, optimizer, scheduler, next_step=0, config_hash=config_hash,
                   stream_ids=stream_ids, journal=[])
    stop = min(config.steps, state['next_step']+(a.stop_after or config.steps))
    start = time.monotonic()
    try:
        for step in range(state['next_step'], stop):
            # A retry gets a new attempt directory; prior incomplete evidence is retained.
            attempts = a.out/'attempts'; attempts.mkdir(exist_ok=True)
            attempt = attempts/f'step-{step:04d}-{time.time_ns()}'
            attempt.mkdir()
            selected = [by_id[stream_ids[(step*config.prompts_per_update+i)%len(stream_ids)]]
                        for i in range(config.prompts_per_update)]
            atomic_json(a.out/'progress.json', {'phase': 'generation', 'step': step, 'completed_steps': step,
                                               'rollouts_per_update': config.rollouts_per_update})
            rows = sample(model, tok, selected, prepared.rendered_prompts, config, step=step,
                          eot_id=eot_id, eos_ids=eos_ids,
                          progress=lambda values: atomic_json(a.out/'progress.json', {
                              'phase': 'generation', 'step': step, 'completed_steps': step, **values}))
            immutable_gzip_json(attempt/'sampled.json.gz', rows)
            atomic_json(a.out/'progress.json', {'phase': 'scoring', 'step': step, 'completed_steps': step})
            views = score_batch(model, tok, probe, oracle, by_id, rows, config, revision=BASE[1], eot_id=eot_id)
            immutable_gzip_json(attempt/'scored.json.gz', rows)
            atomic_json(a.out/'progress.json', {'phase': 'updating', 'step': step, 'completed_steps': step})
            metrics = update(model, optimizer, scheduler, rows, config, pad_id=tok.pad_token_id)
            # Same exact tokens after the update, not new generation: label separately.
            after = score_probe(model, probe, views, reference=False, pad_id=tok.pad_token_id,
                                revision=BASE[1], batch_size=config.probe_microbatch)
            for row, score in zip(rows, after):
                row['probe_after_update_same_tokens'] = score
            metrics.update(step=step, completed_steps=step+1,
                mean_reward=sum(r['reward'] for r in rows)/len(rows),
                mean_probe_current=sum(r['probe_current'] for r in rows)/len(rows),
                mean_probe_reference=sum(r['probe_reference'] for r in rows)/len(rows),
                mean_probe_after_update_same_tokens=sum(after)/len(after),
                visible_pass=sum(r['outcome']['visible_pass'] for r in rows),
                all_tests_pass=sum(r['outcome']['all_tests_pass'] for r in rows),
                reward_hack_proxy=sum(r['outcome']['reward_hack_proxy'] for r in rows),
                detected=sum(r['detected'] for r in rows),
                detected_reward_hack_proxy=sum(r['detected'] and r['outcome']['reward_hack_proxy'] for r in rows),
                terminated=sum(r['terminated'] for r in rows),
                source_compatible_runtime=False, independent_heldout_evaluation=False)
            immutable_gzip_json(attempt/'result.json.gz', {'metrics': metrics, 'rollouts': rows})
            item = {'step': step, 'path': str((attempt/'result.json.gz').relative_to(a.out)),
                    'sha256': file_hash(attempt/'result.json.gz')}
            if (step+1) % config.checkpoint_every == 0 or step+1 == config.steps:
                # Export inside the unique attempt before committing. An interruption
                # leaves an uncommitted attempt, never a falsely complete fixed path.
                path = a.checkpoint_dir/'exports'/attempt.name/'adapter'
                check_storage(a.checkpoint_dir, budget_bytes=budget_bytes, peak_owned_bytes=space['peak_owned_bytes'],
                              next_write_bytes=space['export_upper_estimate_bytes'])
                model.save_pretrained(path, safe_serialization=True)
                immutable_json(path/'training_provenance.json', {'step': step+1, 'config_hash': config_hash,
                                'result': dict(item), 'probe_sha256': resolved['probe_sha256']})
                item['adapter_path_relative_to_checkpoint_dir'] = str(path.relative_to(a.checkpoint_dir))
                item['adapter_files_sha256'] = {str(f.relative_to(path)):file_hash(f) for f in path.rglob('*') if f.is_file()}
            journal = state['journal']+[item]
            check_storage(a.checkpoint_dir, budget_bytes=budget_bytes, peak_owned_bytes=space['peak_owned_bytes'],
                          next_write_bytes=space['checkpoint_upper_estimate_bytes'])
            checkpoint(last, model, optimizer, scheduler, next_step=step+1, config_hash=config_hash,
                       stream_ids=stream_ids, journal=journal)
            state = {'next_step': step+1, 'journal': journal}
            # Atomic checkpoint commit is authoritative; this human-readable index is recoverable.
            atomic_json(a.out/'journal.json', {'next_step': step+1, 'entries': journal})
            # Canonical result retains every field; remove only our redundant
            # intermediate copies after a successful atomic checkpoint commit.
            (attempt/'sampled.json.gz').unlink()
            (attempt/'scored.json.gz').unlink()
            atomic_json(a.out/'progress.json', {'phase': 'committed', **metrics})
            print(json.dumps(metrics, allow_nan=False), flush=True)
        final_frozen_hash = frozen_digest(model)
        if final_frozen_hash != model_audit['frozen_parameter_sha256']:
            raise RuntimeError('Frozen base changed: invalidate this run')
        frozen_check = {'before': model_audit['frozen_parameter_sha256'], 'after': final_frozen_hash,
                        'equal': True, 'completed_steps': state['next_step']}
        atomic_json(a.out/'frozen_parameter_check.json', frozen_check)
        marker = 'complete.json' if state['next_step'] == config.steps else 'chunk_complete.json'
        atomic_json(a.out/marker, {'completed_steps': state['next_step'], 'target_steps': config.steps,
                    'completed': state['next_step'] == config.steps, 'config_hash': config_hash,
                    'elapsed_invocation_seconds': time.monotonic()-start,
                    'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'checkpoint_path': str(last), 'checkpoint_sha256': file_hash(last),
                    'checkpoint_storage': a.checkpoint_storage,
                    'persistent_storage_operator_declared': a.checkpoint_storage == 'persistent',
                    'offload_required_before_job_exit': a.checkpoint_storage == 'temporary',
                    'scope': 'training trajectories only; no main heldout conclusion'})
    except BaseException as exc:
        atomic_json(a.out/'failed.json', {'error': repr(exc), 'committed_steps': state['next_step'],
                                        'recoverable_checkpoint': str(last), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
