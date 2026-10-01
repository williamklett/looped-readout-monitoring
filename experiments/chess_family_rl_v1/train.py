"""Repeated-board native RL, matched arms, final held-out test at fixed budget."""
import argparse
import json
from pathlib import Path
import torch
from .common import load, provenance, sha, INITIALIZER_SHA, config
from .schedule import training_order
from .evaluation import evaluate
from experiments.chess_two_pass_rl_v1.interface import command, prompt
from experiments.chess_two_pass_rl_v1.learning import update, disposable
from experiments.chess_two_pass_rl_v1.rollouts import episode
from experiments.chess_two_pass_rl_v1.environment import summarize
from experiments.chess_comparison_v3.runtime import policy, save_adapter
from experiments.chess_comparison_v3.storage import save
from experiments.chess_comparison_v3.integrity import check_runtime
from experiments.chess_comparison_v3.preflight import encode
from experiments.chess_comparison_v3.evidence import model_tensor_sha
from experiments.chess_onemove_v1.scoring import PositionScorer


def main():
    ap = argparse.ArgumentParser()
    for name in ('project', 'root', 'initializer', 'checkpoint-root'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--index', type=int, required=True)
    a = ap.parse_args()
    d, data, parent, data_root = load(a.project)
    run = d['runs'][a.index]
    assert sha(a.initializer) == INITIALIZER_SHA
    out = a.root/'training'/run['run_id']
    out.mkdir(parents=True, exist_ok=False)
    save(out/'config.json', dict(**run, **provenance(), design=d,
        structured_output=False, original_capability_criterion_passed=False,
        user_authorized_limited_baseline=True, adaptive_extension=True))
    history, episodes, actual = [], [], 0
    cfg = config(d, run['arm'])
    try:
        with PositionScorer(a.project/parent['engine']['path'], 8) as scorer:
            scorer.cache.update(json.loads((data_root/'score-cache.json').read_text()))
            with policy(a.initializer, run['seed']) as (model, _, tok):
                model.eval()
                greedy = lambda fen, depth: command(model, tok, fen, depth, d['temperature'], True)
                sampled = lambda fen, depth: command(model, tok, fen, depth, d['temperature'], False)
                row = data['train'][0]
                save(out/'progress.json', dict(phase='runtime_integrity', moves=0,
                    optimizer_steps=0, target_moves=d['moves']))
                ids = torch.tensor([encode(tok, prompt(row['fen']), 'chat')], device=next(model.parameters()).device)
                save(out/'runtime-integrity.json', check_runtime(model, ids, lambda k:greedy(row['fen'], k)))
                save(out/'tokenization.json', dict(new_vocabulary=False, output_mask=False,
                    prompt_variant='piece_inventory_examples', same_decoder_for_audit_and_execution=True))
                rng, cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state_all()
                try:
                    probe = episode(sampled, row, 7000000+run['seed'], cfg, scorer)
                    save(out/'gradient-integrity.json', disposable(model, tok, probe, d['learning_rate']))
                finally:
                    torch.random.set_rng_state(rng)
                    torch.cuda.set_rng_state_all(cuda)
                source = model_tensor_sha(model)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                       lr=d['learning_rate'], weight_decay=0.)

                def measure(move, split):
                    save(out/'progress.json', dict(phase=split+'_evaluation', moves=move,
                        optimizer_steps=actual, target_moves=d['moves']))
                    result = evaluate(greedy, data[split], split, cfg, scorer, d)
                    model.assert_frozen()
                    # All evaluation traces use the existing lossless compressed format.
                    save(out/f'development-{split}-{move:04d}.json', dict(**result,
                        moves=move, optimizer_steps=actual,
                        policy_tensor_sha256=model_tensor_sha(model)))

                measure(0, 'development')
                for item in training_order(data['train'], run['seed'], d['epochs']):
                    move, row = item['move'], item['row']
                    save(out/'progress.json', dict(phase='sampled_rollout', moves=move-1,
                        epoch=item['epoch'], optimizer_steps=actual, target_moves=d['moves']))
                    value = episode(sampled, row, item['stop_seed'], cfg, scorer)
                    before = model_tensor_sha(model)
                    result = update(model, tok, opt, value, d['clip_norm'])
                    actual += int(result['optimizer_step_applied'])
                    after = model_tensor_sha(model)
                    save(out/f'game-{move:04d}.json', dict(move=move, epoch=item['epoch'],
                        optimizer_steps=actual, **result, tensor_sha_before=before,
                        tensor_sha_after=after, episode=value))
                    episodes.append(value)
                    history.append(dict(step=move, moves=move, epoch=item['epoch'],
                        position_id=row['id'], optimizer_steps=actual, reward=value['reward'],
                        reason=value['reason'], gradient_norm=result['gradient_norm'],
                        max_parameter_change=result['max_parameter_change'],
                        summary=summarize(episodes[-30:])))
                    save(out/'history.json', history)
                    save(out/'progress.json', dict(phase='update_complete', moves=move,
                        epoch=item['epoch'], optimizer_steps=actual, target_moves=d['moves']))
                    if move in d['evaluation_moves']:
                        measure(move, 'development')
                path = a.checkpoint_root/run['run_id']/'adapter.pt'
                save_adapter(model, path)
                # No test-time optimization or checkpoint selection.
                measure(d['moves'], 'train')
                measure(d['moves'], 'test')
                final = dict(completed=True, **run, **provenance(), moves=d['moves'],
                    epochs=d['epochs'], optimizer_steps=actual,
                    initializer_tensor_sha256=source, adapter_tensor_sha256=model_tensor_sha(model),
                    adapter_sha256=sha(path), checkpoint_path=str(path), frozen_verified=True,
                    adaptive_extension=True, structured_output=False,
                    test_evaluations=1, training_summary=summarize(episodes))
                save(out/'adapter-manifest.json', final)
                save(out/'complete.json', final)
    except BaseException as exc:
        save(out/'error.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
