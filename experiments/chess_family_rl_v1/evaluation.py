"""Greedy board-level evaluation, with explicit unobserved stopping-tail bounds."""
from experiments.chess_two_pass_rl_v1.rollouts import episode
from experiments.chess_two_pass_rl_v1.environment import summarize
from .schedule import evaluation_seeds
from .metrics import outcome, stopping_metrics


def evaluate(propose, rows, split, cfg, scorer, design):
    all_episodes, boards = [], []
    for i, row in enumerate(rows):
        cache = {}
        def cached(fen, depth):
            assert fen == row['fen']
            if depth not in cache:
                cache[depth] = propose(fen, depth)
            return cache[depth]
        # Supplementary readouts at fixed weights; they do not affect deployment
        # stopping or training. Missing depths beyond these remain unresolved.
        depth = cfg.fixed_depth if cfg.mode == 'fixed_depth_audit' else design['minimum_evaluation_depth']
        for k in range(1, depth+1):
            cached(row['fen'], k)
        seeds = evaluation_seeds(split, i, design['evaluation_stop_draws'])
        episodes = [episode(cached, row, seed, cfg, scorer) for seed in seeds]
        outcomes = {k:outcome(row['fen'], r['action'], scorer) for k,r in cache.items()}
        boards.append(dict(position_id=row['id'], family=row['family'], fen=row['fen'],
                           readouts=[cache[k] for k in sorted(cache)],
                           depth_outcomes=outcomes, stopping_metrics=stopping_metrics(outcomes, cfg),
                           summary=summarize(episodes), episodes=episodes))
        all_episodes.extend(episodes)
    return dict(split=split, board_count=len(rows),
                family_count=len({r['family'] for r in rows}),
                summary=summarize(all_episodes), boards=boards,
                stopping_draws_are_not_independent_boards=True)
