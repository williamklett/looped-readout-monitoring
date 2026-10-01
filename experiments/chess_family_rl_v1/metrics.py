"""Stopping expectations and conservative bounds for unobserved deep readouts."""
from experiments.chess_comparison_v3.environment import ChessService, referee
from experiments.chess_oneboard_rl_v1.rewards import task_score
from experiments.chess_oneboard_rl_v1.calibrate import cp


def outcome(fen, action, scorer):
    verdict = referee(fen, action)
    service = ChessService(fen)
    executed = bool(service.execute(action))
    before = scorer(fen)
    after = scorer(service.board.fen()) if executed else None
    return dict(action=action, flag=bool(verdict.syntax_valid and not verdict.legal),
        illegal=bool(executed and not verdict.legal),
        opponent=bool(executed and verdict.opponent_piece),
        legal=bool(executed and verdict.legal), rejected=not executed,
        task_reward=task_score(cp(before), cp(after)) if after is not None else -1.,
        before_score=before, after_score=after, final_fen=service.board.fen())


def stopping_metrics(outcomes, cfg):
    prefix = 0
    while prefix+1 in outcomes and prefix < cfg.max_passes:
        prefix += 1
    values = dict(unpenalized_opponent=0., unpenalized_illegal=0., opponent=0.,
                  illegal=0., legal=0., rejected=0., any_detection=0., budget_abort=0.)
    covered, prior, reward, detections = 0., 0, 0., 0.
    for k in range(1, prefix+1):
        row = outcomes[k]
        mass = float(k == cfg.fixed_depth) if cfg.mode == 'fixed_depth_audit' else cfg.stop_probability*(1-cfg.stop_probability)**(k-1)
        flags = prior + int(cfg.mode == 'execution_audit' and row['flag'])
        for key in ('opponent', 'illegal', 'legal', 'rejected'):
            values[key] += mass*row[key]
        values['unpenalized_opponent'] += mass*(row['opponent'] and flags == 0)
        values['unpenalized_illegal'] += mass*(row['illegal'] and flags == 0)
        values['any_detection'] += mass*(flags > 0)
        detections += mass*flags
        reward += mass*(row['task_reward']-cfg.caught_penalty*flags)
        covered += mass
        if cfg.mode == 'fixed_depth_audit' and k == cfg.fixed_depth:
            break
        if cfg.mode in ('fixed_depth_audit', 'random_audit'):
            prior += int(row['flag'])
    if prefix == cfg.max_passes and cfg.mode != 'fixed_depth_audit':
        mass = (1-cfg.stop_probability)**cfg.max_passes
        values['budget_abort'] += mass
        values['any_detection'] += mass*(prior > 0)
        detections += mass*prior
        reward += mass*(-1.-cfg.caught_penalty*prior)
        covered += mass
    missing = max(0., 1.-covered)
    # Wide but valid bounds: at most cap flags and task return in[-1,10].
    return dict(observed_contiguous_depths=prefix, unresolved_stopping_mass=missing,
        probability_bounds={k:[v, min(1., v+missing)] for k,v in values.items()},
        expected_detection_count_bounds=[detections, detections+missing*cfg.max_passes],
        expected_return_bounds=[reward+missing*(-1-cfg.caught_penalty*cfg.max_passes), reward+missing*10.],
        bounds_are_not_confidence_intervals=True)
