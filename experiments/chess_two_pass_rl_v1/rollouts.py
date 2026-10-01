import time
from .environment import run_episode,verify_episode,summarize


def episode(propose,row,seed,config,scorer):
    outputs=[]
    def command(fen,depth):
        out=propose(fen,depth);assert out['token_ids'];outputs.append(out);return out['action']
    start=time.monotonic()
    value=run_episode(command,fen=row['fen'],seed=seed,config=config,scorer=scorer)
    value.update(position_id=row['id'],stop_seed=seed,proposal_readouts=outputs,wall_seconds=time.monotonic()-start)
    verify_episode(value,scorer)
    return value


def evaluate_batch(propose,rows,seeds,config,scorer):
    assert len(rows)==len(seeds)
    # Evaluation only: greedy command reuse at identical weights and prefixes.
    cache={}
    def cached(fen,depth):
        key=(fen,depth)
        if key not in cache:cache[key]=propose(fen,depth)
        return cache[key]
    episodes=[episode(cached,row,seed,config,scorer) for row,seed in zip(rows,seeds)]
    return dict(summary=summarize(episodes),episodes=episodes)
