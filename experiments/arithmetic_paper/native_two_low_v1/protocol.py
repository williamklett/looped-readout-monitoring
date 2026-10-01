"""Native two-digit outputs; frozen prospective predictions, no fitted constants."""
import random

DEPTHS=7
HAZARDS=(.25,.5,.75)
SEEDS=(17,29,43)
FACTORS=(.1,.2,.4,.5)
THRESHOLD=.1

def stopping_weights(p):
    assert 0<p<1
    q=[p*(1-p)**k for k in range(6)]+[(1-p)**6]
    w=[sum(q[n-1]/(n-1) for n in range(k+1,8)) for k in range(1,8)]
    return q,w

def cells():
    out=[]
    for seed in SEEDS:
        for p in HAZARDS:
            q,w=stopping_weights(p);boundary=q[3]/(.1*w[3])
            for f in FACTORS:
                lam=f*boundary
                out.append(dict(index=len(out),seed=seed,stop_probability=p,audit_probability=1-p,
                    boundary_factor=f,penalty_weight=lam,primary_lambda=boundary,
                    predicted_mass=[min(1.,q[k]/(lam*w[k])) if lam*w[k] else 1. for k in range(7)],
                    name=f'p{p:g}-f{f:g}-s{seed}'))
    return out

def depth_schedule(step,seed):
    rng=random.Random(197000+step+10000*(seed-17))
    ks=list(range(1,8))+[rng.randint(1,7)];rng.shuffle(ks)
    return ks

def predictions():
    return dict(schema='native-two-digit-ce-low-extension-v1',formula='c[j,k]*=min(1,q[k]/(lambda*w[k]))',
        normalization='Mean over two CE terms and mean over two pre-exit penalties; factors 1/2 cancel.',
        prefix='Position2 theoretical prediction is conditioned on the correct first digit (teacher forcing).',
        limitation='Independent probability relaxation, not an SGD convergence or greedy accuracy theorem.',
        primary_depth=4,threshold=.1,cells=cells(),
        stopping=[dict(p=p,audit=1-p,q=stopping_weights(p)[0],w=stopping_weights(p)[1]) for p in HAZARDS])
