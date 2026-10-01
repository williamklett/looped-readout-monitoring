"""Arm-independent training orders and exogenous stopping seeds."""
import random


def training_order(rows, seed, epochs):
    rng = random.Random(5000000 + seed)
    for epoch in range(1, epochs+1):
        order = list(rows)
        rng.shuffle(order)
        for offset, row in enumerate(order):
            move = (epoch-1)*len(rows)+offset+1
            yield dict(move=move, epoch=epoch, row=row,
                       stop_seed=1000000+seed*10000+move)


def evaluation_seeds(split, row_index, count):
    # Same draws across arms, training seeds and checkpoints; never training RNG.
    base = dict(train=20000000, development=30000000, test=40000000)[split]
    return [base + row_index*1000 + j for j in range(count)]
