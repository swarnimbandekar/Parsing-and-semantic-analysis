"""Random-testing helper: do original and optimised IR behave identically?"""

import random

from . import ast_nodes as A
from .ir import run


def random_inputs(prog, rng: random.Random):
    env = {}
    for it in prog.items:
        if not isinstance(it, A.InputDecl):
            continue
        if it.ty.nullable and rng.random() < 0.3:
            env[it.name] = None
        elif it.ty.base == "string":
            env[it.name] = "Jane Doe"
        elif it.ty.base == "int":
            lo, hi = it.rng or (0, 100)
            env[it.name] = rng.randint(int(lo), int(hi))
        else:
            lo, hi = it.rng or (0.0, 100.0)
            env[it.name] = rng.uniform(lo, hi)
    return env


def equivalent(result, trials=20, seed=0):
    """True iff ir and ir_opt produce identical output events on random inputs."""
    rng = random.Random(seed)
    for _ in range(trials):
        env = random_inputs(result.ast, rng)
        a = run(result.ir, env)
        b = run(result.ir_opt, env)
        if a != b:
            return False, env, a, b
    return True, None, None, None
