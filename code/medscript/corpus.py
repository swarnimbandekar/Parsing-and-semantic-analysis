"""Labelled corpus construction (valid programs + fault-injected mutants)."""

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

from .driver import compile_source
from .diagnostics import FAMILIES
from .generator import generate_valid, to_source
from .mutations import OPERATORS, PRIORITY

# operators whose faults are invisible to the rule-based front end
SOFT_OPS = {"logic_inverted", "logic_wrong_message"}


@dataclass
class Sample:
    source: str
    label: int
    family: str
    op: str
    secondary: Optional[str] = None


def manifests(family, op_name, res):
    if family == "lexical":
        return any(d.code.startswith("LEX") for d in res.diagnostics)
    if family == "syntax":
        return any(d.code.startswith("SYN") for d in res.diagnostics)
    if family == "logic" and op_name in SOFT_OPS:
        return True
    return any(d.family == family for d in res.diagnostics)


def make_sample(rng: random.Random, family: str, double_prob: float = 0.2,
                stats: Optional[dict] = None) -> Optional[Sample]:
    """Return one labelled program of the requested family."""
    if family == "valid":
        for _ in range(20):
            lines, _ = generate_valid(rng)
            src = to_source(lines)
            if compile_source(src, codegen=False).ok:
                return Sample(src, 0, "valid", "none")
        return None
    for _ in range(60):
        lines, _ = generate_valid(rng)
        secondary = None
        if rng.random() < double_prob:
            lower = PRIORITY[PRIORITY.index(family) + 1:]
            if lower:
                sf = rng.choice(lower)
                out = rng.choice(OPERATORS[sf])(lines, rng)
                if out:
                    lines, secondary = out, sf
        op = rng.choice(OPERATORS[family])
        out = op(lines, rng)
        if stats is not None:
            st = stats.setdefault(op.__name__, [0, 0])
            st[0] += 1
        if out is None or out == lines:
            continue
        src = to_source(out)
        res = compile_source(src, codegen=False)
        if manifests(family, op.__name__, res):
            if stats is not None:
                stats[op.__name__][1] += 1
            return Sample(src, FAMILIES.index(family), family, op.__name__, secondary)
    return None


def build_corpus(per_class: int, seed: int = 0, double_prob: float = 0.2, stats=None):
    """Balanced corpus: `per_class` unique programs for each of the 10 classes."""
    rng = random.Random(seed)
    seen, out = set(), []
    for fam in FAMILIES:
        n = 0
        while n < per_class:
            s = make_sample(rng, fam, double_prob, stats)
            if s is None:
                continue
            h = hashlib.md5(s.source.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append(s)
            n += 1
    return out
