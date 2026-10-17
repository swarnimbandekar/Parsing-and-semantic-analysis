"""Compilation driver: runs the phases in order and records timings."""

import time
from dataclasses import dataclass, field
from typing import List

from .lexer import tokenize
from .parser import parse
from .semantic import analyze
from .ir import lower, optimise, count_instructions
from .diagnostics import Diagnostic


@dataclass
class CompileResult:
    source: str
    tokens: list = field(default_factory=list)
    ast: object = None
    diagnostics: List[Diagnostic] = field(default_factory=list)
    ir: object = None
    ir_opt: object = None
    timings: dict = field(default_factory=dict)
    sem_ran: bool = False

    @property
    def errors(self):
        return [d for d in self.diagnostics if d.is_error]

    @property
    def warnings(self):
        return [d for d in self.diagnostics if not d.is_error]

    @property
    def ok(self):
        return not self.errors


def compile_source(src: str, codegen: bool = True) -> CompileResult:
    """Lex, parse, analyse and (when error free) lower and optimise `src`."""
    r = CompileResult(src)
    t0 = time.perf_counter()
    r.tokens, lex_d = tokenize(src)
    t1 = time.perf_counter()
    r.ast, syn_d = parse(r.tokens)
    t2 = time.perf_counter()
    r.diagnostics = list(lex_d) + list(syn_d)
    sem_d = []
    if not syn_d:                       # semantic phase needs a trustworthy AST
        sem_d, _ = analyze(r.ast)
        r.sem_ran = True
    t3 = time.perf_counter()
    r.diagnostics += sem_d
    r.diagnostics.sort(key=lambda d: (d.line, d.col))
    r.timings = {"lex": t1 - t0, "parse": t2 - t1, "sem": t3 - t2}
    if codegen and r.ok:
        r.ir = lower(r.ast)
        t4 = time.perf_counter()
        r.ir_opt = optimise(r.ir)
        t5 = time.perf_counter()
        r.timings["ir"] = t4 - t3
        r.timings["opt"] = t5 - t4
        r.timings["n_ir"] = count_instructions(r.ir)
        r.timings["n_ir_opt"] = count_instructions(r.ir_opt)
    return r
