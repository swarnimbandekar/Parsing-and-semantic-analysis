"""Feature extraction: CompileResult -> fixed-length numeric vector.

Feature groups
  tok  : token histogram and bracket balance (lexer output)
  ast  : structural counts of the parse tree
  diag : per-code diagnostic counts and first-error descriptors (all phases)
  pat  : clinical rule patterns (comparison direction vs alert vocabulary)
"""

import math
import re

import numpy as np

from . import ast_nodes as A
from .diagnostics import CODE_LIST, FAMILIES, PHASES
from .generator import HIGH_WORDS, LOW_WORDS

_KIND_GROUPS = {
    "ID": ["ID"], "NUM": ["NUM"], "STR": ["STR"], "UNIT": ["UNIT"],
    "kw_decl": ["input", "const", "func", "let"],
    "kw_ctrl": ["if", "else", "while", "return"],
    "kw_type": ["int", "float", "bool", "string"],
    "kw_misc": ["phi", "range", "true", "false"],
    "lpar": ["("], "rpar": [")"], "lbrace": ["{"], "rbrace": ["}"],
    "semi": [";"], "colon": [":"], "comma": [","], "qmark": ["?"], "caret": ["^"],
    "arith": ["+", "-", "*", "/", "%"],
    "cmp_lt": ["<", "<="], "cmp_gt": [">", ">="], "cmp_eq": ["==", "!="],
    "logic": ["&&", "||", "!"], "assign": ["="], "arrow": ["->"],
}

_NODE_TYPES = [A.Let, A.Assign, A.If, A.While, A.Return, A.ExprStmt, A.Block, A.InputDecl,
               A.ConstDecl, A.FuncDecl, A.Call, A.Binary, A.Unary, A.Num, A.Var, A.Str]

NAMES, GROUPS = [], []


def _add(group, name):
    NAMES.append(name)
    GROUPS.append(group)


_add("tok", "log_ntok")
_add("tok", "log_nlines")
for g in _KIND_GROUPS:
    _add("tok", f"tok_{g}")
_add("tok", "brace_balance")
_add("tok", "paren_balance")
_add("tok", "ends_with_closer")
_add("tok", "semi_per_stmt")
for t in _NODE_TYPES:
    _add("ast", f"ast_{t.__name__}")
_add("ast", "ast_log_nodes")
_add("ast", "ast_depth")
_add("ast", "n_optional_inputs")
_add("ast", "n_phi_inputs")
_add("ast", "n_ranged_inputs")
_add("ast", "n_sinks")
_add("ast", "n_anonymize")
_add("ast", "n_present")
_add("ast", "n_zero_divisors")
for c in CODE_LIST:
    _add("diag", f"diag_{c}")
_add("diag", "n_errors")
_add("diag", "n_warnings")
_add("diag", "sem_ran")
for f in FAMILIES[1:]:
    _add("diag", f"first_fam_{f}")
for p in PHASES:
    _add("diag", f"first_phase_{p}")
_add("diag", "n_families")
_add("diag", "worst_family_rank")
for n in ("high_gt", "high_lt", "low_gt", "low_lt", "alert_rules", "alert_mismatch"):
    _add("pat", f"pat_{n}")

N_FEATURES = len(NAMES)
GROUP_IDX = {g: np.array([i for i, x in enumerate(GROUPS) if x == g]) for g in set(GROUPS)}
_WORD = re.compile(r"[a-z]+")
_PRIORITY = ["syntax", "lexical", "phi", "range", "unit", "missing", "type", "name", "logic"]


def _depth(n, d=0):
    if isinstance(n, list):
        return max([_depth(x, d) for x in n] or [d])
    if not hasattr(n, "__dataclass_fields__"):
        return d
    best = d + 1
    for f in n.__dataclass_fields__:
        v = getattr(n, f)
        if isinstance(v, (A.Node, list)):
            best = max(best, _depth(v, d + 1))
    return best


def _pattern_counts(toks):
    """Count comparison direction x alert vocabulary near each comparison."""
    c = dict(high_gt=0, high_lt=0, low_gt=0, low_lt=0)
    n = len(toks)
    for i, t in enumerate(toks):
        if t.kind not in ("<", ">", "<=", ">="):
            continue
        words = set()
        j = i - 1
        while j >= max(0, i - 6) and toks[j].kind not in ("if", "let"):
            if toks[j].kind == "ID":
                words |= set(_WORD.findall(toks[j].text.lower()))
            j -= 1
        j = i + 1
        while j < min(n, i + 16) and toks[j].kind not in ("if", "let", "EOF"):
            if toks[j].kind in ("STR", "ID"):
                words |= set(_WORD.findall(toks[j].text.lower().replace("_", " ")))
            j += 1
        hi, lo = bool(words & HIGH_WORDS), bool(words & LOW_WORDS)
        if hi == lo:
            continue
        gt = t.kind in (">", ">=")
        key = ("high" if hi else "low") + ("_gt" if gt else "_lt")
        c[key] += 1
    return c


def extract(res) -> np.ndarray:
    v = dict.fromkeys(NAMES, 0.0)
    toks = res.tokens
    v["log_ntok"] = math.log1p(len(toks))
    v["log_nlines"] = math.log1p(res.source.count("\n") + 1)
    kinds = {}
    for t in toks:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
    for g, ks in _KIND_GROUPS.items():
        v[f"tok_{g}"] = math.log1p(sum(kinds.get(k, 0) for k in ks))
    v["brace_balance"] = kinds.get("{", 0) - kinds.get("}", 0)
    v["paren_balance"] = kinds.get("(", 0) - kinds.get(")", 0)
    body = [t for t in toks if t.kind != "EOF"]
    v["ends_with_closer"] = 1.0 if body and body[-1].kind in ("}", ";") else 0.0

    prog = res.ast
    counts = {t: 0 for t in _NODE_TYPES}
    n_nodes = 0
    for nd in A.walk(prog):
        n_nodes += 1
        for t in _NODE_TYPES:
            if type(nd) is t:
                counts[t] += 1
        if isinstance(nd, A.InputDecl):
            v["n_optional_inputs"] += nd.ty.nullable
            v["n_phi_inputs"] += nd.phi
            v["n_ranged_inputs"] += nd.rng is not None
        elif isinstance(nd, A.Call):
            if nd.name in ("print", "export"):
                v["n_sinks"] += 1
            elif nd.name == "anonymize":
                v["n_anonymize"] += 1
            elif nd.name == "present":
                v["n_present"] += 1
        elif isinstance(nd, A.Binary) and nd.op == "/" and isinstance(nd.right, A.Num) \
                and nd.right.value == 0:
            v["n_zero_divisors"] += 1
    for t in _NODE_TYPES:
        v[f"ast_{t.__name__}"] = math.log1p(counts[t])
    v["ast_log_nodes"] = math.log1p(n_nodes)
    v["ast_depth"] = _depth(prog)
    stmts = counts[A.Let] + counts[A.Assign] + counts[A.Return] + counts[A.ExprStmt] \
        + counts[A.InputDecl] + counts[A.ConstDecl]
    v["semi_per_stmt"] = kinds.get(";", 0) / max(1, stmts)

    errs = [d for d in res.diagnostics if d.is_error]
    for d in res.diagnostics:
        v[f"diag_{d.code}"] += 1
    v["n_errors"] = len(errs)
    v["n_warnings"] = len(res.diagnostics) - len(errs)
    v["sem_ran"] = 1.0 if res.sem_ran else 0.0
    if res.diagnostics:
        first = errs[0] if errs else res.diagnostics[0]
        if first.family != "valid":
            v[f"first_fam_{first.family}"] = 1.0
        v[f"first_phase_{first.phase}"] = 1.0
    fams = {d.family for d in res.diagnostics if d.family != "valid" and d.is_error}
    v["n_families"] = len(fams)
    v["worst_family_rank"] = min((_PRIORITY.index(f) for f in fams), default=len(_PRIORITY)) / len(_PRIORITY)

    p = _pattern_counts(toks)
    for k, c in p.items():
        v[f"pat_{k}"] = c
    v["pat_alert_rules"] = sum(p.values())
    v["pat_alert_mismatch"] = p["high_lt"] + p["low_gt"]
    return np.array([v[n] for n in NAMES], dtype=np.float32)
