"""Fault-injection operators.

Each operator takes the program as a list of lines and returns a mutated
copy (or None when it is not applicable).  Operators are grouped by the
error family they are meant to create.  Whether the fault *manifests* (the
compiler reports a diagnostic of the intended family) is checked by the
corpus builder, not here."""

import random
import re

from .generator import HIGH_WORDS, LOW_WORDS, fmt

UNIT_RE = r"(?:degC|bpm|mmHg|kg|ml|mg)"
CMP_LIT = re.compile(r"([<>]=?|==|!=) (\d+(?:\.\d+)?) (" + UNIT_RE + r")\b(?![/^])")
INPUT_RE = re.compile(r"^input (\w+): ([\w/^]+)(\??)( phi)?(?: range\(([-\d.]+), ([-\d.]+)\))?;$")
LET_RE = re.compile(r"^let (\w+)(?:: ([\w/^?]+))?(?: = (.*))?;$")

SWAP_UNIT = {"degC": "mmHg", "kg": "lb", "bpm": "mmHg", "mmHg": "bpm", "ml": "mg", "mg": "ml"}
DECL_SWAP = {"kg": "lb", "degC": "degF", "mmHg": "bpm", "bpm": "mmHg", "ml": "mg"}


# --------------------------------------------------------------------- utils
def _in_string(line, pos):
    return line[:pos].count('"') % 2 == 1


def _inputs(lines):
    out = {}
    for i, l in enumerate(lines):
        m = INPUT_RE.match(l)
        if m:
            out[m.group(1)] = dict(idx=i, unit=m.group(2), opt=bool(m.group(3)),
                                   phi=bool(m.group(4)),
                                   lo=float(m.group(5)) if m.group(5) else None,
                                   hi=float(m.group(6)) if m.group(6) else None)
    return out


def _top_points(lines, start):
    """Indices at which a new top-level statement can be inserted."""
    pts, depth = [], 0
    for i, l in enumerate(lines):
        if i >= start and depth == 0:
            pts.append(i)
        depth += l.count("{") - l.count("}")
    if depth == 0:
        pts.append(len(lines))
    return pts


def _body_start(lines):
    """Index of the first top-level statement after inputs / consts / functions."""
    depth, last, in_func = 0, 0, False
    for i, l in enumerate(lines):
        t = l.strip()
        if depth == 0 and t.startswith("func "):
            in_func = True
        elif depth == 0 and not in_func:
            if t == "" or t.startswith(("//", "input ", "const ")):
                last = i + 1
            else:
                break
        depth += l.count("{") - l.count("}")
        if in_func and depth == 0:
            in_func = False
            last = i + 1
    return last


def _insert(lines, idx, new):
    return lines[:idx] + list(new) + lines[idx:]


def _code_idx(lines):
    return [i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("//")]


def _funcs(lines):
    """name -> (def_index, ret_type)."""
    out = {}
    for i, l in enumerate(lines):
        m = re.match(r"^func (\w+)\(.*\) -> ([\w/^?]+) \{$", l)
        if m:
            out[m.group(1)] = (i, m.group(2))
    return out


def _split_args(s):
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        if ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    args.append(cur.strip())
    return args


def _typo(word, rng):
    if len(word) < 3:
        return word + "x"
    i = rng.randrange(1, len(word) - 1)
    k = rng.randrange(3)
    if k == 0:
        return word[:i] + word[i + 1:]                          # deletion
    if k == 1 and word[i] != word[i + 1]:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]  # transposition
    return word[:i] + word[i] + word[i:]                        # duplication


# ================================================================== LEXICAL
def lex_illegal(lines, rng):
    idx = _code_idx(lines)
    rng.shuffle(idx)
    for i in idx:
        l = lines[i]
        sp = [p for p, ch in enumerate(l) if ch == " " and not _in_string(l, p) and p > 0]
        if sp:
            p = rng.choice(sp)
            ch = rng.choice(["@", "#", "$", "~", "&", "|", "`", "\\"])
            out = list(lines)
            out[i] = l[:p] + " " + ch + l[p:]
            return out
    return None


def lex_number(lines, rng):
    cands = []
    for i, l in enumerate(lines):
        if l.strip().startswith("//"):
            continue
        for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", l):
            if not _in_string(l, m.start()):
                cands.append((i, m))
    if not cands:
        return None
    i, m = rng.choice(cands)
    t = m.group()
    if "." in t:
        bad = rng.choice([t + ".5", t + "x"])
    else:
        bad = rng.choice([t + "x", t + "l", t + ".5.5", t.replace("0", "O", 1) if "0" in t else t + "o"])
    out = list(lines)
    out[i] = lines[i][:m.start()] + bad + lines[i][m.end():]
    return out


def lex_string(lines, rng):
    cands = [i for i, l in enumerate(lines) if re.search(r'print\("[^"]*"\);', l)]
    if not cands:
        return None
    i = rng.choice(cands)
    out = list(lines)
    out[i] = re.sub(r'(print\("[^"]*)"\);', r"\1);", lines[i], count=1)
    return out


# =================================================================== SYNTAX
def syn_semicolon(lines, rng):
    cands = [i for i, l in enumerate(lines) if l.rstrip().endswith(";") and not l.strip().startswith("//")]
    if not cands:
        return None
    i = rng.choice(cands)
    out = list(lines)
    out[i] = lines[i].rstrip()[:-1]
    return out


def syn_paren(lines, rng):
    cands = [i for i, l in enumerate(lines) if re.match(r"^\s*(if|while|\} else if) \(.*\) \{$", l)]
    if not cands:
        return None
    i = rng.choice(cands)
    l = lines[i]
    k = l.rfind(")")
    out = list(lines)
    out[i] = l[:k] + l[k + 1:]
    return out


def syn_close_brace(lines, rng):
    cands = [i for i, l in enumerate(lines) if l.strip() == "}"]
    if not cands:
        return None
    i = rng.choice(cands)
    return lines[:i] + lines[i + 1:]


def syn_open_brace(lines, rng):
    cands = [i for i, l in enumerate(lines) if l.rstrip().endswith(" {")]
    if not cands:
        return None
    i = rng.choice(cands)
    out = list(lines)
    out[i] = lines[i].rstrip()[:-2]
    return out


def syn_operand(lines, rng):
    cands = [(i, m) for i, l in enumerate(lines) for m in [re.search(r"([<>]=?) (\d+(?:\.\d+)?)( [\w/^]+)?(?=\))", l)]
             if m and l.strip().startswith(("if", "} else if", "while"))]
    if not cands:
        return None
    i, m = rng.choice(cands)
    out = list(lines)
    out[i] = lines[i][:m.start()] + m.group(1) + lines[i][m.end():]
    return out


def syn_keyword(lines, rng):
    cands = [i for i, l in enumerate(lines) if re.match(r"^\s*(if|let|return|while) ", l)]
    if not cands:
        return None
    i = rng.choice(cands)
    kw = re.match(r"^(\s*)(if|let|return|while)", lines[i])
    bad = {"if": "iff", "let": "lett", "return": "retrun", "while": "whlie"}[kw.group(2)]
    out = list(lines)
    out[i] = kw.group(1) + bad + lines[i][kw.end():]
    return out


# ====================================================================== NAME
def name_typo(lines, rng):
    decls = {}
    for i, l in enumerate(lines):
        m = LET_RE.match(l)
        if m:
            decls[m.group(1)] = i
    for k, v in _inputs(lines).items():
        decls[k] = v["idx"]
    names = list(decls)
    rng.shuffle(names)
    for n in names:
        occ = [(i, m) for i, l in enumerate(lines) if i != decls[n] and not l.strip().startswith("//")
               for m in re.finditer(r"\b" + re.escape(n) + r"\b", l) if not _in_string(l, m.start())]
        if occ:
            i, m = rng.choice(occ)
            t = _typo(n, rng)
            if t == n:
                continue
            out = list(lines)
            out[i] = lines[i][:m.start()] + t + lines[i][m.end():]
            return out
    return None


def name_delete_decl(lines, rng):
    cands = []
    for i, l in enumerate(lines):
        m = LET_RE.match(l) or INPUT_RE.match(l)
        if m:
            n = m.group(1)
            rest = "\n".join(x for j, x in enumerate(lines) if j != i)
            if re.search(r"\b" + re.escape(n) + r"\b", rest):
                cands.append(i)
    if not cands:
        return None
    i = rng.choice(cands)
    return lines[:i] + lines[i + 1:]


def name_redecl(lines, rng):
    cands = [i for i, l in enumerate(lines) if LET_RE.match(l) and l.startswith("let ")]
    if not cands:
        return None
    i = rng.choice(cands)
    return lines[:i + 1] + [lines[i]] + lines[i + 1:]


def name_uninit(lines, rng):
    cands = []
    for i, l in enumerate(lines):
        m = LET_RE.match(l)
        if m and m.group(2) and m.group(3):
            rest = "\n".join(lines[i + 1:])
            if re.search(r"\b" + re.escape(m.group(1)) + r"\b", rest):
                cands.append((i, m))
    if not cands:
        return None
    i, m = rng.choice(cands)
    out = list(lines)
    out[i] = f"let {m.group(1)}: {m.group(2)};"
    return out


def name_func(lines, rng):
    fs = _funcs(lines)
    cands = [(i, n) for i, l in enumerate(lines) if not l.startswith("func ")
             for n in fs if re.search(r"\b" + n + r"\(", l)]
    if not cands:
        return None
    i, n = rng.choice(cands)
    out = list(lines)
    out[i] = re.sub(r"\b" + n + r"\(", _typo(n, rng) + "(", lines[i], count=1)
    return out


def name_assign_input(lines, rng):
    ins = {k: v for k, v in _inputs(lines).items()
           if v["lo"] is not None and v["unit"] in ("kg", "degC", "bpm", "mmHg", "ml", "lb")}
    if not ins:
        return None
    k = rng.choice(list(ins))
    v = ins[k]
    val = fmt(round((v["lo"] + v["hi"]) / 2, 1))
    pts = _top_points(lines, _body_start(lines))
    return _insert(lines, rng.choice(pts), [f"{k} = {val} {v['unit']};"])


# ====================================================================== TYPE
def type_string_assign(lines, rng):
    cands = []
    for i, l in enumerate(lines):
        m = re.match(r"^let (\w+): ([\w/^]+) = .*;$", l)
        if m and m.group(2) not in ("string", "bool"):
            cands.append((i, m))
    if not cands:
        return None
    i, m = rng.choice(cands)
    bad = rng.choice(['"n/a"', "true"])
    out = list(lines)
    out[i] = f"let {m.group(1)}: {m.group(2)} = {bad};"
    return out


def type_bool_assign(lines, rng):
    # insert a typed boolean flag and assign a number to it
    pts = _top_points(lines, _body_start(lines))
    v = f"flag{rng.randrange(10)}"
    ins = [f"let {v}: bool = false;", f"{v} = {rng.choice([1, 0, 2])};", f"print({v});"]
    return _insert(lines, rng.choice(pts), ins)


def type_arity(lines, rng):
    fs = _funcs(lines)
    cands = []
    for i, l in enumerate(lines):
        if l.startswith("func "):
            continue
        for n in fs:
            m = re.search(r"\b" + n + r"\(", l)
            if m:
                cands.append((i, n, m))
    if not cands:
        return None
    i, n, m = rng.choice(cands)
    l = lines[i]
    s = m.end()
    depth, e = 1, s
    while e < len(l) and depth:
        depth += (l[e] == "(") - (l[e] == ")")
        e += 1
    args = _split_args(l[s:e - 1])
    if len(args) >= 2 and rng.random() < 0.7:
        args = args[:-1]
    else:
        args = args + ["1"]
    out = list(lines)
    out[i] = l[:s] + ", ".join(args) + l[e - 1:]
    return out


def type_cond(lines, rng):
    cands = []
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*)if \((\w+) ([<>]=?) .*\) \{$", l)
        if m and "&&" not in l:
            cands.append((i, m))
    if not cands:
        return None
    i, m = rng.choice(cands)
    out = list(lines)
    out[i] = f"{m.group(1)}if ({m.group(2)}) {{"
    return out


def type_return(lines, rng):
    fs = _funcs(lines)
    cands, cur = [], None
    for i, l in enumerate(lines):
        m = re.match(r"^func \w+\(.*\) -> ([\w/^?]+) \{$", l)
        if m:
            cur = m.group(1)
        elif l == "}":
            cur = None
        elif cur and cur != "bool" and re.match(r"^\s+return .+;$", l):
            cands.append(i)
    if not cands:
        return None
    i = rng.choice(cands)
    ind = re.match(r"^(\s*)", lines[i]).group(1)
    out = list(lines)
    out[i] = f"{ind}return {rng.choice(['true', chr(34) + 'ok' + chr(34)])};"
    return out


def type_float_int(lines, rng):
    pts = _top_points(lines, _body_start(lines))
    v = f"count{rng.randrange(10)}"
    ins = [f"let {v}: int = {rng.choice(['2.5', '3.75', '0.5'])};", f"print({v});"]
    return _insert(lines, rng.choice(pts), ins)


# ====================================================================== UNIT
def unit_swap_literal(lines, rng):
    cands = [(i, m) for i, l in enumerate(lines) if not l.strip().startswith("//")
             for m in CMP_LIT.finditer(l) if m.group(3) in SWAP_UNIT and not _in_string(l, m.start())]
    if not cands:
        return None
    i, m = rng.choice(cands)
    out = list(lines)
    new = f"{m.group(1)} {m.group(2)} {SWAP_UNIT[m.group(3)]}"
    out[i] = lines[i][:m.start()] + new + lines[i][m.end():]
    return out


def unit_drop(lines, rng):
    cands = [(i, m) for i, l in enumerate(lines) if not l.strip().startswith("//")
             for m in CMP_LIT.finditer(l) if float(m.group(2)) != 0]
    if not cands:
        return None
    i, m = rng.choice(cands)
    out = list(lines)
    out[i] = lines[i][:m.start()] + f"{m.group(1)} {m.group(2)}" + lines[i][m.end():]
    return out


def unit_decl(lines, rng):
    ins = {k: v for k, v in _inputs(lines).items() if v["unit"] in DECL_SWAP}
    if not ins:
        return None
    k = rng.choice(list(ins))
    i = ins[k]["idx"]
    out = list(lines)
    out[i] = lines[i].replace(f": {ins[k]['unit']}", f": {DECL_SWAP[ins[k]['unit']]}", 1)
    return out


def unit_rate(lines, rng):
    cands = [(i, m) for i, l in enumerate(lines) for m in [re.search(r"(\d+(?:\.\d+)?) mg/kg", l)]
             if m and not l.startswith("func ")]
    if not cands:
        return None
    i, m = rng.choice(cands)
    out = list(lines)
    out[i] = lines[i][:m.start()] + f"{m.group(1)} " + rng.choice(["mg/ml", "mg", "mg/lb"]) + lines[i][m.end():]
    return out


def unit_return(lines, rng):
    cands = [i for i, l in enumerate(lines) if l.strip() == "return w / (h * h);"]
    if not cands:
        return None
    i = cands[0]
    out = list(lines)
    out[i] = "    return w / h;"
    return out


def unit_add(lines, rng):
    ins = {k: v for k, v in _inputs(lines).items()
           if v["unit"] in SWAP_UNIT and not v["opt"] and v["lo"] is not None}
    if not ins:
        return None
    k = rng.choice(list(ins))
    u = ins[k]["unit"]
    pts = _top_points(lines, _body_start(lines))
    v = f"adj{rng.randrange(100)}"
    new = [f"let {v}: {u} = {k} + {rng.choice([1, 2, 5])} {SWAP_UNIT[u]};", f"print({v});"]
    return _insert(lines, rng.choice(pts), new)


# ===================================================================== RANGE
def range_literal(lines, rng):
    cands = [(i, m) for i, l in enumerate(lines) if not l.strip().startswith("//")
             for m in CMP_LIT.finditer(l) if not _in_string(l, m.start())]
    if not cands:
        return None
    i, m = rng.choice(cands)
    t = m.group(2)
    if "." in t and rng.random() < 0.6:
        bad = t.replace(".", "", 1)                    # 38.0 -> 380 (dropped decimal point)
    else:
        bad = str(int(float(t)) * rng.choice([10, 100]))  # 100 -> 1000 (extra zero)
    out = list(lines)
    out[i] = lines[i][:m.start()] + f"{m.group(1)} {bad} {m.group(3)}" + lines[i][m.end():]
    return out


def range_threshold(lines, rng):
    ins = _inputs(lines)
    cands = []
    for i, l in enumerate(lines):
        for m in re.finditer(r"\b(\w+) ([<>]=?) (\d+(?:\.\d+)?) (" + UNIT_RE + r")\b(?![/^])", l):
            if m.group(1) in ins and ins[m.group(1)]["lo"] is not None:
                cands.append((i, m, ins[m.group(1)]))
    if not cands:
        return None
    i, m, d = rng.choice(cands)
    span = d["hi"] - d["lo"]
    new = d["hi"] + max(1.0, 0.1 * span) if rng.random() < 0.6 else d["lo"] - max(0.5, 0.05 * span)
    if new < 0:
        new = d["hi"] + max(1.0, 0.1 * span)
    out = list(lines)
    out[i] = lines[i][:m.start(3)] + fmt(round(new, 1)) + lines[i][m.end(3):]
    return out


def range_decl(lines, rng):
    cands = [i for i, l in enumerate(lines) if INPUT_RE.match(l) and "range(" in l]
    if not cands:
        return None
    i = rng.choice(cands)
    m = INPUT_RE.match(lines[i])
    out = list(lines)
    out[i] = re.sub(r"range\(([-\d.]+), ([-\d.]+)\)", r"range(\2, \1)", lines[i])
    return out


def range_const(lines, rng):
    cands = [i for i, l in enumerate(lines) if re.match(r"^const \w+: mg = \d+ mg;$", l)]
    if not cands:
        return None
    i = rng.choice(cands)
    out = list(lines)
    out[i] = re.sub(r"= (\d+) mg", lambda m: f"= {int(m.group(1)) * 100} mg", lines[i])
    return out


# ======================================================================= PHI
def _ensure_phi(lines, rng):
    ins = _inputs(lines)
    phis = [k for k, v in ins.items() if v["phi"]]
    if phis:
        return lines, rng.choice(phis)
    n = rng.choice(["patient_name", "pname", "full_name", "mrn"])
    idxs = [v["idx"] for v in ins.values()]
    pos = (max(idxs) + 1) if idxs else 0
    return _insert(lines, pos, [f"input {n}: string phi;"]), n


def phi_direct(lines, rng):
    lines, n = _ensure_phi(lines, rng)
    pts = _top_points(lines, _body_start(lines))
    sink = rng.choice(["print", "export"])
    return _insert(lines, rng.choice(pts), [f"{sink}({n});"])


def phi_alias(lines, rng):
    lines, n = _ensure_phi(lines, rng)
    pts = _top_points(lines, _body_start(lines))
    a = f"label{rng.randrange(10)}"
    return _insert(lines, rng.choice(pts), [f"let {a} = {n};", f"export({a});"])


def phi_concat(lines, rng):
    lines, n = _ensure_phi(lines, rng)
    pts = _top_points(lines, _body_start(lines))
    return _insert(lines, rng.choice(pts), [f'print("patient: " + {n});'])


def phi_removeanon(lines, rng):
    cands = [i for i, l in enumerate(lines) if "anonymize(" in l]
    if not cands:
        return None
    i = rng.choice(cands)
    out = list(lines)
    out[i] = re.sub(r"anonymize\((\w+)\)", r"\1", lines[i], count=1)
    return out


def phi_func(lines, rng):
    lines, n = _ensure_phi(lines, rng)
    fn = rng.choice(["format_note", "make_label", "build_msg"])
    bs = _body_start(lines)
    f = ["", f"func {fn}(s: string) -> string {{", '    return "note: " + s;', "}"]
    lines = _insert(lines, bs, f)
    pts = _top_points(lines, bs + len(f))
    return _insert(lines, rng.choice(pts), [f"print({fn}({n}));"])


# ================================================================== MISSING
def miss_unguard(lines, rng):
    cands = []
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*)if \(present\((\w+)\)\) \{$", l)
        if m:
            cands.append((i, "bare", m))
        m = re.match(r"^(\s*)if \(present\((\w+)\) && (.*)\) \{$", l)
        if m:
            cands.append((i, "and", m))
        m = re.match(r"^(\s*)if \(!present\((\w+)\)\) \{$", l)
        if m:
            cands.append((i, "neg", m))
    if not cands:
        return None
    i, kind, m = rng.choice(cands)
    out = list(lines)
    if kind == "bare":
        out[i] = f"{m.group(1)}{{"
    elif kind == "and":
        out[i] = f"{m.group(1)}if ({m.group(3)}) {{"
    else:
        out[i] = f"{m.group(1)}if (false) {{"
    return out


def miss_new_use(lines, rng):
    ins = _inputs(lines)
    opts = [k for k, v in ins.items() if v["opt"]]
    if opts:
        k = rng.choice(opts)
        lines2 = lines
    else:
        k = rng.choice(["creatinine", "lactate", "bilirubin"])
        lines2 = _insert(lines, max([v["idx"] for v in ins.values()] + [-1]) + 1,
                         [f"input {k}: mg/ml? range(0, 15);"])
    pts = _top_points(lines2, _body_start(lines2))
    v = f"scaled{rng.randrange(10)}"
    return _insert(lines2, rng.choice(pts), [f"let {v} = {k} * {rng.choice([2, 3])};", f"print({v});"])


def miss_make_optional(lines, rng):
    ins = {k: v for k, v in _inputs(lines).items() if not v["opt"] and not v["phi"] and v["lo"] is not None}
    if not ins:
        return None
    k = rng.choice(list(ins))
    i = ins[k]["idx"]
    out = list(lines)
    out[i] = lines[i].replace(f": {ins[k]['unit']} ", f": {ins[k]['unit']}? ", 1)
    return out


# ==================================================================== LOGIC
def logic_div_zero(lines, rng):
    cands = [i for i, l in enumerate(lines) if re.match(r"^let \w+: ml/hour = \w+ / \w+;$", l)]
    if cands and rng.random() < 0.6:
        i = rng.choice(cands)
        out = list(lines)
        out[i] = re.sub(r"/ \w+;$", "/ 0 hour;", lines[i])
        return out
    ins = {k: v for k, v in _inputs(lines).items() if not v["opt"] and not v["phi"] and v["lo"] is not None}
    if not ins:
        return None
    k = rng.choice(list(ins))
    pts = _top_points(lines, _body_start(lines))
    v = f"ratio{rng.randrange(10)}"
    return _insert(lines, rng.choice(pts), [f"let {v} = {k} / 0;", f"print({v});"])


def logic_range_zero(lines, rng):
    ins = _inputs(lines)
    cands = []
    for i, l in enumerate(lines):
        for m in re.finditer(r"/ (\w+)\b", l):
            n = m.group(1)
            if n in ins and ins[n]["lo"] is not None and ins[n]["lo"] > 0:
                cands.append(n)
    if not cands:
        return None
    n = rng.choice(cands)
    i = ins[n]["idx"]
    out = list(lines)
    out[i] = re.sub(r"range\(([-\d.]+),", "range(0,", lines[i], count=1)
    return out


def logic_missing_return(lines, rng):
    cands = []
    for i, l in enumerate(lines):
        if l == "}" and i > 0 and re.match(r"^    return .+;$", lines[i - 1]):
            j = i - 1
            while j >= 0 and not lines[j].startswith("func "):
                j -= 1
            if j >= 0 and not lines[j].endswith("-> void {"):
                cands.append(i - 1)
    if not cands:
        return None
    i = rng.choice(cands)
    return lines[:i] + lines[i + 1:]


def logic_unreachable(lines, rng):
    cands = [i for i, l in enumerate(lines)
             if re.match(r"^    return .*;$", l) and i + 1 < len(lines) and lines[i + 1] == "}"]
    if not cands:
        return None
    i = rng.choice(cands)
    return _insert(lines, i + 1, ['    print("done");'])


def _alert_rules(lines):
    """Indices of `if (var OP lit) {` lines whose next line prints an alert word."""
    out = []
    for i, l in enumerate(lines[:-1]):
        m = re.match(r"^(\s*)if \((.*?) ([<>]=?) (.*)\) \{$", l)
        if m and "&&" not in l:
            nxt = lines[i + 1]
            w = re.search(r'print\("ALERT (\w+)"\)', nxt)
            if w and (w.group(1) in HIGH_WORDS or w.group(1) in LOW_WORDS):
                out.append((i, m, w.group(1)))
    return out


def logic_inverted(lines, rng):
    cands = _alert_rules(lines)
    if not cands:
        return None
    i, m, word = rng.choice(cands)
    flip = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
    out = list(lines)
    out[i] = f"{m.group(1)}if ({m.group(2)} {flip[m.group(3)]} {m.group(4)}) {{"
    return out


def logic_wrong_message(lines, rng):
    cands = _alert_rules(lines)
    if not cands:
        return None
    i, m, word = rng.choice(cands)
    pool = LOW_WORDS if word in HIGH_WORDS else HIGH_WORDS
    other = rng.choice(sorted(pool))
    out = list(lines)
    out[i + 1] = lines[i + 1].replace(word, other)
    return out


def logic_constcond(lines, rng):
    cands = [i for i, l in enumerate(lines) if re.match(r"^\s*if \(.*\) \{$", l)
             and "present(" not in l]
    if not cands:
        return None
    i = rng.choice(cands)
    ind = re.match(r"^(\s*)", lines[i]).group(1)
    out = list(lines)
    out[i] = f"{ind}if ({rng.choice(['1 > 2', '2 > 1', 'false', 'true'])}) {{"
    return out


OPERATORS = {
    "lexical": [lex_illegal, lex_number, lex_string],
    "syntax": [syn_semicolon, syn_paren, syn_close_brace, syn_open_brace, syn_operand, syn_keyword],
    "name": [name_typo, name_delete_decl, name_redecl, name_uninit, name_func, name_assign_input],
    "type": [type_string_assign, type_bool_assign, type_arity, type_cond, type_return, type_float_int],
    "unit": [unit_swap_literal, unit_drop, unit_decl, unit_rate, unit_return, unit_add],
    "range": [range_literal, range_threshold, range_decl, range_const],
    "phi": [phi_direct, phi_alias, phi_concat, phi_removeanon, phi_func],
    "missing": [miss_unguard, miss_new_use, miss_make_optional],
    "logic": [logic_div_zero, logic_range_zero, logic_missing_return, logic_unreachable,
              logic_inverted, logic_wrong_message, logic_constcond],
}

# Root-cause priority used to label multi-fault programs (higher wins).
PRIORITY = ["syntax", "lexical", "phi", "range", "unit", "missing", "type", "name", "logic"]
