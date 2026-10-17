"""Three-address-code generation, a small optimiser and a reference VM.

Instruction forms (tuples)::

    ('copy', dst, a)            ('bin', dst, op, a, b)      ('un', dst, op, a)
    ('label', L)                ('jmp', L)                  ('jz', a, L)
    ('param', a)                ('call', dst, fn, nargs)    ('ret', a | None)

Operands are variable names (str) or constants wrapped in :class:`K`.
"""

from dataclasses import dataclass

from . import ast_nodes as A


@dataclass(frozen=True)
class K:
    v: object

    def __repr__(self):
        return repr(self.v)


# =============================================================== lowering
class Lowerer:
    def __init__(self):
        self.ntemp = 0
        self.nlabel = 0
        self.nvar = 0
        self.code = []
        self.scopes = [{}]

    def temp(self):
        self.ntemp += 1
        return f"t{self.ntemp}"

    def label(self):
        self.nlabel += 1
        return f"L{self.nlabel}"

    def emit(self, *ins):
        self.code.append(ins)

    def lookup(self, name):
        for sc in reversed(self.scopes):
            if name in sc:
                return sc[name]
        return name        # inputs / consts keep their source name

    def declare(self, name):
        self.nvar += 1
        u = f"{name}@{self.nvar}"
        self.scopes[-1][name] = u
        return u

    # ---- expressions
    def expr(self, e):
        if isinstance(e, A.Num):
            return K(e.value if not e.is_int else int(e.value))
        if isinstance(e, A.Str):
            return K(e.value)
        if isinstance(e, A.BoolLit):
            return K(e.value)
        if isinstance(e, A.Var):
            return self.lookup(e.name)
        if isinstance(e, A.Unary):
            a = self.expr(e.operand)
            t = self.temp()
            self.emit("un", t, e.op, a)
            return t
        if isinstance(e, A.Binary):
            if e.op in ("&&", "||"):
                return self.short_circuit(e)
            a = self.expr(e.left)
            b = self.expr(e.right)
            t = self.temp()
            self.emit("bin", t, e.op, a, b)
            return t
        if isinstance(e, A.Call):
            args = [self.expr(x) for x in e.args]
            for a in args:
                self.emit("param", a)
            t = self.temp()
            self.emit("call", t, e.name, len(args))
            return t
        raise TypeError(e)

    def short_circuit(self, e):
        res = self.temp()
        end = self.label()
        a = self.expr(e.left)
        self.emit("copy", res, a)
        if e.op == "&&":
            self.emit("jz", res, end)
        else:
            nt = self.temp()
            self.emit("un", nt, "!", res)
            self.emit("jz", nt, end)
        b = self.expr(e.right)
        self.emit("copy", res, b)
        self.emit("label", end)
        return res

    # ---- statements
    def stmt(self, s):
        if isinstance(s, A.Let):
            v = self.expr(s.init) if s.init is not None else None
            u = self.declare(s.name)
            if v is not None:
                self.emit("copy", u, v)
        elif isinstance(s, A.Assign):
            v = self.expr(s.value)
            self.emit("copy", self.lookup(s.name), v)
        elif isinstance(s, A.ExprStmt):
            self.expr(s.expr)
        elif isinstance(s, A.Return):
            self.emit("ret", self.expr(s.value) if s.value is not None else None)
        elif isinstance(s, A.Block):
            self.block(s)
        elif isinstance(s, A.If):
            self.if_(s)
        elif isinstance(s, A.While):
            top, end = self.label(), self.label()
            self.emit("label", top)
            c = self.expr(s.cond)
            self.emit("jz", c, end)
            self.block(s.body)
            self.emit("jmp", top)
            self.emit("label", end)

    def if_(self, s):
        els, end = self.label(), self.label()
        c = self.expr(s.cond)
        self.emit("jz", c, els)
        self.block(s.then)
        if s.els is not None:
            self.emit("jmp", end)
        self.emit("label", els)
        if s.els is not None:
            if isinstance(s.els, A.Block):
                self.block(s.els)
            else:
                self.if_(s.els)
            self.emit("label", end)

    def block(self, b):
        self.scopes.append({})
        for s in b.stmts:
            self.stmt(s)
        self.scopes.pop()


def lower(prog: A.Program):
    """Return {'main': code, 'funcs': {name: (params, code)}}."""
    funcs = {}
    lw = Lowerer()
    for it in prog.items:
        if isinstance(it, A.FuncDecl):
            fl = Lowerer()
            fl.ntemp, fl.nlabel = 0, 0
            params = []
            for p in it.params:
                params.append(fl.declare(p.name))
            fl.block(it.body)
            fl.emit("ret", None)
            funcs[it.name] = (params, fl.code)
        elif isinstance(it, A.ConstDecl):
            lw.emit("copy", it.name, lw.expr(it.init))
        elif isinstance(it, A.InputDecl):
            continue
        else:
            lw.stmt(it)
    return {"main": lw.code, "funcs": funcs}


# ============================================================== optimiser
_FOLD = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "<": lambda a, b: a < b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


def _fold_bin(op, a, b):
    if op == "/":
        return None if b == 0 else a / b
    if op == "%":
        return None if b == 0 else a % b
    if op == "+" and (isinstance(a, str) != isinstance(b, str)):
        return None
    return _FOLD[op](a, b)


def _is_leader_break(ins):
    return ins[0] in ("label", "jmp", "jz", "ret", "call")


def fold_and_propagate(code):
    """Constant folding, constant/copy propagation, local CSE and algebraic
    simplification inside basic blocks."""
    out = []
    env = {}        # var -> K | var (copy)
    avail = {}      # (op, a, b) -> temp holding value

    def val(x):
        return env.get(x, x) if isinstance(x, str) else x

    def kill(v):
        env.pop(v, None)
        for k in [k for k, c in env.items() if c == v]:
            env.pop(k)
        for k in [k for k, t in avail.items() if v in (k[1], k[2], t)]:
            avail.pop(k)

    for ins in code:
        op = ins[0]
        if op == "label":
            env.clear()
            avail.clear()
            out.append(ins)
        elif op == "copy":
            v = val(ins[2])
            kill(ins[1])
            if v != ins[1]:
                env[ins[1]] = v
            out.append(("copy", ins[1], v))
        elif op == "un":
            a = val(ins[3])
            kill(ins[1])
            if isinstance(a, K):
                r = (not a.v) if ins[2] == "!" else -a.v
                env[ins[1]] = K(r)
                out.append(("copy", ins[1], K(r)))
            else:
                out.append(("un", ins[1], ins[2], a))
        elif op == "bin":
            _, d, o, a, b = ins
            a, b = val(a), val(b)
            kill(d)
            if isinstance(a, K) and isinstance(b, K):
                r = _fold_bin(o, a.v, b.v)
                if r is not None:
                    env[d] = K(r)
                    out.append(("copy", d, K(r)))
                    continue
            simp = None
            if o in ("+", "-") and isinstance(b, K) and b.v == 0 and not isinstance(b.v, bool):
                simp = a
            elif o == "+" and isinstance(a, K) and a.v == 0 and not isinstance(a.v, bool):
                simp = b
            elif o in ("*", "/") and isinstance(b, K) and b.v == 1 and not isinstance(b.v, bool):
                simp = a
            elif o == "*" and isinstance(a, K) and a.v == 1 and not isinstance(a.v, bool):
                simp = b
            if simp is not None:
                env[d] = simp
                out.append(("copy", d, simp))
                continue
            key = (o, a, b)
            if key in avail and d != avail[key]:
                env[d] = avail[key]
                out.append(("copy", d, avail[key]))
                continue
            if d != a and d != b:
                avail[key] = d
            out.append(("bin", d, o, a, b))
        elif op == "jz":
            c = val(ins[1])
            if isinstance(c, K):
                if not c.v:
                    out.append(("jmp", ins[2]))
                # else: condition always true -> fall through (drop)
            else:
                out.append(("jz", c, ins[2]))
        elif op == "param":
            out.append(("param", val(ins[1])))
        elif op == "call":
            kill(ins[1])
            env.clear()
            avail.clear()
            out.append(ins)
        elif op == "ret":
            out.append(("ret", val(ins[1]) if ins[1] is not None else None))
            env.clear()
            avail.clear()
        else:
            out.append(ins)
    return out


def remove_unreachable(code):
    out, dead = [], False
    for ins in code:
        if ins[0] == "label":
            dead = False
        if not dead:
            out.append(ins)
        if ins[0] in ("jmp", "ret"):
            dead = True
    targets = {i[-1] for i in out if i[0] in ("jmp", "jz")}
    return [i for i in out if i[0] != "label" or i[1] in targets]


def _reads(ins):
    op = ins[0]
    if op == "copy":
        return [ins[2]]
    if op == "bin":
        return [ins[3], ins[4]]
    if op == "un":
        return [ins[3]]
    if op in ("jz", "param"):
        return [ins[1]]
    if op == "ret":
        return [ins[1]] if ins[1] is not None else []
    return []


def dead_stores(code, keep=()):
    """Remove pure assignments whose destination is never read."""
    changed = True
    while changed:
        changed = False
        used = {x for i in code for x in _reads(i) if isinstance(x, str)}
        new = []
        for i in code:
            if i[0] in ("copy", "bin", "un") and isinstance(i[1], str) \
                    and i[1] not in used and i[1] not in keep \
                    and (i[1].startswith("t") or "@" in i[1]):
                changed = True
                continue
            new.append(i)
        code = new
    return code


def fuse_temp_copies(code):
    """t = a op b ; v = t   ==>   v = a op b   (when t is single-use)."""
    uses = {}
    for i in code:
        for x in _reads(i):
            if isinstance(x, str):
                uses[x] = uses.get(x, 0) + 1
    out, k = [], 0
    while k < len(code):
        i = code[k]
        nxt = code[k + 1] if k + 1 < len(code) else None
        if i[0] in ("bin", "un") and nxt and nxt[0] == "copy" and nxt[2] == i[1] \
                and i[1].startswith("t") and uses.get(i[1], 0) == 1:
            out.append((i[0], nxt[1]) + i[2:])
            k += 2
            continue
        out.append(i)
        k += 1
    return out


def optimise_code(code, keep=()):
    for _ in range(4):
        before = len(code)
        code = fold_and_propagate(code)
        code = remove_unreachable(code)
        code = dead_stores(code, keep)
        code = fuse_temp_copies(code)
        if len(code) == before:
            break
    return code


def optimise(ir):
    keep = {i[1] for i in ir["main"] if i[0] == "copy" and not i[1].startswith("t")
            and "@" not in i[1]}
    main = optimise_code(ir["main"], keep)
    funcs = {n: (p, optimise_code(c)) for n, (p, c) in ir["funcs"].items()}
    return {"main": main, "funcs": funcs}


def count_instructions(ir):
    return len(ir["main"]) + sum(len(c) for _, c in ir["funcs"].values())


def format_ir(ir):
    lines = []

    def fmt(code):
        for i in code:
            op = i[0]
            if op == "label":
                lines.append(f"{i[1]}:")
            elif op == "copy":
                lines.append(f"    {i[1]} = {i[2]}")
            elif op == "bin":
                lines.append(f"    {i[1]} = {i[3]} {i[2]} {i[4]}")
            elif op == "un":
                lines.append(f"    {i[1]} = {i[2]}{i[3]}")
            elif op == "jmp":
                lines.append(f"    goto {i[1]}")
            elif op == "jz":
                lines.append(f"    ifFalse {i[1]} goto {i[2]}")
            elif op == "param":
                lines.append(f"    param {i[1]}")
            elif op == "call":
                lines.append(f"    {i[1]} = call {i[2]}, {i[3]}")
            elif op == "ret":
                lines.append(f"    return {i[1] if i[1] is not None else ''}".rstrip())
    for n, (params, code) in ir["funcs"].items():
        lines.append(f"func {n}({', '.join(params)}):")
        fmt(code)
    lines.append("main:")
    fmt(ir["main"])
    return "\n".join(lines)


# ==================================================================== VM
class VMError(Exception):
    pass


def run(ir, inputs, max_steps=200000):
    """Execute IR; returns the list of (sink, value) output events."""
    events = []
    glob = dict(inputs)
    steps = [0]

    def builtin(name, a):
        if name in ("print", "export"):
            events.append((name, a[0] if len(a) == 1 else tuple(a)))
            return None
        if name == "present":
            return a[0] is not None
        if name == "anonymize":
            return "ANON"
        if name == "abs":
            return abs(a[0])
        if name == "min":
            return min(a)
        if name == "max":
            return max(a)
        if name == "lb_to_kg":
            return a[0] * 0.45359237
        if name == "degF_to_degC":
            return (a[0] - 32) * 5.0 / 9.0
        if name == "g_to_mg":
            return a[0] * 1000.0
        raise VMError(f"unknown builtin {name}")

    def exec_code(code, local, is_main):
        labels = {i[1]: k for k, i in enumerate(code) if i[0] == "label"}
        stack = []
        pc = 0

        def get(x):
            if isinstance(x, K):
                return x.v
            if x in local:
                return local[x]
            if x in glob:
                return glob[x]
            raise VMError(f"unbound {x}")

        def put(x, v):
            if is_main and "@" not in x and not x.startswith("t"):
                glob[x] = v
            else:
                local[x] = v

        while pc < len(code):
            steps[0] += 1
            if steps[0] > max_steps:
                raise VMError("step limit")
            ins = code[pc]
            op = ins[0]
            pc += 1
            if op == "copy":
                put(ins[1], get(ins[2]))
            elif op == "bin":
                a, b = get(ins[3]), get(ins[4])
                o = ins[2]
                if o == "/":
                    r = a / b
                elif o == "%":
                    r = a % b
                else:
                    r = _FOLD[o](a, b)
                put(ins[1], r)
            elif op == "un":
                a = get(ins[3])
                put(ins[1], (not a) if ins[2] == "!" else -a)
            elif op == "jmp":
                pc = labels[ins[1]]
            elif op == "jz":
                if not get(ins[1]):
                    pc = labels[ins[2]]
            elif op == "param":
                stack.append(get(ins[1]))
            elif op == "call":
                n = ins[3]
                args = stack[len(stack) - n:] if n else []
                del stack[len(stack) - n:]
                name = ins[2]
                if name in ir["funcs"]:
                    params, fcode = ir["funcs"][name]
                    put(ins[1], exec_code(fcode, dict(zip(params, args)), False))
                else:
                    put(ins[1], builtin(name, args))
            elif op == "ret":
                return get(ins[1]) if ins[1] is not None else None
        return None

    exec_code(ir["main"], {}, True)
    return events
