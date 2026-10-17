"""Semantic analysis for MedScript.

Checks performed
  * name resolution, redeclaration, definite assignment
  * base-type checking with int -> float widening
  * unit-of-measure checking (dimension vectors over base units)
  * physiological range / threshold plausibility (interval reasoning)
  * PHI taint tracking from `phi` inputs to the print / export sinks
  * missing-value (nullable) checking with flow-sensitive refinement
  * division by zero (constant and interval based), missing return,
    unreachable code and constant conditions
"""

from dataclasses import dataclass
from typing import Optional

from . import ast_nodes as A
from .ast_nodes import (Ty, T_ERR, T_VOID, T_BOOL, T_STR, T_INT, T_FLOAT,
                        dims_mul, dims_str)
from .diagnostics import Diagnostic

# Plausible magnitudes for literals carrying a single physical unit.
PLAUSIBLE = {
    "kg": (0.2, 500.0), "lb": (0.4, 1100.0), "degC": (20.0, 45.0),
    "degF": (68.0, 113.0), "bpm": (0.0, 300.0), "mmHg": (0.0, 350.0),
    "metre": (0.2, 2.5), "ml": (0.0, 10000.0), "mg": (0.0, 20000.0),
    "gram": (0.0, 20.0), "hour": (0.0, 72.0),
}

KG = (("kg", 1),)
LB = (("lb", 1),)
MG = (("mg", 1),)
G = (("gram", 1),)
DEGC = (("degC", 1),)
DEGF = (("degF", 1),)


@dataclass
class Info:
    ty: Ty
    taint: bool = False
    const: object = None
    rng: Optional[tuple] = None
    sym: object = None


@dataclass
class Symbol:
    name: str
    ty: Ty
    kind: str                 # input const var param
    line: int
    phi: bool = False
    rng: Optional[tuple] = None
    const: object = None
    used: bool = False


@dataclass
class FuncSig:
    name: str
    params: list
    ret: Ty


class Flow:
    """Flow-sensitive facts: definitely-assigned symbols and refinements."""

    def __init__(self, assigned=None, facts=None):
        self.assigned = set(assigned or ())
        self.facts = set(facts or ())

    def copy(self):
        return Flow(self.assigned, self.facts)

    def with_facts(self, facts):
        f = self.copy()
        f.facts |= facts
        return f


BUILTIN_SINKS = {"print", "export"}
BUILTINS = {"present", "anonymize", "abs", "min", "max", "lb_to_kg",
            "degF_to_degC", "g_to_mg"} | BUILTIN_SINKS


class SemanticAnalyzer:
    def __init__(self):
        self.diags = []
        self.globals = {}
        self.main = {}
        self.scopes = []
        self.funcs = {}
        self.cur_func: Optional[FuncSig] = None

    # ------------------------------------------------------------ reporting
    def err(self, code, msg, node):
        self.diags.append(Diagnostic(code, msg, node.line, node.col))

    # --------------------------------------------------------------- scopes
    def lookup(self, name):
        for sc in reversed(self.scopes):
            if name in sc:
                return sc[name]
        return None

    def declare(self, sym, node):
        sc = self.scopes[-1]
        if sym.name in sc:
            self.err("SEM002", f"redeclaration of '{sym.name}'", node)
            return sc[sym.name]
        sc[sym.name] = sym
        return sym

    def push(self):
        self.scopes.append({})

    def pop(self):
        sc = self.scopes.pop()
        for s in sc.values():
            if s.kind == "var" and not s.used:
                self.diags.append(Diagnostic("W002", f"variable '{s.name}' is never used", s.line, 0))

    # ----------------------------------------------------------- entry point
    def analyze(self, prog: A.Program):
        for it in prog.items:
            if isinstance(it, A.FuncDecl):
                if it.name in self.funcs:
                    self.err("SEM002", f"redefinition of function '{it.name}'", it)
                else:
                    self.funcs[it.name] = FuncSig(it.name, [p.ty for p in it.params], it.ret)
        flow = Flow()
        self.scopes = [self.globals, self.main]
        for it in prog.items:
            if isinstance(it, A.InputDecl):
                self.check_input(it, flow)
            elif isinstance(it, A.ConstDecl):
                self.check_const(it, flow)
            elif isinstance(it, A.FuncDecl):
                self.check_func(it)
            else:
                self.scopes = [self.globals, self.main]
                self.check_stmt(it, flow)
        for s in self.main.values():
            if s.kind == "var" and not s.used:
                self.diags.append(Diagnostic("W002", f"variable '{s.name}' is never used", s.line, 0))
        return self.diags

    # --------------------------------------------------------- declarations
    def check_input(self, d, flow):
        sym = Symbol(d.name, d.ty, "input", d.line, phi=d.phi, rng=d.rng)
        if d.rng and d.rng[0] > d.rng[1]:
            self.err("DOM001", f"empty range [{d.rng[0]}, {d.rng[1]}] for '{d.name}'", d)
        self.scopes = [self.globals]
        self.declare(sym, d)
        flow.assigned.add(id(sym))
        self.scopes = [self.globals, self.main]

    def check_const(self, d, flow):
        self.scopes = [self.globals]
        info = self.expr(d.init, flow)
        self.compat(d.ty, info, d.init, "const initialiser")
        sym = Symbol(d.name, d.ty, "const", d.line, phi=info.taint, const=info.const,
                     rng=info.rng)
        self.declare(sym, d)
        flow.assigned.add(id(sym))
        self.scopes = [self.globals, self.main]

    def check_func(self, f):
        sig = self.funcs.get(f.name)
        saved = self.scopes, self.cur_func
        self.scopes = [self.globals, {}]
        self.cur_func = FuncSig(f.name, [p.ty for p in f.params], f.ret)
        flow = Flow()
        for p in f.params:
            s = self.declare(Symbol(p.name, p.ty, "param", p.line), p)
            flow.assigned.add(id(s))
        term = self.block_noscope(f.body, flow)
        if f.ret.base != "void" and not term:
            self.err("SEM007", f"function '{f.name}' may finish without returning a value", f)
        self.pop()
        self.scopes, self.cur_func = saved

    # ----------------------------------------------------------- statements
    def block(self, blk, flow):
        self.push()
        term = self.block_noscope(blk, flow)
        self.pop()
        return term

    def block_noscope(self, blk, flow):
        term = False
        warned = False
        for st in blk.stmts:
            if term and not warned:
                self.err("W001", "unreachable code", st)
                warned = True
            t = self.check_stmt(st, flow)
            term = term or t
        return term

    def check_stmt(self, st, flow):
        if isinstance(st, A.Let):
            return self.s_let(st, flow)
        if isinstance(st, A.Assign):
            return self.s_assign(st, flow)
        if isinstance(st, A.If):
            return self.s_if(st, flow)
        if isinstance(st, A.While):
            return self.s_while(st, flow)
        if isinstance(st, A.Return):
            return self.s_return(st, flow)
        if isinstance(st, A.Block):
            return self.block(st, flow)
        if isinstance(st, A.ExprStmt):
            self.expr(st.expr, flow)
            return False
        if isinstance(st, (A.InputDecl, A.ConstDecl, A.FuncDecl)):
            self.err("SYN002", "declaration not allowed here", st)
        return False

    def s_let(self, st, flow):
        info = self.expr(st.init, flow) if st.init is not None else None
        if st.ty is None:
            if info is None:
                self.err("SEM004", f"cannot infer type of '{st.name}'", st)
                ty = T_ERR
            else:
                ty = Ty(info.ty.base, info.ty.dims) if info.ty is not T_ERR else T_ERR
        else:
            ty = st.ty
            if info is not None:
                self.compat(ty, info, st.init, f"initialiser of '{st.name}'")
        sym = Symbol(st.name, ty, "var", st.line, phi=bool(info and info.taint))
        sym = self.declare(sym, st)
        if info is not None:
            flow.assigned.add(id(sym))
        return False

    def s_assign(self, st, flow):
        info = self.expr(st.value, flow)
        sym = self.lookup(st.name)
        if sym is None:
            self.err("SEM001", f"assignment to undeclared variable '{st.name}'", st)
            return False
        if sym.kind in ("input", "const"):
            self.err("SEM009", f"cannot assign to {sym.kind} '{st.name}'", st)
            return False
        self.compat(sym.ty, info, st.value, f"assignment to '{st.name}'")
        flow.assigned.add(id(sym))
        sym.phi = info.taint
        return False

    def s_while(self, st, flow):
        c = self.cond(st.cond, flow, "while")
        body = flow.with_facts(self.refine(st.cond, True))
        self.block(st.body, body)
        return False

    def s_return(self, st, flow):
        if self.cur_func is None:
            if st.value is not None:
                self.expr(st.value, flow)
            return True
        ret = self.cur_func.ret
        if st.value is None:
            if ret.base != "void":
                self.err("SEM004", "return without a value in non-void function", st)
        else:
            info = self.expr(st.value, flow)
            if ret.base == "void":
                self.err("SEM004", "void function returns a value", st)
            else:
                self.compat(ret, info, st.value, "return value")
        return True

    def s_if(self, st, flow):
        self.cond(st.cond, flow, "if")
        pos = self.refine(st.cond, True)
        neg = self.refine(st.cond, False)
        tf = flow.with_facts(pos)
        t_term = self.block(st.then, tf)
        ef = flow.with_facts(neg)
        e_term = False
        if st.els is not None:
            e_term = (self.block(st.els, ef) if isinstance(st.els, A.Block)
                      else self.s_if(st.els, ef))
        if t_term and e_term:
            return True
        if t_term:
            flow.assigned, flow.facts = ef.assigned, ef.facts
        elif e_term:
            flow.assigned, flow.facts = tf.assigned, tf.facts
        else:
            flow.assigned = tf.assigned & ef.assigned
            flow.facts = tf.facts & ef.facts
        return False

    def cond(self, e, flow, what):
        info = self.expr(e, flow)
        if info.ty is not T_ERR and info.ty.base != "bool":
            self.err("SEM008", f"{what} condition has type {info.ty}, expected bool", e)
        elif isinstance(info.const, bool):
            self.err("W003", f"{what} condition is always {str(info.const).lower()}", e)
        return info

    # ---------------------------------------------------------- refinements
    def refine(self, e, positive):
        """Facts that hold when `e` evaluates to `positive`."""
        if isinstance(e, A.Call) and e.name == "present" and len(e.args) == 1 \
                and isinstance(e.args[0], A.Var):
            s = self.lookup(e.args[0].name)
            if s is not None and positive:
                return {("present", id(s))}
            return set()
        if isinstance(e, A.Unary) and e.op == "!":
            return self.refine(e.operand, not positive)
        if isinstance(e, A.Binary):
            if e.op == "&&":
                return self.refine(e.left, True) | self.refine(e.right, True) if positive else set()
            if e.op == "||":
                return set() if positive else self.refine(e.left, False) | self.refine(e.right, False)
            if e.op in ("<", ">", "<=", ">=", "==", "!="):
                return self._nonzero_fact(e, positive)
        return set()

    _NEG = {"<": ">=", ">": "<=", "<=": ">", ">=": "<", "==": "!=", "!=": "=="}
    _FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}

    def _nonzero_fact(self, e, positive):
        op, a, b = e.op, e.left, e.right
        if isinstance(a, A.Num) and isinstance(b, A.Var):
            a, b, op = b, a, self._FLIP[op]
        if not (isinstance(a, A.Var) and isinstance(b, A.Num)):
            return set()
        s = self.lookup(a.name)
        if s is None:
            return set()
        if not positive:
            op = self._NEG[op]
        c = b.value
        ok = ((op == ">" and c >= 0) or (op == ">=" and c > 0) or (op == "!=" and c == 0)
              or (op == "<" and c <= 0) or (op == "<=" and c < 0) or (op == "==" and c != 0))
        return {("nonzero", id(s))} if ok else set()

    # ---------------------------------------------------------- compatibility
    def compat(self, declared: Ty, info: Info, node, what):
        a = info.ty
        if declared is T_ERR or a is T_ERR:
            return True
        ok_base = declared.base == a.base or (declared.base == "float" and a.base == "int")
        if not ok_base:
            self.err("SEM004", f"type mismatch in {what}: expected {declared}, found {a}", node)
            return False
        if declared.numeric and declared.dims != a.dims:
            if isinstance(node, A.Num) and node.value == 0 and not node.dims:
                return True
            self.err("SEM005", f"unit mismatch in {what}: expected [{dims_str(declared.dims)}], "
                               f"found [{dims_str(a.dims)}]", node)
            return False
        return True

    # ---------------------------------------------------------- expressions
    def expr(self, e, flow) -> Info:
        if isinstance(e, A.Num):
            return self.e_num(e)
        if isinstance(e, A.Str):
            return Info(T_STR, const=e.value)
        if isinstance(e, A.BoolLit):
            return Info(T_BOOL, const=e.value)
        if isinstance(e, A.Var):
            return self.e_var(e, flow)
        if isinstance(e, A.Call):
            return self.e_call(e, flow)
        if isinstance(e, A.Unary):
            return self.e_unary(e, flow)
        if isinstance(e, A.Binary):
            return self.e_binary(e, flow)
        return Info(T_ERR)

    def e_num(self, e):
        if e.dims:
            if len(e.dims) == 1 and e.dims[0][1] == 1:
                u = e.dims[0][0]
                lo, hi = PLAUSIBLE.get(u, (None, None))
                if lo is not None and not (lo <= e.value <= hi):
                    self.err("DOM001", f"value {e.text} {u} is outside the plausible range "
                                       f"[{lo}, {hi}] {u}", e)
            ty = Ty("float", e.dims)
        else:
            ty = T_INT if e.is_int else T_FLOAT
        return Info(ty, const=e.value, rng=(e.value, e.value))

    def e_var(self, e, flow):
        s = self.lookup(e.name)
        if s is None:
            self.err("SEM001", f"undeclared identifier '{e.name}'", e)
            return Info(T_ERR)
        s.used = True
        if id(s) not in flow.assigned and s.kind in ("var",):
            self.err("SEM003", f"'{e.name}' may be used before it is initialised", e)
            flow.assigned.add(id(s))
        ty = s.ty
        if ty.nullable and ("present", id(s)) not in flow.facts:
            self.err("DOM003", f"optional value '{e.name}' used without a presence check", e)
            ty = Ty(ty.base, ty.dims)
        elif ty.nullable:
            ty = Ty(ty.base, ty.dims)
        rng = s.rng if s.rng and s.rng[0] <= s.rng[1] else (None if s.const is None or isinstance(s.const, (bool, str))
                                    else (s.const, s.const))
        return Info(ty, taint=s.phi, const=s.const, rng=rng, sym=s)

    def e_unary(self, e, flow):
        a = self.expr(e.operand, flow)
        if a.ty is T_ERR:
            return Info(T_ERR)
        if e.op == "!":
            if a.ty.base != "bool":
                self.err("SEM004", f"operator '!' needs bool, found {a.ty}", e)
                return Info(T_ERR)
            return Info(T_BOOL, const=None if a.const is None else (not a.const))
        if not a.ty.numeric:
            self.err("SEM004", f"unary '-' needs a number, found {a.ty}", e)
            return Info(T_ERR)
        rng = None if a.rng is None else (-a.rng[1], -a.rng[0])
        return Info(a.ty, a.taint, None if a.const is None else -a.const, rng)

    def e_binary(self, e, flow):
        op = e.op
        if op in ("&&", "||"):
            l = self.expr(e.left, flow)
            rf = flow.with_facts(self.refine(e.left, op == "&&"))
            r = self.expr(e.right, rf)
            for x, side in ((l, e.left), (r, e.right)):
                if x.ty is not T_ERR and x.ty.base != "bool":
                    self.err("SEM004", f"operator '{op}' needs bool operands, found {x.ty}", side)
                    return Info(T_BOOL)
            c = None
            if isinstance(l.const, bool) and isinstance(r.const, bool):
                c = (l.const and r.const) if op == "&&" else (l.const or r.const)
            return Info(T_BOOL, const=c)
        l = self.expr(e.left, flow)
        r = self.expr(e.right, flow)
        taint = l.taint or r.taint
        if l.ty is T_ERR or r.ty is T_ERR:
            return Info(T_ERR, taint)
        if op == "+" and l.ty.base == "string" and r.ty.base == "string":
            return Info(T_STR, taint)
        if not (l.ty.numeric and r.ty.numeric):
            if op in ("==", "!=") and l.ty.base == r.ty.base:
                return Info(T_BOOL)
            self.err("SEM004", f"operator '{op}' not defined for {l.ty} and {r.ty}", e)
            return Info(T_ERR, taint)
        same = l.ty.dims == r.ty.dims
        zero_ok = (isinstance(e.left, A.Num) and e.left.value == 0 and not e.left.dims) or \
                  (isinstance(e.right, A.Num) and e.right.value == 0 and not e.right.dims)
        floaty = "float" if "float" in (l.ty.base, r.ty.base) else "int"
        if op in ("+", "-", "<", ">", "<=", ">=", "==", "!="):
            if not same and not zero_ok:
                self.err("SEM005", f"unit mismatch: [{dims_str(l.ty.dims)}] {op} "
                                   f"[{dims_str(r.ty.dims)}]", e)
                return Info(T_BOOL if op not in ("+", "-") else T_ERR, taint)
            dims = l.ty.dims or r.ty.dims
            if op in ("<", ">", "<=", ">=", "==", "!="):
                self._check_threshold(e, l, r)
                return Info(T_BOOL, const=self._fold_cmp(op, l, r))
            ty = Ty(floaty if not dims else "float", dims)
            return Info(ty, taint, self._fold(op, l.const, r.const), self._interval(op, l.rng, r.rng))
        if op == "*":
            dims = dims_mul(l.ty.dims, r.ty.dims)
            ty = Ty("float" if dims else floaty, dims)
            return Info(ty, taint, self._fold(op, l.const, r.const), self._interval(op, l.rng, r.rng))
        if op in ("/", "%"):
            self._check_divisor(e, r, flow)
            if op == "%":
                if l.ty.base != "int" or r.ty.base != "int":
                    self.err("SEM004", "operator '%' needs int operands", e)
                    return Info(T_ERR, taint)
                return Info(T_INT, taint, self._fold(op, l.const, r.const))
            dims = dims_mul(l.ty.dims, r.ty.dims, -1)
            ty = Ty("float", dims)
            return Info(ty, taint, self._fold(op, l.const, r.const), self._interval(op, l.rng, r.rng))
        return Info(T_ERR, taint)

    def _check_divisor(self, e, r, flow):
        if r.const is not None and not isinstance(r.const, (bool, str)) and r.const == 0:
            self.err("DOM004", "division by constant zero", e)
            return
        if r.rng and r.rng[0] <= 0 <= r.rng[1] and r.const is None:
            s = r.sym
            if s is not None and ("nonzero", id(s)) in flow.facts:
                return
            if s is None and not isinstance(e.right, A.Var):
                # composite divisor: only flag when its interval reaches zero
                pass
            self.err("DOM005", "divisor may be zero (its range includes 0)", e)

    def _check_threshold(self, e, l, r):
        """Comparing a ranged variable with a constant outside its range."""
        for var, other in ((l, r), (r, l)):
            if var.sym is not None and var.sym.rng and other.const is not None \
                    and not isinstance(other.const, (bool, str)) and var.const is None:
                lo, hi = var.sym.rng
                if other.const < lo or other.const > hi:
                    self.err("DOM001", f"comparison against {other.const:g}, outside the declared "
                                       f"range [{lo:g}, {hi:g}] of '{var.sym.name}'", e)
                    return

    @staticmethod
    def _fold(op, a, b):
        if a is None or b is None or isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
            return None
        try:
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                return a / b
            if op == "%":
                return a % b
        except ZeroDivisionError:
            return None
        return None

    @staticmethod
    def _fold_cmp(op, l, r):
        a, b = l.const, r.const
        if a is None or b is None or isinstance(a, str) or isinstance(b, str):
            return None
        return {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b,
                "==": a == b, "!=": a != b}[op]

    @staticmethod
    def _interval(op, a, b):
        if a is None or b is None or a[0] > a[1] or b[0] > b[1]:
            return None
        if op == "+":
            return (a[0] + b[0], a[1] + b[1])
        if op == "-":
            return (a[0] - b[1], a[1] - b[0])
        if op == "*":
            p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
            return (min(p), max(p))
        if op == "/":
            if b[0] <= 0 <= b[1]:
                return None
            p = [a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1]]
            return (min(p), max(p))
        return None

    # ---------------------------------------------------------------- calls
    def e_call(self, e, flow):
        name = e.name
        if name == "present":
            if len(e.args) != 1:
                self.err("SEM006", "present() takes exactly one argument", e)
            elif isinstance(e.args[0], A.Var):
                s = self.lookup(e.args[0].name)
                if s is None:
                    self.err("SEM001", f"undeclared identifier '{e.args[0].name}'", e.args[0])
                else:
                    s.used = True
            else:
                self.expr(e.args[0], flow)
            return Info(T_BOOL)
        args = [self.expr(a, flow) for a in e.args]
        taint = any(a.taint for a in args)
        if name in BUILTIN_SINKS:
            if not args:
                self.err("SEM006", f"{name}() needs at least one argument", e)
            for a, node in zip(args, e.args):
                if a.taint:
                    self.err("DOM002", f"protected health information reaches {name}() "
                                       "without anonymize()", node)
            return Info(T_VOID)
        if name == "anonymize":
            if len(args) != 1 or (args[0].ty is not T_ERR and args[0].ty.base != "string"):
                self.err("SEM006", "anonymize() expects one string argument", e)
            return Info(T_STR, taint=False)
        if name in ("abs", "min", "max"):
            want = 1 if name == "abs" else 2
            if len(args) != want or any(a.ty is not T_ERR and not a.ty.numeric for a in args):
                self.err("SEM006", f"{name}() expects {want} numeric argument(s)", e)
                return Info(T_ERR)
            if want == 2 and args[0].ty.dims != args[1].ty.dims:
                self.err("SEM005", f"unit mismatch in {name}()", e)
            return Info(args[0].ty, taint)
        conv = {"lb_to_kg": (LB, KG), "degF_to_degC": (DEGF, DEGC), "g_to_mg": (G, MG)}
        if name in conv:
            src, dst = conv[name]
            if len(args) != 1:
                self.err("SEM006", f"{name}() expects one argument", e)
            elif args[0].ty is not T_ERR:
                self.compat(Ty("float", src), args[0], e.args[0], f"argument of {name}()")
            return Info(Ty("float", dst), taint)
        sig = self.funcs.get(name)
        if sig is None:
            self.err("SEM001", f"call to undeclared function '{name}'", e)
            return Info(T_ERR)
        if len(args) != len(sig.params):
            self.err("SEM006", f"'{name}' expects {len(sig.params)} argument(s), got {len(args)}", e)
            return Info(sig.ret, taint)
        for i, (pty, a, node) in enumerate(zip(sig.params, args, e.args), 1):
            self.compat(pty, a, node, f"argument {i} of '{name}'")
        return Info(sig.ret, taint)


def analyze(prog):
    sa = SemanticAnalyzer()
    return sa.analyze(prog), sa
