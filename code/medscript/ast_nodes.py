"""AST node definitions and the static type representation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Dims = Tuple[Tuple[str, int], ...]   # canonical unit exponents, e.g. (('kg', -1), ('mg', 1))


def dims_mul(a: Dims, b: Dims, sign: int = 1) -> Dims:
    d = dict(a)
    for u, e in b:
        d[u] = d.get(u, 0) + sign * e
    return tuple(sorted((u, e) for u, e in d.items() if e != 0))


def dims_str(d: Dims) -> str:
    if not d:
        return "1"
    num = [u if e == 1 else f"{u}^{e}" for u, e in d if e > 0]
    den = [u if e == -1 else f"{u}^{-e}" for u, e in d if e < 0]
    s = "*".join(num) if num else "1"
    if den:
        s += "/" + "/".join(den)
    return s


@dataclass(frozen=True)
class Ty:
    base: str                  # int float bool string void error
    dims: Dims = ()
    nullable: bool = False

    def __str__(self):
        s = self.base if not self.dims else dims_str(self.dims)
        return s + ("?" if self.nullable else "")

    @property
    def numeric(self):
        return self.base in ("int", "float")


T_ERR = Ty("error")
T_VOID = Ty("void")
T_BOOL = Ty("bool")
T_STR = Ty("string")
T_INT = Ty("int")
T_FLOAT = Ty("float")


# ----------------------------------------------------------------- expressions
@dataclass
class Node:
    line: int = 0
    col: int = 0


@dataclass
class Num(Node):
    text: str = "0"
    dims: Dims = ()

    @property
    def value(self):
        return float(self.text)

    @property
    def is_int(self):
        return "." not in self.text and not self.dims


@dataclass
class Str(Node):
    value: str = ""


@dataclass
class BoolLit(Node):
    value: bool = False


@dataclass
class Var(Node):
    name: str = ""


@dataclass
class Call(Node):
    name: str = ""
    args: list = field(default_factory=list)


@dataclass
class Unary(Node):
    op: str = ""
    operand: object = None


@dataclass
class Binary(Node):
    op: str = ""
    left: object = None
    right: object = None


# ------------------------------------------------------------------ statements
@dataclass
class Let(Node):
    name: str = ""
    ty: Optional[Ty] = None
    init: object = None


@dataclass
class Assign(Node):
    name: str = ""
    value: object = None


@dataclass
class If(Node):
    cond: object = None
    then: "Block" = None
    els: object = None      # Block | If | None


@dataclass
class While(Node):
    cond: object = None
    body: "Block" = None


@dataclass
class Return(Node):
    value: object = None


@dataclass
class ExprStmt(Node):
    expr: object = None


@dataclass
class Block(Node):
    stmts: List[object] = field(default_factory=list)


# ---------------------------------------------------------------- declarations
@dataclass
class InputDecl(Node):
    name: str = ""
    ty: Ty = None
    phi: bool = False
    rng: Optional[Tuple[float, float]] = None


@dataclass
class ConstDecl(Node):
    name: str = ""
    ty: Ty = None
    init: object = None


@dataclass
class Param(Node):
    name: str = ""
    ty: Ty = None


@dataclass
class FuncDecl(Node):
    name: str = ""
    params: List[Param] = field(default_factory=list)
    ret: Ty = None
    body: Block = None


@dataclass
class Program(Node):
    items: List[object] = field(default_factory=list)


def walk(node):
    """Pre-order traversal over all AST nodes."""
    yield node
    if isinstance(node, list):
        for x in node:
            yield from walk(x)
        return
    if not hasattr(node, "__dataclass_fields__"):
        return
    for f in node.__dataclass_fields__:
        v = getattr(node, f)
        if isinstance(v, (Node, list)):
            yield from walk(v)
