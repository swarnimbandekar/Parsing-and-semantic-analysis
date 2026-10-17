"""Recursive-descent parser with panic-mode / insertion error recovery.

Grammar (EBNF)::

    program   = { decl } ;
    decl      = inputDecl | constDecl | funcDecl | stmt ;
    inputDecl = "input" ID ":" type [ "phi" ] [ "range" "(" num "," num ")" ] ";" ;
    constDecl = "const" ID ":" type "=" expr ";" ;
    funcDecl  = "func" ID "(" [ param { "," param } ] ")" "->" type block ;
    type      = ( "int" | "float" | "bool" | "string" | unitType ) [ "?" ] ;
    unitType  = unitTerm { ( "*" | "/" ) unitTerm } ;
    unitTerm  = UNIT [ "^" NUM ] ;
    stmt      = letStmt | ifStmt | whileStmt | returnStmt | assignOrCall | block ;
    expr      = or ;  (usual precedence: || && == != < > <= >= + - * / % unary)
"""

from . import ast_nodes as A
from .ast_nodes import Ty, dims_mul
from .diagnostics import Diagnostic

_STMT_START = {"let", "if", "while", "return", "const", "func", "input", "{", "}"}


class _Resync(Exception):
    pass


class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.p = 0
        self.diags = []
        self._last_err_pos = -1

    # ------------------------------------------------------------- utilities
    @property
    def tok(self):
        return self.toks[self.p]

    def peek(self, k=1):
        return self.toks[min(self.p + k, len(self.toks) - 1)]

    def advance(self):
        t = self.tok
        if t.kind != "EOF":
            self.p += 1
        return t

    def at(self, *kinds):
        return self.tok.kind in kinds

    def error(self, code, msg, tok=None):
        tok = tok or self.tok
        if self._last_err_pos == self.p and code != "SYN003":
            return                      # one report per position
        self._last_err_pos = self.p
        self.diags.append(Diagnostic(code, msg, tok.line, tok.col))

    def expect(self, kind, what=None):
        """Match `kind`; on mismatch report and *insert* the token virtually."""
        if self.tok.kind == kind:
            return self.advance()
        what = what or f"'{kind}'"
        if self.tok.kind == "EOF":
            self.error("SYN003", f"unexpected end of file, expected {what}")
        else:
            prev = self.toks[self.p - 1] if self.p else self.tok
            anchor = prev if kind in (";", ")", "}") else self.tok
            self.error("SYN001", f"expected {what} but found '{self.tok.text or self.tok.kind}'", anchor)
        return None

    def sync(self):
        """Panic mode: skip to a statement boundary."""
        while not self.at("EOF"):
            if self.at(";"):
                self.advance()
                return
            if self.tok.kind in _STMT_START:
                return
            self.advance()

    # ------------------------------------------------------------- top level
    def parse_program(self):
        prog = A.Program(1, 1)
        while not self.at("EOF"):
            start = self.p
            try:
                if self.at("}"):
                    self.error("SYN002", "unmatched '}'")
                    self.advance()
                    continue
                prog.items.append(self.parse_decl())
            except _Resync:
                self.sync()
            if self.p == start:
                self.advance()
        return prog

    def parse_decl(self):
        if self.at("input"):
            return self.parse_input()
        if self.at("const"):
            return self.parse_const()
        if self.at("func"):
            return self.parse_func()
        return self.parse_stmt()

    def parse_input(self):
        t = self.advance()
        name = self.expect("ID", "identifier")
        self.expect(":")
        ty = self.parse_type()
        phi, rng = False, None
        if self.at("phi"):
            self.advance()
            phi = True
        if self.at("range"):
            self.advance()
            self.expect("(")
            lo = self.parse_signed_number()
            self.expect(",")
            hi = self.parse_signed_number()
            self.expect(")")
            rng = (lo, hi)
        self.expect(";")
        return A.InputDecl(t.line, t.col, name.text if name else "<error>", ty, phi, rng)

    def parse_signed_number(self):
        sign = 1.0
        if self.at("-"):
            self.advance()
            sign = -1.0
        t = self.expect("NUM", "number")
        return sign * float(t.text) if t else 0.0

    def parse_const(self):
        t = self.advance()
        name = self.expect("ID", "identifier")
        self.expect(":")
        ty = self.parse_type()
        self.expect("=")
        init = self.parse_expr()
        self.expect(";")
        return A.ConstDecl(t.line, t.col, name.text if name else "<error>", ty, init)

    def parse_func(self):
        t = self.advance()
        name = self.expect("ID", "function name")
        self.expect("(")
        params = []
        if not self.at(")"):
            while True:
                pn = self.expect("ID", "parameter name")
                self.expect(":")
                pt = self.parse_type()
                params.append(A.Param(t.line, t.col, pn.text if pn else "<error>", pt))
                if self.at(","):
                    self.advance()
                    continue
                break
        self.expect(")")
        self.expect("->")
        ret = self.parse_type()
        body = self.parse_block()
        return A.FuncDecl(t.line, t.col, name.text if name else "<error>", params, ret, body)

    # ----------------------------------------------------------------- types
    def parse_type(self):
        if self.at("int", "float", "bool", "string"):
            base = self.advance().kind
            ty = Ty(base)
        elif self.at("UNIT"):
            dims = self.parse_unit_expr()
            ty = Ty("float", dims)
        else:
            self.error("SYN001", f"expected a type but found '{self.tok.text or self.tok.kind}'")
            raise _Resync()
        if self.at("?"):
            self.advance()
            ty = Ty(ty.base, ty.dims, True)
        return ty

    def parse_unit_expr(self):
        dims = ()
        sign = 1
        while True:
            u = self.expect("UNIT", "unit")
            e = 1
            if self.at("^"):
                self.advance()
                n = self.expect("NUM", "exponent")
                e = int(float(n.text)) if n else 1
            if u:
                dims = dims_mul(dims, ((u.text, e),), sign)
            if self.at("*") and self.peek().kind == "UNIT":
                self.advance()
                sign = 1
                continue
            if self.at("/") and self.peek().kind == "UNIT":
                self.advance()
                sign = -1
                continue
            break
        return dims

    # ------------------------------------------------------------ statements
    def parse_block(self):
        t = self.tok
        if not self.at("{"):
            self.expect("{")
            return A.Block(t.line, t.col, [])
        self.advance()
        blk = A.Block(t.line, t.col)
        while not self.at("}", "EOF"):
            start = self.p
            try:
                blk.stmts.append(self.parse_stmt())
            except _Resync:
                self.sync()
            if self.p == start:
                self.advance()
        if self.at("}"):
            self.advance()
        else:
            self.error("SYN003", "unexpected end of file, expected '}' to close block")
        return blk

    def parse_stmt(self):
        t = self.tok
        k = t.kind
        if k == "let":
            return self.parse_let()
        if k == "if":
            return self.parse_if()
        if k == "while":
            self.advance()
            self.expect("(")
            cond = self.parse_expr()
            self.expect(")")
            return A.While(t.line, t.col, cond, self.parse_block())
        if k == "return":
            self.advance()
            val = None if self.at(";") else self.parse_expr()
            self.expect(";")
            return A.Return(t.line, t.col, val)
        if k == "{":
            return self.parse_block()
        if k == "ID" and self.peek().kind == "=":
            self.advance()
            self.advance()
            val = self.parse_expr()
            self.expect(";")
            return A.Assign(t.line, t.col, t.text, val)
        e = self.parse_expr()
        self.expect(";")
        return A.ExprStmt(t.line, t.col, e)

    def parse_let(self):
        t = self.advance()
        name = self.expect("ID", "variable name")
        ty = None
        if self.at(":"):
            self.advance()
            ty = self.parse_type()
        init = None
        if self.at("="):
            self.advance()
            init = self.parse_expr()
        self.expect(";")
        return A.Let(t.line, t.col, name.text if name else "<error>", ty, init)

    def parse_if(self):
        t = self.advance()
        self.expect("(")
        cond = self.parse_expr()
        self.expect(")")
        then = self.parse_block()
        els = None
        if self.at("else"):
            self.advance()
            els = self.parse_if() if self.at("if") else self.parse_block()
        return A.If(t.line, t.col, cond, then, els)

    # ----------------------------------------------------------- expressions
    _LEVELS = [("||",), ("&&",), ("==", "!="), ("<", ">", "<=", ">="), ("+", "-"), ("*", "/", "%")]

    def parse_expr(self, level=0):
        if level == len(self._LEVELS):
            return self.parse_unary()
        left = self.parse_expr(level + 1)
        while self.tok.kind in self._LEVELS[level]:
            op = self.advance()
            right = self.parse_expr(level + 1)
            left = A.Binary(op.line, op.col, op.kind, left, right)
        return left

    def parse_unary(self):
        if self.at("-", "!"):
            op = self.advance()
            return A.Unary(op.line, op.col, op.kind, self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        t = self.tok
        if t.kind == "NUM":
            self.advance()
            dims = ()
            if self.at("UNIT"):
                dims = self.parse_unit_expr()
            return A.Num(t.line, t.col, t.text, dims)
        if t.kind == "STR":
            self.advance()
            return A.Str(t.line, t.col, t.text)
        if t.kind in ("true", "false"):
            self.advance()
            return A.BoolLit(t.line, t.col, t.kind == "true")
        if t.kind == "ID":
            self.advance()
            if self.at("("):
                self.advance()
                args = []
                if not self.at(")"):
                    while True:
                        args.append(self.parse_expr())
                        if self.at(","):
                            self.advance()
                            continue
                        break
                self.expect(")")
                return A.Call(t.line, t.col, t.text, args)
            return A.Var(t.line, t.col, t.text)
        if t.kind == "(":
            self.advance()
            e = self.parse_expr()
            self.expect(")")
            return e
        if t.kind == "EOF":
            self.error("SYN003", "unexpected end of file in expression")
        else:
            self.error("SYN002", f"unexpected token '{t.text or t.kind}' in expression")
        raise _Resync()


def parse(tokens):
    p = Parser(tokens)
    prog = p.parse_program()
    return prog, p.diags
