"""Hand-written scanner for MedScript.

The scanner is a single-pass longest-match recogniser.  Lexical errors are
recorded and scanning continues (illegal characters are dropped, malformed
numbers and unterminated strings are repaired), so the parser still receives a
usable token stream."""

import re
from dataclasses import dataclass

from .diagnostics import Diagnostic

KEYWORDS = {
    "input", "const", "func", "let", "if", "else", "while", "return",
    "true", "false", "int", "float", "bool", "string", "phi", "range",
}
UNITS = {"kg", "lb", "gram", "mg", "ml", "bpm", "mmHg", "degC", "degF", "metre", "hour"}

_ID = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_NUMBLOB = re.compile(r"[0-9][A-Za-z_0-9.]*")
_NUMOK = re.compile(r"[0-9]+(\.[0-9]+)?$")
_STR = re.compile(r'"[^"\n]*"')
_OPS2 = ("->", "==", "!=", "<=", ">=", "&&", "||")
_OPS1 = set("+-*/%<>=!(){},;:?^")


@dataclass(frozen=True)
class Token:
    kind: str    # 'ID' 'NUM' 'STR' 'UNIT' 'EOF' or the keyword / operator text
    text: str
    line: int
    col: int


def tokenize(src: str):
    """Return (tokens, diagnostics).  The token list always ends with EOF."""
    toks, diags = [], []
    i, line, line_start = 0, 1, 0
    n = len(src)
    while i < n:
        ch = src[i]
        col = i - line_start + 1
        if ch == "\n":
            line += 1
            i += 1
            line_start = i
            continue
        if ch in " \t\r":
            i += 1
            continue
        if src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
            continue
        if ch.isdigit():
            m = _NUMBLOB.match(src, i)
            blob = m.group()
            if _NUMOK.match(blob):
                toks.append(Token("NUM", blob, line, col))
            else:
                diags.append(Diagnostic("LEX002", f"malformed number '{blob}'", line, col))
                lead = re.match(r"[0-9]+(\.[0-9]+)?", blob)
                toks.append(Token("NUM", lead.group(), line, col))
            i = m.end()
            continue
        if ch.isalpha() or ch == "_":
            m = _ID.match(src, i)
            word = m.group()
            if word in KEYWORDS:
                toks.append(Token(word, word, line, col))
            elif word in UNITS:
                toks.append(Token("UNIT", word, line, col))
            else:
                toks.append(Token("ID", word, line, col))
            i = m.end()
            continue
        if ch == '"':
            m = _STR.match(src, i)
            if m:
                toks.append(Token("STR", m.group()[1:-1], line, col))
                i = m.end()
            else:
                j = src.find("\n", i)
                j = n if j < 0 else j
                diags.append(Diagnostic("LEX003", "unterminated string literal", line, col))
                toks.append(Token("STR", src[i + 1:j], line, col))
                i = j
            continue
        two = src[i:i + 2]
        if two in _OPS2:
            toks.append(Token(two, two, line, col))
            i += 2
            continue
        if ch in _OPS1:
            toks.append(Token(ch, ch, line, col))
            i += 1
            continue
        diags.append(Diagnostic("LEX001", f"illegal character {ch!r}", line, col))
        i += 1
    toks.append(Token("EOF", "", line, i - line_start + 1))
    return toks, diags
