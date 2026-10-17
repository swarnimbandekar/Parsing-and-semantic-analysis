"""Diagnostic codes and the error-family taxonomy used by the classifier."""

from dataclasses import dataclass

# Error families = class labels of the learned classifier (index = label id).
FAMILIES = [
    "valid",     # 0
    "lexical",   # 1
    "syntax",    # 2
    "name",      # 3  undeclared / redeclared / uninitialised
    "type",      # 4  base-type, arity and condition errors
    "unit",      # 5  unit-of-measure mismatches
    "range",     # 6  physiologically implausible values / thresholds
    "phi",       # 7  protected health information reaching a sink
    "missing",   # 8  optional (missing) value used without a presence check
    "logic",     # 9  division by zero, missing return, inverted clinical rule
]

# code -> (family, severity, phase)
CODES = {
    "LEX001": ("lexical", "error", "lex"),     # illegal character
    "LEX002": ("lexical", "error", "lex"),     # malformed number
    "LEX003": ("lexical", "error", "lex"),     # unterminated string
    "SYN001": ("syntax", "error", "parse"),    # expected token not found
    "SYN002": ("syntax", "error", "parse"),    # unexpected token in expression
    "SYN003": ("syntax", "error", "parse"),    # unexpected end of file
    "SEM001": ("name", "error", "sem"),        # undeclared identifier / function
    "SEM002": ("name", "error", "sem"),        # redeclaration
    "SEM003": ("name", "error", "sem"),        # use before initialisation
    "SEM009": ("name", "error", "sem"),        # assignment to input / const
    "SEM004": ("type", "error", "sem"),        # type mismatch
    "SEM006": ("type", "error", "sem"),        # wrong argument count / type
    "SEM008": ("type", "error", "sem"),        # condition is not boolean
    "SEM005": ("unit", "error", "sem"),        # unit-of-measure mismatch
    "DOM001": ("range", "error", "sem"),       # value / threshold out of range
    "DOM002": ("phi", "error", "sem"),         # PHI reaches an output sink
    "DOM003": ("missing", "error", "sem"),     # optional value used unguarded
    "SEM007": ("logic", "error", "sem"),       # missing return on some path
    "DOM004": ("logic", "error", "sem"),       # division by constant zero
    "DOM005": ("logic", "error", "sem"),       # possible division by zero
    "W001": ("logic", "warning", "sem"),       # unreachable code
    "W002": ("valid", "warning", "sem"),       # unused variable
    "W003": ("logic", "warning", "sem"),       # constant condition
}

CODE_LIST = list(CODES)
PHASES = ["lex", "parse", "sem"]


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    line: int
    col: int = 0

    @property
    def family(self) -> str:
        return CODES[self.code][0]

    @property
    def severity(self) -> str:
        return CODES[self.code][1]

    @property
    def phase(self) -> str:
        return CODES[self.code][2]

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def __str__(self) -> str:
        return f"{self.line}:{self.col}: {self.severity} {self.code}: {self.message}"
