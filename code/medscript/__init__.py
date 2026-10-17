"""MedScript: a small domain-specific language for medical data processing,
together with the front end (lexer, parser, semantic analyser), a three-address
code back end and the feature extractor used by the learned error classifier."""

from .driver import compile_source, CompileResult  # noqa: F401
from .diagnostics import Diagnostic, FAMILIES, CODES  # noqa: F401
