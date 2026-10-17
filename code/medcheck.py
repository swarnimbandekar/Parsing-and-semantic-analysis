"""medcheck: compile a MedScript program and classify its errors.

    python medcheck.py examples/phi_leak.med
    python medcheck.py examples/triage_ok.med --ir
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from medscript import compile_source                 # noqa: E402
from medscript.diagnostics import FAMILIES           # noqa: E402
from medscript.features import extract               # noqa: E402
from medscript.ir import format_ir                   # noqa: E402

MODEL = os.path.join(HERE, "models", "rf_classifier.joblib")


def classify(res):
    """Return [(family, probability)] sorted by probability, or None without a model."""
    if not os.path.exists(MODEL):
        return None
    import joblib
    clf = joblib.load(MODEL)
    p = clf.predict_proba(extract(res).reshape(1, -1))[0]
    return sorted(((FAMILIES[c], float(q)) for c, q in zip(clf.classes_, p)), key=lambda t: -t[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--ir", action="store_true", help="print optimised three-address code")
    a = ap.parse_args()
    with open(a.file, encoding="utf-8") as fh:
        src = fh.read()
    res = compile_source(src)
    lines = src.splitlines()
    print(f"== {a.file}: {len(res.tokens) - 1} tokens, "
          f"{len(res.errors)} error(s), {len(res.warnings)} warning(s)")
    for d in res.diagnostics:
        print(f"  {d}")
        if 1 <= d.line <= len(lines):
            print(f"      | {lines[d.line - 1].strip()}")
    ranking = classify(res)
    if ranking:
        top = ", ".join(f"{f} {p:.2f}" for f, p in ranking[:3])
        print(f"== learned root-cause estimate: {top}")
        if res.ok and ranking[0][0] != "valid":
            print("   note: the rule-based front end found no error, but the classifier suspects a "
                  f"'{ranking[0][0]}' fault (possible clinical-logic error)")
    if res.ok and a.ir:
        print("== optimised IR")
        print(format_ir(res.ir_opt))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
