"""Inline every \\input{...} of src/main_src.tex into one self-contained main.tex and compile it.

    python build_report.py          # inline + pdflatex x2
    python build_report.py --no-pdf # only write main.tex
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MIKTEX = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "miktex", "bin", "x64")


def inline(path, depth=0):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        text = f.read()

    def rep(m):
        name = m.group(1)
        if not name.endswith(".tex"):
            name += ".tex"
        return inline(name, depth + 1)

    return re.sub(r"^\\input\{([^}]+)\}[ \t]*$", rep, text, flags=re.M)


def main():
    text = inline("src/main_src.tex")
    with open(os.path.join(ROOT, "main.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote main.tex", len(text.splitlines()), "lines")
    if "--no-pdf" in sys.argv:
        return
    env = dict(os.environ, PATH=MIKTEX + os.pathsep + os.environ["PATH"])
    for i in range(2):
        r = subprocess.run([os.path.join(MIKTEX, "pdflatex.exe"), "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-3000:])
            sys.exit(r.returncode)
    log = open(os.path.join(ROOT, "main.log"), encoding="utf-8", errors="replace").read()
    pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
    print("pdf pages:", pages.group(1) if pages else "?")
    for pat in ("Overfull \\\\hbox", "undefined", "Citation .* undefined", "Reference .* undefined"):
        n = len(re.findall(pat, log))
        print(f"  {pat}: {n}")


if __name__ == "__main__":
    main()
