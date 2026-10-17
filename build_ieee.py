"""Convert the report sources in src/ to an IEEEtran two-column paper and compile it.

    python build_ieee.py            # writes main_ieee.tex and main_ieee.pdf
    python build_ieee.py --no-pdf
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MIKTEX = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "miktex", "bin", "x64")

WIDE = {"fig:arch", "fig:confusion", "fig:augment", "fig:augclass", "fig:pca", "fig:anomaly",
        "tab:survey", "tab:learning", "tab:scaling", "tab:augrf", "tab:augmlp", "tab:augtests",
        "tab:anomaly", "tab:ops", "tab:diagcat"}

SECTION_LABELS = [
    ("\\section{Introduction}", "sec:intro"),
    ("\\section{Related Work}", "sec:related"),
    ("\\section{Proposed Methodology}", "sec:method"),
    ("\\section{Experimental Setup}", "sec:setup"),
    ("\\section{Results and Discussion}", "sec:results"),
    ("\\section{Conclusion and Future Work}", "sec:concl"),
    ("\\subsection{Threats to Validity and Limitations}", "sec:threats"),
    ("\\subsection{Augmentation under Scarcity: Does the GAN Help? (E3)}", "sec:aug"),
    ("\\section{MedScript Grammar}", "app:grammar"),
    ("\\section{Diagnostic Catalogue}", "app:diag"),
    ("\\section{Fault-Injection Operators}", "app:ops"),
    ("\\section{Running the Code}", "app:run"),
]

FILES = ["sec1_intro", "sec2_related", "sec3_method", "sec4_setup", "sec5a_results", "sec5b_results",
         "sec5c_results", "sec5d_results", "sec6_conclusion"]


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def balanced(text, start):
    """Return index just after the brace group that opens at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced")


def longtable_to_table(text):
    """longtable -> table* + tabular (two-column papers cannot use longtable)."""
    out, pos = "", 0
    while True:
        i = text.find("\\begin{longtable}", pos)
        if i < 0:
            return out + text[pos:]
        out += text[pos:i]
        j = text.index("{", i + len("\\begin{longtable}"))
        k = balanced(text, j)
        spec = text[j:k]
        end = text.index("\\end{longtable}", k)
        body = text[k:end]
        cap = re.search(r"\\caption\{(.*?)\}\\label\{(.*?)\}\\\\\n", body, flags=re.S)
        if cap:
            caption, label = cap.group(1), cap.group(2)
            body = body.replace(cap.group(0), "")
        else:
            caption, label = "Diagnostic codes of the MedScript compiler.", "tab:diagcat"
        body = body.replace("\\endhead\n", "")
        out += ("\\begin{table*}[t]\n\\centering\n\\caption{" + caption + "}\\label{" + label +
                "}\n\\scriptsize\n\\begin{tabular}" + spec + body + "\\end{tabular}\n\\end{table*}")
        pos = end + len("\\end{longtable}")


def floats(text):
    pat = re.compile(r"\\begin\{(figure|table|algorithm)\}\[H\](.*?)\\end\{\1\}", re.S)

    def rep(m):
        kind, body = m.group(1), m.group(2)
        label = re.search(r"\\label\{([^}]+)\}", body)
        wide = bool(label and label.group(1) in WIDE)
        if kind == "algorithm":
            return "\\begin{algorithm}[t]" + body + "\\end{algorithm}"
        env = kind + ("*" if wide else "")
        if not wide:
            body = body.replace("\\textwidth", "\\columnwidth")
            body = re.sub(r"width=0\.\d+\\columnwidth", r"width=\\columnwidth", body)
        body = body.replace("\\small", "\\footnotesize")
        body = re.sub(r"\\resizebox\{(\\textwidth|\\columnwidth)\}\{!\}\{(\\input\{[^}]+\})\}",
                      r"\\adjustbox{max width=\1}{\2}", body)
        if not wide:
            body = re.sub(r"^\\input\{(tables/[^}]+)\}$", r"\\adjustbox{max width=\\columnwidth}{\\input{\1}}",
                          body, flags=re.M)
        return f"\\begin{{{env}}}[t]" + body + f"\\end{{{env}}}"

    return pat.sub(rep, text)


def convert(text):
    text = re.sub(r"^\\newpage\s*$", "", text, flags=re.M)
    for s, lab in SECTION_LABELS:
        text = text.replace(s, s + "\\label{" + lab + "}")
    # cross references
    text = text.replace("Section~5.8", "Section~\\ref{sec:threats}")
    text = text.replace("Section~5.4", "Section~\\ref{sec:aug}")
    for d, lab in (("2", "related"), ("3", "method"), ("4", "setup"), ("5", "results"), ("6", "concl")):
        text = re.sub(r"Section~" + d + r"(?![.\d])", "Section~\\\\ref{sec:" + lab + "}", text)
    for a, lab in (("A", "grammar"), ("B", "diag"), ("C", "ops")):
        text = text.replace("Appendix~" + a, "Appendix~\\ref{app:" + lab + "}")
    # wording: paper instead of report
    for a, b in (("This report", "This paper"), ("this report", "this paper"), ("the report", "the paper"),
                 ("The report", "The paper"), ("Organisation of the Report", "Organisation of the Paper")):
        text = text.replace(a, b)
    text = text.replace("Figures~\\ref", "Figs.~\\ref").replace("Figure~\\ref", "Fig.~\\ref")
    text = text.replace("\\appendix\n", "\\appendices\n")
    text = text.replace("{\\small\n\\input{tables/operators.tex}\n}", "\\input{tables/operators.tex}")
    # narrow-column table specs
    text = text.replace("\\begin{tabular}{p{1.9cm}p{7.4cm}p{4.0cm}}", "\\begin{tabular}{p{1.1cm}p{4.2cm}p{2.5cm}}")
    text = text.replace("\\begin{tabular}{clp{7.4cm}}", "\\begin{tabular}{clp{5.1cm}}")
    text = text.replace("\\begin{tabular}{ll}\n\\toprule\n\\textbf{Component}",
                        "\\begin{tabular}{lp{5.4cm}}\n\\toprule\n\\textbf{Component}")
    text = text.replace("\\begin{tabular}{lrp{9.4cm}}", "\\begin{tabular}{lrp{5.3cm}}")
    # listings: use the global style
    text = re.sub(r",basicstyle=\\ttfamily\\\w+", "", text)
    text = re.sub(r"basicstyle=\\ttfamily\\\w+,", "", text)
    text = re.sub(r"\[language=\{\},numbers=none\]", "[language={}]", text)
    text = longtable_to_table(text)
    return floats(text)


def inline(text):
    def rep(m):
        name = m.group(1)
        if not name.endswith(".tex"):
            name += ".tex"
        sub = read(name)
        if "longtable" in sub:
            sub = longtable_to_table(sub)
        return sub.rstrip("\n")
    return re.sub(r"\\input\{(tables/(?!numbers)[^}]+)\}", rep, text)


def main():
    parts = [read("ieee/head.tex")]
    for f in FILES:
        parts.append(convert(read(f"src/{f}.tex")))
    # drop the section-level heading comment banners and the original abstract
    body = "\n".join(parts)
    app = convert(read("src/appendix.tex"))
    refs = read("src/refs.tex").replace("\\newpage", "").replace("{99}", "{38}")
    ack = ("\\section*{Acknowledgment}\nThe authors thank Dr.\\ Prakash Kumar Sarangi, School of Computer Science and "
           "Engineering (SCOPE), VIT-AP University, for guidance throughout the Compiler Design Lab.\n\n")
    app = app.replace("\\section{Diagnostic Catalogue}\\label{app:diag}",
                      "\\section{Diagnostic Catalogue}\\label{app:diag}\nTable~\\ref{tab:diagcat} lists the 23 diagnostic "
                      "codes with their phase, severity and error family.\n")
    text = body + "\n" + app + "\n" + ack + refs + "\n\\end{document}\n"
    text = inline(text)
    with open(os.path.join(ROOT, "main_ieee.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote main_ieee.tex", len(text.splitlines()), "lines")
    if "--no-pdf" in sys.argv:
        return
    env = dict(os.environ, PATH=MIKTEX + os.pathsep + os.environ["PATH"])
    for _ in range(2):
        r = subprocess.run([os.path.join(MIKTEX, "pdflatex.exe"), "-interaction=nonstopmode", "-halt-on-error",
                            "main_ieee.tex"], cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-3500:])
            sys.exit(r.returncode)
    log = open(os.path.join(ROOT, "main_ieee.log"), encoding="utf-8", errors="replace").read()
    pages = re.search(r"Output written on main_ieee.pdf \((\d+) pages", log)
    print("pdf pages:", pages.group(1) if pages else "?")
    for pat in ("Overfull \\\\hbox", "Citation .* undefined", "Reference .* undefined"):
        print(f"  {pat}: {len(re.findall(pat, log))}")


if __name__ == "__main__":
    main()
