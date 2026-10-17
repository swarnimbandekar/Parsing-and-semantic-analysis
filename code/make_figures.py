"""Turns results/*.json into figures (../figs) and LaTeX table fragments (../tables)."""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "figs")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "serif"})
COL = {"none": "#555555", "class_weight": "#8c564b", "ros": "#1f77b4", "jitter": "#2ca02c",
       "smote": "#ff7f0e", "gan_bce": "#d62728", "gan_wgan": "#9467bd"}
LABEL = {"none": "No augmentation", "class_weight": "Class weights", "ros": "Random oversampling",
         "jitter": "Gaussian jitter", "smote": "SMOTE", "gan_bce": "cGAN (BCE)", "gan_wgan": "cGAN (WGAN-GP)"}


def J(n):
    with open(os.path.join(RES, n)) as f:
        return json.load(f)


def tex(name, s):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(s)


def pm(a, d=3):
    return f"{np.mean(a):.{d}f}\\,$\\pm$\\,{np.std(a):.{d}f}"


def esc(s):
    return s.replace("_", "\\_")


# ---------------------------------------------------------------- rules vs ML
def fig_confusion():
    r = J("rules_vs_ml.json")
    F = r["families"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for a, (title, cm) in zip(ax, (("Rule baseline (first error)", r["rule_first"]["cm"]),
                                   ("Random forest (all features)", r["models"]["RF"]["cm"]))):
        m = np.array(cm, float)
        m = m / m.sum(1, keepdims=True)
        a.imshow(m, cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(len(F)))
        a.set_yticks(range(len(F)))
        a.set_xticklabels(F, rotation=60, ha="right", fontsize=7)
        a.set_yticklabels(F, fontsize=7)
        a.set_title(title, fontsize=9)
        a.set_xlabel("predicted")
        a.grid(False)
        for i in range(len(F)):
            for j in range(len(F)):
                if m[i, j] >= 0.01:
                    a.text(j, i, f"{m[i, j]:.2f}".lstrip("0") if m[i, j] < 1 else "1", ha="center", va="center",
                           fontsize=5.5, color="white" if m[i, j] > 0.6 else "black")
    ax[0].set_ylabel("true family")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confusion.pdf"))
    plt.close(fig)

    # tables
    rows = []
    for name, d in (("Rule: first error", r["rule_first"]), ("Rule: most severe family", r["rule_prio"])):
        rows.append((name, d["acc"], d["macro_f1"], d["acc_single"], d["acc_double"]))
    for name, d in r["models"].items():
        rows.append((name if name != "RF" else "Random forest",
                     d["acc"], d["macro_f1"], d["acc_single"], d["acc_double"]))
    body = "".join(f"{n} & {a:.4f} & {f:.4f} & {s:.4f} & {dd:.4f}\\\\\n" for n, a, f, s, dd in rows)
    tex("rules_vs_ml.tex",
        "\\begin{tabular}{lrrrr}\n\\toprule\n\\textbf{Classifier} & \\textbf{Accuracy} & \\textbf{Macro-F1} & "
        "\\textbf{Acc.\\ single-fault} & \\textbf{Acc.\\ two-fault}\\\\\n\\midrule\n" + body[:-1] +
        "\\bottomrule\n\\end{tabular}\n")

    F = r["families"]
    body = ""
    for i, f in enumerate(F):
        body += (f"{f} & {r['rule_first']['f1'][i]:.3f} & {r['rule_prio']['f1'][i]:.3f} & "
                 f"{r['models']['RF']['precision'][i]:.3f} & {r['models']['RF']['recall'][i]:.3f} & "
                 f"{r['models']['RF']['f1'][i]:.3f}\\\\\n")
    tex("perclass.tex",
        "\\begin{tabular}{lrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{\\textbf{Rule F1}} & "
        "\\multicolumn{3}{c}{\\textbf{Random forest}}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
        "\\textbf{Family} & first & severe & P & R & F1\\\\\n\\midrule\n" + body[:-1] +
        "\\bottomrule\n\\end{tabular}\n")

    top = r["rf_top_features"][:8]
    body = "".join(f"\\texttt{{{esc(n)}}} & {v:.3f}\\\\\n" for n, v in top)
    tex("topfeat.tex", "\\begin{tabular}{lr}\n\\toprule\n\\textbf{Feature} & \\textbf{Importance}\\\\\n\\midrule\n"
        + body[:-1] + "\\bottomrule\n\\end{tabular}\n")


def fig_learning():
    l = J("learning.json")
    sizes = l["sizes"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for (v, d), mk in zip(l["views"].items(), "osd^"):
        m = [np.mean(d[str(n)]) for n in sizes]
        s = [np.std(d[str(n)]) for n in sizes]
        ax.errorbar(sizes, m, yerr=s, marker=mk, ms=4, capsize=2, label=v)
    ax.set_xscale("log")
    ax.set_xlabel("training programs per class")
    ax.set_ylabel("macro-F1 (test)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="center right")
    fig.savefig(os.path.join(FIG, "learning.pdf"))
    plt.close(fig)
    body = ""
    for v, d in l["views"].items():
        body += f"{v} & " + " & ".join(f"{np.mean(d[str(n)]):.3f}" for n in sizes) + "\\\\\n"
    tex("learning.tex", "\\begin{tabular}{l" + "r" * len(sizes) + "}\n\\toprule\n\\textbf{Feature view} & " +
        " & ".join(f"\\textbf{{{n}}}" for n in sizes) + "\\\\\n\\midrule\n" + body[:-1] + "\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ augmentation
def fig_augment():
    a = J("augment.json")
    nreals, methods = a["nreals"], a["methods"]
    runs = a["runs"]

    def vals(m, n, key):
        return [r[m][key] for r in runs if r["nreal"] == n]

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, key, title in zip(axs, ("rf_f1", "mlp_f1"), ("Random forest", "MLP")):
        for m in methods:
            mean = [np.mean(vals(m, n, key)) for n in nreals]
            sd = [np.std(vals(m, n, key)) for n in nreals]
            ax.errorbar(nreals, mean, yerr=sd, marker="o", ms=3, capsize=1.5, color=COL[m], label=LABEL[m],
                        lw=1.2)
        ax.set_xscale("log")
        ax.set_xticks(nreals)
        ax.set_xticklabels(nreals)
        ax.set_xlabel("real programs per faulty class")
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel("macro-F1 (test)")
    axs[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "augment.pdf"))
    plt.close(fig)

    for key, name in (("rf_f1", "augment_rf.tex"), ("mlp_f1", "augment_mlp.tex")):
        body = ""
        for m in methods:
            cells = []
            for n in nreals:
                v = vals(m, n, key)
                best = max(np.mean(vals(mm, n, key)) for mm in methods)
                c = pm(v)
                cells.append(f"\\textbf{{{c}}}" if abs(np.mean(v) - best) < 1e-12 else c)
            body += f"{LABEL[m]} & " + " & ".join(cells) + "\\\\\n"
        tex(name, "\\begin{tabular}{l" + "c" * len(nreals) + "}\n\\toprule\n\\textbf{Method} & " +
            " & ".join(f"\\textbf{{{n}}}" for n in nreals) + "\\\\\n\\midrule\n" + body[:-1] +
            "\\bottomrule\n\\end{tabular}\n")

    # paired tests: each GAN variant vs. no augmentation and vs. SMOTE / jitter
    rows = ""
    summary = {}
    for n in nreals:
        for g in ("gan_bce", "gan_wgan"):
            gv = np.array(vals(g, n, "rf_f1"))
            cells = []
            for b in ("none", "smote", "jitter"):
                bv = np.array(vals(b, n, "rf_f1"))
                diff = gv - bv
                t, p = stats.ttest_rel(gv, bv)
                cells.append(f"{diff.mean():+.3f} ({p:.3f})")
                summary[f"{g}_vs_{b}_{n}"] = (float(diff.mean()), float(p))
            rows += f"{n} & {LABEL[g]} & " + " & ".join(cells) + "\\\\\n"
    tex("augment_tests.tex", "\\begin{tabular}{rlccc}\n\\toprule\n\\textbf{Real/class} & \\textbf{GAN} & "
        "\\textbf{vs.\\ none} & \\textbf{vs.\\ SMOTE} & \\textbf{vs.\\ jitter}\\\\\n\\midrule\n" + rows[:-1] +
        "\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(RES, "augment_tests.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # per-class F1 at 20 real examples
    from medscript.diagnostics import FAMILIES
    n0 = 20
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.13
    show = ["none", "ros", "jitter", "smote", "gan_bce", "gan_wgan"]
    for k, m in enumerate(show):
        pcs = np.array([r[m]["rf_pc"] for r in runs if r["nreal"] == n0])
        ax.bar(np.arange(10) + (k - 2.5) * w, pcs.mean(0), w, color=COL[m], label=LABEL[m])
    ax.set_xticks(range(10))
    ax.set_xticklabels(FAMILIES, rotation=30)
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=6.5, ncol=3, loc="lower left")
    fig.savefig(os.path.join(FIG, "augment_perclass.pdf"))
    plt.close(fig)

    # fidelity of the synthetic rows
    fid = [r for r in runs if r["seed"] == 0 and r["nreal"] == n0][0]["fidelity"]
    body = ""
    for m in ("ros", "jitter", "smote", "gan_bce", "gan_wgan"):
        d = fid[m]
        body += f"{LABEL[m]} & {d['label_validity']:.3f} & {d['nn_dist']:.2f} & {100 * d['dup_rate']:.1f}\\\\\n"
    tex("fidelity.tex", "\\begin{tabular}{lrrr}\n\\toprule\n\\textbf{Generator} & \\textbf{Label validity} & "
        "\\textbf{NN distance} & \\textbf{Exact copies (\\%)}\\\\\n\\midrule\n" + body[:-1] + "\\bottomrule\n"
        "\\end{tabular}\n")
    with open(os.path.join(TAB, "heldout_nn.tex"), "w") as f:
        f.write(f"{fid['real_heldout_nn_dist']:.2f}")


def fig_gan_pca():
    """PCA view of real vs. synthetic rows for four classes (one GAN trained with 100 real rows/class)."""
    from sklearn.decomposition import PCA
    from ml.gan import CGAN
    from medscript.diagnostics import FAMILIES
    import run_experiments as R
    d = R.load_data()
    te, pool = R.split(d)
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(7)
    tr = np.concatenate([rng.choice(pool[c], 100, replace=False) for c in range(10)])
    g = CGAN(X.shape[1], iters=2500, seed=3).fit(X[tr], y[tr])
    show = [5, 7, 8, 9]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.0))
    for ax, c in zip(axs, show):
        real = X[tr][y[tr] == c]
        syn = g.sample(np.full(100, c))
        pca = PCA(2).fit(np.vstack([real, X[tr]]))
        for A, col, lab, mk in ((pca.transform(X[tr][y[tr] != c]), "#cccccc", "other classes", "."),
                                (pca.transform(real), "#1f77b4", "real", "o"),
                                (pca.transform(syn), "#d62728", "GAN", "x")):
            ax.scatter(A[:, 0], A[:, 1], s=8 if mk != "." else 4, c=col, marker=mk, label=lab, alpha=0.8,
                       linewidths=0.8)
        ax.set_title(FAMILIES[c], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].legend(fontsize=5.5, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gan_pca.pdf"))
    plt.close(fig)
    hist = np.array(g.history)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(hist[:, 0], hist[:, 1], label="discriminator loss")
    ax.plot(hist[:, 0], hist[:, 2], label="generator loss")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "gan_loss.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------ anomaly
def fig_anomaly():
    a = J("anomaly.json")["auroc"]
    cols = list(next(iter(a.values())).keys())
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    names = list(a)
    w = 0.8 / len(names)
    for k, n in enumerate(names):
        ax.bar(np.arange(len(cols)) + (k - (len(names) - 1) / 2) * w, [a[n][c] for c in cols], w, label=n)
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30)
    ax.set_ylabel("AUROC vs. valid programs")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=6.5, ncol=3, loc="lower left")
    fig.savefig(os.path.join(FIG, "anomaly.pdf"))
    plt.close(fig)
    body = ""
    for c in cols:
        body += esc(c) + " & " + " & ".join(f"{a[n][c]:.3f}" for n in names) + "\\\\\n"
    tex("anomaly.tex", "\\begin{tabular}{l" + "r" * len(names) + "}\n\\toprule\n\\textbf{Fault family} & " +
        " & ".join("\\textbf{" + n.replace("(", "\\\\(") .replace("\\\\(", "(") + "}" for n in names) +
        "\\\\\n\\midrule\n" + body[:-1] + "\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------------- perf
def fig_perf():
    p = J("perf.json")
    sc = p["scaling"]
    loc = [s["loc"] for s in sc]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    tot = [sum(s[k] for k in ("lex", "parse", "sem", "ir", "opt")) * 1000 for s in sc]
    for k, lab in (("lex", "lexer"), ("parse", "parser"), ("sem", "semantic"), ("ir", "IR gen"), ("opt", "optimiser")):
        ax.plot(loc, [s[k] * 1000 for s in sc], marker="o", ms=3, label=lab)
    ax.plot(loc, tot, "k--", marker="s", ms=3, label="total")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("program size (lines)")
    ax.set_ylabel("time (ms)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "scaling.pdf"))
    plt.close(fig)
    body = ""
    for s, t in zip(sc, tot):
        body += (f"{s['loc']:,} & {s['ntok']:,} & {s['lex'] * 1000:.1f} & {s['parse'] * 1000:.1f} & "
                 f"{s['sem'] * 1000:.1f} & {s['ir'] * 1000:.1f} & {s['opt'] * 1000:.1f} & {t:.1f} & "
                 f"{s['n_ir']:,} & {s['n_ir_opt']:,}\\\\\n").replace(",", "{,}")
    tex("scaling.tex", "\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n\\textbf{Lines} & \\textbf{Tokens} & "
        "\\textbf{Lex} & \\textbf{Parse} & \\textbf{Sem} & \\textbf{IR} & \\textbf{Opt} & \\textbf{Total} & "
        "\\textbf{IR ins.} & \\textbf{Opt.\\ ins.}\\\\\n\\midrule\n" + body[:-1] + "\\bottomrule\n\\end{tabular}\n")


def operator_table():
    y = J("operator_yield.json")
    sys.path.insert(0, HERE)
    from medscript.mutations import OPERATORS
    desc = {
        "lex_illegal": "insert an illegal character", "lex_number": "malform a numeric literal",
        "lex_string": "drop a closing quote", "syn_semicolon": "delete a semicolon",
        "syn_paren": "delete a closing parenthesis", "syn_close_brace": "delete a closing brace",
        "syn_open_brace": "delete an opening brace", "syn_operand": "delete a comparison operand",
        "syn_keyword": "misspell a keyword", "name_typo": "misspell a variable use",
        "name_delete_decl": "delete a declaration", "name_redecl": "duplicate a declaration",
        "name_uninit": "remove an initialiser", "name_func": "misspell a function call",
        "name_assign_input": "assign to an input", "type_string_assign": "assign a string/bool to a number",
        "type_bool_assign": "assign a number to a bool", "type_arity": "change an argument count",
        "type_cond": "use a number as a condition", "type_return": "return the wrong type",
        "type_float_int": "assign a float to an int", "unit_swap_literal": "swap the unit of a literal",
        "unit_drop": "drop the unit of a literal", "unit_decl": "change a declared input unit",
        "unit_rate": "pass a wrong dose-rate unit", "unit_return": "return a wrong unit",
        "unit_add": "add quantities of different units", "range_literal": "implausible threshold literal",
        "range_threshold": "threshold outside declared range", "range_decl": "invert a declared range",
        "range_const": "implausible dose constant", "phi_direct": "print a PHI input",
        "phi_alias": "export an alias of PHI", "phi_concat": "concatenate PHI into output",
        "phi_removeanon": "remove an anonymize() call", "phi_func": "leak PHI through a function",
        "miss_unguard": "remove a presence guard", "miss_new_use": "use an optional value unguarded",
        "miss_make_optional": "make a required input optional", "logic_div_zero": "divide by constant zero",
        "logic_range_zero": "divisor range includes zero", "logic_missing_return": "delete a return",
        "logic_unreachable": "code after return", "logic_inverted": "invert a clinical comparison",
        "logic_wrong_message": "mismatch alert text and rule", "logic_constcond": "constant if-condition"}
    rows = ""
    for fam, ops in OPERATORS.items():
        for k, op in enumerate(ops):
            n = op.__name__
            v = y.get(n, {"tried": 0, "manifested": 0})
            rate = 100.0 * v["manifested"] / max(1, v["tried"])
            rows += (f"{fam if k == 0 else ''} & \\texttt{{{esc(n)}}} & {desc[n]} & {v['tried']} & "
                     f"{v['manifested']} & {rate:.0f}\\\\\n")
        rows += "\\midrule\n"
    tex("operators.tex", "\\begin{longtable}{llp{5.2cm}rrr}\n\\caption{Fault-injection operators and yield during "
        "corpus construction.}\\label{tab:ops}\\\\\n\\toprule\n\\textbf{Family} & \\textbf{Operator} & "
        "\\textbf{Fault injected} & \\textbf{Tried} & \\textbf{Kept} & \\textbf{\\%}\\\\\n\\midrule\n\\endhead\n"
        + rows[:-9] + "\\bottomrule\n\\end{longtable}\n")



def diag_tables():
    g = J("gan_diag.json")["runs"]
    rows = ""
    for b in (False, True):
        for l, name in (("bce", "cGAN (BCE)"), ("wgan-gp", "cGAN (WGAN-GP)")):
            r = [x for x in g if x["balanced"] == b and x["loss"] == l]
            rows += (f"{name} & {'class-balanced' if b else 'uniform'} & {pm([x['validity'] for x in r])} & "
                     f"{pm([x['f1'] for x in r])}\\\\\n")
    tex("gan_diag.tex", "\\begin{tabular}{llcc}\n\\toprule\n\\textbf{Model} & \\textbf{Mini-batches} & "
        "\\textbf{Label validity} & \\textbf{Macro-F1}\\\\\n\\midrule\n" + rows[:-1] +
        "\\bottomrule\n\\end{tabular}\n")
    c = J("anomaly_check.json")
    rows = ""
    for name, key in (("All 101 features", "all"), ("Without pattern features", "no_pat"),
                      ("Pattern features only", "pat_only")):
        r = [x for x in c["runs"] if x["cols"] == key]
        rows += f"{name} & {pm([x['logic'] for x in r])} & {pm([x['all faults'] for x in r])}\\\\\n"
    tex("anomaly_check.tex", "\\begin{tabular}{lcc}\n\\toprule\n\\textbf{Discriminator input} & "
        "\\textbf{AUROC, logic} & \\textbf{AUROC, all faults}\\\\\n\\midrule\n" + rows[:-1] +
        "\\bottomrule\n\\end{tabular}\n")


def numbers():
    """Key numbers as LaTeX macros so the text never drifts from the results."""
    r = J("rules_vs_ml.json")
    m = J("corpus_meta.json")
    p = J("perf.json")
    a = J("augment.json") if os.path.exists(os.path.join(RES, "augment.json")) else {"seconds": 0}
    out = {
        "NProg": f"{m['n_programs']:,}".replace(",", "{,}"), "NFeat": m["n_features"],
        "NTrain": f"{r['n_train']:,}".replace(",", "{,}"), "NTest": f"{r['n_test']:,}".replace(",", "{,}"),
        "MeanLoc": f"{m['mean_loc']:.1f}", "TotalLoc": f"{m['total_loc']:,}".replace(",", "{,}"),
        "NDouble": f"{m['double_fault']:,}".replace(",", "{,}"),
        "DoubleShare": f"{100 * m['double_fault'] / (m['n_programs'] - m['per_class']):.1f}",
        "RuleFirstAcc": f"{100 * r['rule_first']['acc']:.1f}", "RulePrioAcc": f"{100 * r['rule_prio']['acc']:.1f}",
        "RuleFirstDouble": f"{100 * r['rule_first']['acc_double']:.1f}",
        "RFAcc": f"{100 * r['models']['RF']['acc']:.2f}", "MLPAcc": f"{100 * r['models']['MLP']['acc']:.2f}",
        "HGBAcc": f"{100 * r['models']['HistGB']['acc']:.2f}", "LRAcc": f"{100 * r['models']['LogReg']['acc']:.2f}",
        "RFMacro": f"{r['models']['RF']['macro_f1']:.4f}", "RFDouble": f"{100 * r['models']['RF']['acc_double']:.1f}",
        "RuleLogicRecall": f"{100 * r['rule_prio']['recall'][9]:.1f}",
        "RFLogicRecall": f"{100 * r['models']['RF']['recall'][9]:.1f}",
        "MsPerKloc": f"{p['fit']['ms_per_kloc']:.0f}", "ScaleRSq": f"{p['fit']['r2']:.3f}",
        "ScaleSlope": f"{p['fit']['loglog_slope']:.2f}",
        "OptMean": f"{100 * p['opt']['mean_reduction']:.1f}", "OptMax": f"{100 * p['opt']['max_reduction']:.1f}",
        "OptPrograms": p["opt"]["programs"], "OptMismatch": p["opt"]["behaviour_mismatches"],
        "AugMinutes": f"{a['seconds'] / 60:.0f}",
        "HeldoutNN": f"{[r for r in a['runs'] if r['seed'] == 0 and r['nreal'] == 20][0]['fidelity']['real_heldout_nn_dist']:.1f}" if a.get("runs") else "0",
    }
    with open(os.path.join(TAB, "numbers.tex"), "w") as f:
        for k, v in out.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    which = sys.argv[1:] or ["confusion", "learning", "augment", "pca", "anomaly", "perf", "ops", "diag", "numbers"]
    fn = dict(confusion=fig_confusion, learning=fig_learning, augment=fig_augment, pca=fig_gan_pca,
              anomaly=fig_anomaly, perf=fig_perf, ops=operator_table, diag=diag_tables, numbers=numbers)
    for w in which:
        print("figure:", w, flush=True)
        fn[w]()
