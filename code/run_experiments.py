"""Reproduces every number in the report.

    python run_experiments.py            # run all stages (cached in results/)
    python run_experiments.py --force    # recompute everything

Stages: corpus, rules_vs_ml, learning, augment, anomaly, perf
"""

import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier  # noqa
from sklearn.linear_model import LogisticRegression                                           # noqa
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,       # noqa
                             confusion_matrix, roc_auc_score)
from sklearn.neural_network import MLPClassifier                                              # noqa
from sklearn.pipeline import make_pipeline                                                    # noqa
from sklearn.preprocessing import StandardScaler                                              # noqa
from sklearn.svm import OneClassSVM                                                           # noqa

from medscript import compile_source                                                          # noqa
from medscript.corpus import build_corpus                                                     # noqa
from medscript.diagnostics import FAMILIES                                                    # noqa
from medscript.equiv import equivalent                                                        # noqa
from medscript.features import extract, GROUP_IDX, NAMES                                      # noqa
from medscript.generator import generate_valid, to_source                                     # noqa
from medscript.mutations import PRIORITY                                                      # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
SEED, PER_CLASS, N_TEST = 42, 1000, 200
K = len(FAMILIES)


def jdump(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def cached(name, force):
    return os.path.exists(os.path.join(RES, name)) and not force


# ------------------------------------------------------------------ helpers
def rule_first(res):
    """Baseline 1: family of the first reported error (what a developer sees first)."""
    errs = [d for d in res.diagnostics if d.is_error]
    if errs:
        return FAMILIES.index(errs[0].family)
    if any(d.family == "logic" for d in res.diagnostics):
        return FAMILIES.index("logic")
    return 0


def rule_priority(res):
    """Baseline 2: the most severe family among all reported errors."""
    fams = {d.family for d in res.diagnostics if d.is_error and d.family != "valid"}
    if fams:
        return FAMILIES.index(min(fams, key=PRIORITY.index))
    if any(d.family == "logic" for d in res.diagnostics):
        return FAMILIES.index("logic")
    return 0


def featurise(chunk):
    out = []
    for src in chunk:
        r = compile_source(src, codegen=False)
        out.append((extract(r), rule_first(r), rule_priority(r), src.count("\n") + 1))
    return out


def mk_models(seed=0):
    return {
        "RF": RandomForestClassifier(300, random_state=seed, n_jobs=1),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=400, random_state=seed)),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
    }


def per_class(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=list(range(K)), zero_division=0)
    return dict(precision=pr.tolist(), recall=rc.tolist(), f1=f1.tolist())


def load_data():
    d = np.load(os.path.join(RES, "corpus.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def split(d):
    """Fixed stratified split: N_TEST programs per class held out, never used for training."""
    rng = np.random.default_rng(SEED)
    y = d["y"]
    te, pool = [], {c: [] for c in range(K)}
    for c in range(K):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        te += idx[:N_TEST].tolist()
        pool[c] = idx[N_TEST:].tolist()
    return np.array(te), pool


# ----------------------------------------------------------------- stages
def stage_corpus(force):
    if cached("corpus.npz", force):
        return
    t0 = time.time()
    stats = {}
    C = build_corpus(PER_CLASS, seed=SEED, stats=stats)
    t1 = time.time()
    srcs = [s.source for s in C]
    chunks = [srcs[i:i + 250] for i in range(0, len(srcs), 250)]
    rows = [r for ch in Parallel(n_jobs=10)(delayed(featurise)(c) for c in chunks) for r in ch]
    X = np.stack([r[0] for r in rows])
    np.savez_compressed(
        os.path.join(RES, "corpus.npz"), X=X, y=np.array([s.label for s in C]),
        rule_first=np.array([r[1] for r in rows]), rule_prio=np.array([r[2] for r in rows]),
        loc=np.array([r[3] for r in rows]), op=np.array([s.op for s in C]),
        secondary=np.array([s.secondary or "" for s in C]))
    os.makedirs(os.path.join(RES, "samples"), exist_ok=True)
    seen = {}
    for s in C:
        k = seen.get(s.family, 0)
        if k < 3:
            with open(os.path.join(RES, "samples", f"{s.family}_{k}.med"), "w", encoding="utf-8") as f:
                f.write(s.source)
        seen[s.family] = k + 1
    jdump("operator_yield.json", {k: {"tried": v[0], "manifested": v[1]} for k, v in stats.items()})
    jdump("corpus_meta.json", dict(
        n_programs=len(C), per_class=PER_CLASS, seed=SEED, build_seconds=t1 - t0,
        feature_seconds=time.time() - t1, n_features=int(X.shape[1]),
        mean_loc=float(np.mean([r[3] for r in rows])), total_loc=int(sum(r[3] for r in rows)),
        double_fault=int(sum(1 for s in C if s.secondary)),
        feature_names=NAMES))
    print(f"corpus: {len(C)} programs, {X.shape[1]} features, {time.time() - t0:.0f}s")


def fit_eval(model, X, y, Xt, yt):
    model.fit(X, y)
    p = model.predict(Xt)
    return p, float(accuracy_score(yt, p)), float(f1_score(yt, p, average="macro"))


def stage_rules_vs_ml(force):
    if cached("rules_vs_ml.json", force):
        return
    import joblib
    d = load_data()
    X, y = d["X"], d["y"]
    te, pool = split(d)
    tr = np.concatenate([pool[c] for c in range(K)])
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "families": FAMILIES}
    dbl = d["secondary"][te] != ""
    out["double_fault_share_test"] = float(dbl.mean())
    for name in ("rule_first", "rule_prio"):
        p = d[name][te]
        out[name] = dict(acc=float(accuracy_score(y[te], p)), macro_f1=float(f1_score(y[te], p, average="macro")),
                         cm=confusion_matrix(y[te], p, labels=list(range(K))).tolist(),
                         acc_single=float(accuracy_score(y[te][~dbl], p[~dbl])),
                         acc_double=float(accuracy_score(y[te][dbl], p[dbl])), **per_class(y[te], p))
    out["models"] = {}
    for name, m in mk_models().items():
        p, a, f = fit_eval(m, X[tr], y[tr], X[te], y[te])
        out["models"][name] = dict(acc=a, macro_f1=f, cm=confusion_matrix(y[te], p, labels=list(range(K))).tolist(),
                                   acc_single=float(accuracy_score(y[te][~dbl], p[~dbl])),
                                   acc_double=float(accuracy_score(y[te][dbl], p[dbl])), **per_class(y[te], p))
        print(f"  {name}: acc={a:.4f} macroF1={f:.4f}")
        if name == "RF":
            imp = m.feature_importances_
            order = np.argsort(-imp)[:15]
            out["rf_top_features"] = [(NAMES[i], float(imp[i])) for i in order]
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    small = RandomForestClassifier(100, random_state=0, n_jobs=-1, min_samples_leaf=2).fit(X[tr], y[tr])
    joblib.dump(small, os.path.join(HERE, "models", "rf_classifier.joblib"), compress=3)
    jdump("rules_vs_ml.json", out)


def stage_learning(force):
    if cached("learning.json", force):
        return
    d = load_data()
    X, y = d["X"], d["y"]
    te, pool = split(d)
    views = {"token+AST": np.concatenate([GROUP_IDX["tok"], GROUP_IDX["ast"]]),
             "diagnostics": GROUP_IDX["diag"],
             "diagnostics+patterns": np.concatenate([GROUP_IDX["diag"], GROUP_IDX["pat"]]),
             "all features": np.arange(X.shape[1])}
    sizes = [5, 10, 25, 50, 100, 200, 800]

    def job(vn, n, seed):
        rng = np.random.default_rng(seed)
        tr = np.concatenate([rng.choice(pool[c], n, replace=False) for c in range(K)])
        cols = views[vn]
        rf = RandomForestClassifier(200, random_state=seed, n_jobs=1).fit(X[tr][:, cols], y[tr])
        return vn, n, float(f1_score(y[te], rf.predict(X[te][:, cols]), average="macro"))

    res = Parallel(n_jobs=10)(delayed(job)(v, n, s) for v in views for n in sizes for s in range(5))
    out = {"sizes": sizes, "views": {v: {str(n): [r[2] for r in res if r[0] == v and r[1] == n] for n in sizes}
                                     for v in views}}
    jdump("learning.json", out)


METHODS = ["none", "class_weight", "ros", "jitter", "smote", "gan_bce", "gan_wgan"]


def aug_job(nreal, seed, d, te, pool):
    from ml.augment import random_oversample, jitter_oversample, smote, gan_oversample
    import torch
    torch.set_num_threads(1)
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(1000 * nreal + seed)
    idx = [rng.choice(pool[0], 400, replace=False)]
    idx += [rng.choice(pool[c], nreal, replace=False) for c in range(1, K)]
    tr = np.concatenate(idx)
    Xtr, ytr = X[tr], y[tr]
    sets = {"none": (Xtr, ytr), "class_weight": (Xtr, ytr),
            "ros": random_oversample(Xtr, ytr, 400, rng), "jitter": jitter_oversample(Xtr, ytr, 400, rng),
            "smote": smote(Xtr, ytr, 400, rng),
            "gan_bce": gan_oversample(Xtr, ytr, 400, rng, "bce", 2000, seed),
            "gan_wgan": gan_oversample(Xtr, ytr, 400, rng, "wgan-gp", 1000, seed)}
    res = {"nreal": nreal, "seed": seed}
    for m, (Xa, ya) in sets.items():
        rf = RandomForestClassifier(300, random_state=seed, n_jobs=1,
                                    class_weight="balanced" if m == "class_weight" else None).fit(Xa, ya)
        p = rf.predict(X[te])
        mlp = make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=400, random_state=seed)).fit(Xa, ya)
        pm = mlp.predict(X[te])
        res[m] = dict(rf_f1=float(f1_score(y[te], p, average="macro")), rf_acc=float(accuracy_score(y[te], p)),
                      mlp_f1=float(f1_score(y[te], pm, average="macro")),
                      rf_pc=per_class(y[te], p)["f1"], n_syn=int(len(ya) - len(ytr)))
    if seed == 0:       # fidelity diagnostics of the synthetic rows
        full = np.concatenate([pool[c] for c in range(K)])
        oracle = RandomForestClassifier(200, random_state=0, n_jobs=1).fit(X[full], y[full])
        sd = Xtr.std(0) + 1e-3
        fid = {}
        for m in ("ros", "jitter", "smote", "gan_bce", "gan_wgan"):
            Xa, ya = sets[m]
            Xs, ys = Xa[len(ytr):], ya[len(ytr):]
            valid = float((oracle.predict(Xs) == ys).mean())
            nn, dup = [], 0
            for x, c in zip(Xs, ys):
                ref = Xtr[ytr == c]
                dist = np.sqrt((((ref - x) / sd) ** 2).sum(1)).min()
                nn.append(dist)
                dup += dist < 1e-6
            fid[m] = dict(label_validity=valid, nn_dist=float(np.mean(nn)), dup_rate=float(dup / len(Xs)))
        used = set(tr.tolist())
        ref_d = []
        for c in range(1, K):
            held = [i for i in pool[c] if i not in used][:50]
            ref = Xtr[ytr == c]
            for i in held:
                ref_d.append(np.sqrt((((ref - X[i]) / sd) ** 2).sum(1)).min())
        fid["real_heldout_nn_dist"] = float(np.mean(ref_d))
        res["fidelity"] = fid
    return res


def stage_augment(force):
    if cached("augment.json", force):
        return
    d = load_data()
    te, pool = split(d)
    nreals, seeds = [5, 10, 20, 50, 100], list(range(10))
    t0 = time.time()
    out = Parallel(n_jobs=10, verbose=5)(delayed(aug_job)(n, s, d, te, pool) for n in nreals for s in seeds)
    jdump("augment.json", {"nreals": nreals, "seeds": seeds, "methods": METHODS, "runs": out,
                           "seconds": time.time() - t0})


def stage_anomaly(force):
    if cached("anomaly.json", force):
        return
    from ml.gan import CGAN
    d = load_data()
    X, y = d["X"], d["y"]
    te, pool = split(d)
    Xv = X[pool[0]]
    sc = StandardScaler().fit(Xv)
    out = {"auroc": {}}
    scorers = {}
    ifo = IsolationForest(n_estimators=300, random_state=0).fit(Xv)
    scorers["Isolation Forest"] = lambda A: -ifo.score_samples(A)
    oc = OneClassSVM(nu=0.05, gamma="scale").fit(sc.transform(Xv))
    scorers["One-class SVM"] = lambda A: -oc.score_samples(sc.transform(A))
    gans = {}
    for name, loss, it in (("GAN discriminator (BCE)", "bce", 3000), ("GAN critic (WGAN-GP)", "wgan-gp", 1500)):
        g = CGAN(X.shape[1], n_classes=1, loss=loss, iters=it, seed=0).fit(Xv, np.zeros(len(Xv), int))
        gans[name] = g
        scorers[name] = (lambda gg: (lambda A: -gg.critic(A, np.zeros(len(A), int))))(g)
    yt = y[te]
    vt = X[te][yt == 0]
    for sname, fn in scorers.items():
        sv = fn(vt)
        row = {}
        for c in range(1, K):
            sf = fn(X[te][yt == c])
            row[FAMILIES[c]] = float(roc_auc_score(np.r_[np.zeros(len(sv)), np.ones(len(sf))], np.r_[sv, sf]))
        sf = fn(X[te][yt != 0])
        row["all faults"] = float(roc_auc_score(np.r_[np.zeros(len(sv)), np.ones(len(sf))], np.r_[sv, sf]))
        out["auroc"][sname] = row
    vi = te[yt == 0]
    flag = lambda idx: np.array([float(d["rule_first"][i] != 0) for i in idx])  # noqa: E731
    row = {}
    for c in range(1, K):
        ids = te[yt == c]
        row[FAMILIES[c]] = float(roc_auc_score(np.r_[np.zeros(len(vi)), np.ones(len(ids))],
                                               np.r_[flag(vi), flag(ids)]))
    ids = te[yt != 0]
    row["all faults"] = float(roc_auc_score(np.r_[np.zeros(len(vi)), np.ones(len(ids))],
                                            np.r_[flag(vi), flag(ids)]))
    out["auroc"]["Compiler error flag"] = row
    out["gan_history"] = {k: g.history for k, g in gans.items()}
    jdump("anomaly.json", out)


def stage_gan_diag(force):
    """Why does the augmentation GAN not help?  Label validity of its samples, with and without
    class-balanced mini-batches (n_real = 20, five seeds)."""
    if cached("gan_diag.json", force):
        return
    d = load_data()
    te, pool = split(d)

    def job(seed, balanced, loss):
        import torch
        from ml.gan import CGAN
        torch.set_num_threads(1)
        X, y = d["X"], d["y"]
        rng = np.random.default_rng(900 + seed)
        tr = np.concatenate([rng.choice(pool[0], 400, replace=False)] +
                            [rng.choice(pool[c], 20, replace=False) for c in range(1, K)])
        full = np.concatenate([pool[c] for c in range(K)])
        oracle = RandomForestClassifier(200, random_state=0, n_jobs=1).fit(X[full], y[full])
        g = CGAN(X.shape[1], loss=loss, iters=2000 if loss == "bce" else 1000, seed=seed,
                 balanced=balanced).fit(X[tr], y[tr])
        ys = np.repeat(np.arange(1, K), 380)
        Xs = np.vstack([g.sample(np.full(380, c)) for c in range(1, K)])
        validity = float((oracle.predict(Xs) == ys).mean())
        rf = RandomForestClassifier(300, random_state=0, n_jobs=1).fit(np.vstack([X[tr], Xs]), np.r_[y[tr], ys])
        return dict(seed=seed, balanced=balanced, loss=loss, validity=validity,
                    f1=float(f1_score(y[te], rf.predict(X[te]), average="macro")))

    runs = Parallel(n_jobs=10)(delayed(job)(s, b, l) for s in range(5) for b in (False, True)
                               for l in ("bce", "wgan-gp"))
    jdump("gan_diag.json", {"runs": runs})


def stage_anomaly_check(force):
    """Sanity checks for the anomaly-detection result: constant features, seed stability, ablation."""
    if cached("anomaly_check.json", force):
        return
    from ml.gan import CGAN
    d = load_data()
    X, y = d["X"], d["y"]
    te, pool = split(d)
    Xv = X[pool[0]]
    out = {"n_constant_features_in_valid": int((Xv.std(0) < 1e-9).sum()),
           "n_features": int(X.shape[1]),
           "constant_by_group": {g: int((Xv[:, idx].std(0) < 1e-9).sum()) for g, idx in GROUP_IDX.items()}}
    yt = y[te]
    vt = X[te][yt == 0]

    def job(seed, cols_name):
        import torch
        torch.set_num_threads(1)
        cols = (np.arange(X.shape[1]) if cols_name == "all" else
                np.setdiff1d(np.arange(X.shape[1]), GROUP_IDX["pat"]) if cols_name == "no_pat" else
                GROUP_IDX["pat"])
        g = CGAN(len(cols), n_classes=1, loss="bce", iters=3000, seed=seed).fit(Xv[:, cols], np.zeros(len(Xv), int))
        f = lambda A: -g.critic(A[:, cols], np.zeros(len(A), int))   # noqa: E731
        sv = f(vt)
        res = {}
        for c in (9, 0):
            if c == 0:
                continue
            sf = f(X[te][yt == c])
            res["logic"] = float(roc_auc_score(np.r_[np.zeros(len(sv)), np.ones(len(sf))], np.r_[sv, sf]))
        sf = f(X[te][yt != 0])
        res["all faults"] = float(roc_auc_score(np.r_[np.zeros(len(sv)), np.ones(len(sf))], np.r_[sv, sf]))
        return dict(seed=seed, cols=cols_name, **res)

    out["runs"] = Parallel(n_jobs=10)(delayed(job)(s, c) for s in range(5) for c in ("all", "no_pat", "pat_only"))
    # Isolation forest on features that actually vary vs. all features
    vary = np.flatnonzero(Xv.std(0) > 1e-9)
    ifo = IsolationForest(n_estimators=300, random_state=0).fit(Xv)
    ifo2 = IsolationForest(n_estimators=300, random_state=0).fit(Xv[:, vary])
    def auc(fn, c):
        sv = fn(vt)
        sf = fn(X[te][yt == c]) if c else fn(X[te][yt != 0])
        return float(roc_auc_score(np.r_[np.zeros(len(sv)), np.ones(len(sf))], np.r_[sv, sf]))
    out["isolation_forest"] = {"all_features_all_faults": auc(lambda A: -ifo.score_samples(A), 0),
                               "only_varying_features_all_faults": auc(lambda A: -ifo2.score_samples(A[:, vary]), 0)}
    jdump("anomaly_check.json", out)


def scale_program(n):
    head = ["input volume: ml range(10, 5000);", "input duration: hour range(0.5, 24);",
            "input temp: degC range(30, 43);", "input hr: bpm range(20, 250);",
            "input weight: kg range(2, 300);", "",
            "func dose_for(w: kg, rate: mg/kg) -> mg {", "    let d: mg = rate * w;",
            "    if (d > 1000 mg) {", "        return 1000 mg;", "    }", "    return d;", "}", ""]
    body = []
    for k in range(n):
        t = k % 3
        if t == 0:
            body += [f"let r{k}: ml/hour = volume / duration;",
                     f"if (r{k} > {200 + k % 50} ml/hour) {{", '    print("ALERT high_rate");', "}"]
        elif t == 1:
            body += [f"let d{k}: mg = dose_for(weight, {5 + k % 20} mg/kg);", f"print(d{k});"]
        else:
            body += ["if (temp > 38.0 degC && hr > 100 bpm) {", '    print("ALERT sepsis_risk");', "}"]
    return "\n".join(head + body) + "\n"


def stage_perf(force):
    if cached("perf.json", force):
        return
    from medscript.ir import format_ir
    out = {"scaling": []}
    for n in (10, 50, 100, 250, 500, 1000, 2000, 4000):
        src = scale_program(n)
        runs = []
        for _ in range(5):
            r = compile_source(src)
            runs.append(r.timings)
        assert r.ok, r.diagnostics[:3]
        med = {k: float(np.median([t[k] for t in runs])) for k in ("lex", "parse", "sem", "ir", "opt")}
        out["scaling"].append(dict(n=n, loc=src.count("\n") + 1, ntok=len(r.tokens), n_ir=r.timings["n_ir"],
                                   n_ir_opt=r.timings["n_ir_opt"], **med))
    loc = np.array([s["loc"] for s in out["scaling"]], float)
    tot = np.array([s["lex"] + s["parse"] + s["sem"] + s["ir"] + s["opt"] for s in out["scaling"]])
    A = np.vstack([loc, np.ones_like(loc)]).T
    coef, *_ = np.linalg.lstsq(A, tot, rcond=None)
    pred = A @ coef
    r2 = 1 - ((tot - pred) ** 2).sum() / ((tot - tot.mean()) ** 2).sum()
    slope = np.polyfit(np.log(loc[2:]), np.log(tot[2:]), 1)[0]
    out["fit"] = dict(ms_per_kloc=float(coef[0] * 1e6), r2=float(r2), loglog_slope=float(slope))
    red, mism, n = [], 0, 300
    for s in range(n):
        lines, _ = generate_valid(random.Random(5000 + s))
        r = compile_source(to_source(lines))
        red.append(1 - r.timings["n_ir_opt"] / r.timings["n_ir"])
        ok, *_ = equivalent(r, trials=10, seed=s)
        mism += (not ok)
    out["opt"] = dict(programs=n, mean_reduction=float(np.mean(red)), median_reduction=float(np.median(red)),
                      max_reduction=float(np.max(red)), behaviour_mismatches=int(mism))
    with open(os.path.join(HERE, "examples", "triage_ok.med"), encoding="utf-8") as fh:
        r = compile_source(fh.read())
    with open(os.path.join(RES, "example_ir.txt"), "w") as f:
        f.write("=== unoptimised ===\n" + format_ir(r.ir) + "\n\n=== optimised ===\n" + format_ir(r.ir_opt) + "\n")
    jdump("perf.json", out)


STAGES = dict(corpus=stage_corpus, rules_vs_ml=stage_rules_vs_ml, learning=stage_learning,
              augment=stage_augment, anomaly=stage_anomaly, gan_diag=stage_gan_diag,
              anomaly_check=stage_anomaly_check, perf=stage_perf)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage", nargs="*", default=list(STAGES))
    a = ap.parse_args()
    for s in a.stage:
        t = time.time()
        print(f"[{s}] ...", flush=True)
        STAGES[s](a.force)
        print(f"[{s}] done in {time.time() - t:.0f}s", flush=True)
