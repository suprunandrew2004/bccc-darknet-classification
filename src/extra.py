import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit

import common as cm
import load_bccc as ld

OUT = "./results/"
CL = ld.CLASSES
K = len(CL)


def enc(y, classes=CL):
    return y.map({c: i for i, c in enumerate(classes)}).values


def save(R, path):
    json.dump(R, open(path, "w"), ensure_ascii=False, indent=1, default=float)


def short(m):
    return {k: m[k] for k in ("accuracy", "bacc", "f1_macro", "pr_auc_macro",
                              "per_class", "confusion", "fold_f1_mean",
                              "fold_f1_std") if k in m}


def run(name, X, yn, folds, n_classes=K, **kw):
    p, f1, ba, t = cm.oof_predict(name, X, yn, folds, n_classes, **kw)
    m = cm.metrics(yn, p, n_classes)
    m.update(fold_f1_mean=float(f1.mean()), fold_f1_std=cm.sd(f1),
             fold_f1=f1.tolist())
    return m, p


def alpha_study(X, yn, folds):
    res = {}
    for a in (0.5, 0.0):
        m, _ = run("XGBoost", X, yn, folds, weights=a > 0, alpha=a)
        res[str(a)] = short(m)
        res[str(a)]["weights"] = cm.class_weights(yn, K, a).tolist()
        print(f"  alpha={a}: F1={m['f1_macro']:.3f}", flush=True)
    res["1.0_weights"] = cm.class_weights(yn, K, 1.0).tolist()
    return res


def mlp_dup_fit(Xtr, ytr):
    rng = np.random.RandomState(cm.SEED)
    n = np.bincount(ytr).max()
    idx = np.concatenate([rng.choice(np.where(ytr == k)[0], n, replace=True)
                          for k in np.unique(ytr)])
    m = cm.make_model("MLP", K)
    m.fit(Xtr[idx], ytr[idx])
    return m


def mlp_study(X, yn, folds, groups):
    Xv = X.values
    res = {}
    proba = np.zeros((len(yn), K))
    for tr, te in folds:
        proba[te] = mlp_dup_fit(Xv[tr], yn[tr]).predict_proba(Xv[te])
    res["dup"] = short(cm.metrics(yn, proba, K))
    configs = [((128, 64), 1e-4), ((128, 64), 1e-3), ((256, 128), 1e-4),
               ((256, 128), 1e-3)]
    proba = np.zeros((len(yn), K))
    chosen, f1s = [], []
    for tr, te in folds:
        gss = GroupShuffleSplit(1, test_size=0.2, random_state=cm.SEED)
        itr, iva = next(gss.split(tr, yn[tr], groups[tr]))
        itr, iva = tr[itr], tr[iva]
        best, best_s = None, -1
        for hl, a in configs:
            m = cm.make_model("MLP", K)
            m.set_params(mlpclassifier__hidden_layer_sizes=hl,
                         mlpclassifier__alpha=a, mlpclassifier__max_iter=300,
                         mlpclassifier__n_iter_no_change=10)
            sw = cm.class_weights(yn[itr], K)[yn[itr]]
            m.fit(Xv[itr], yn[itr], mlpclassifier__sample_weight=sw)
            s = f1_score(yn[iva], m.predict(Xv[iva]), average="macro",
                         labels=range(K))
            if s > best_s:
                best, best_s = (hl, a), s
        m = cm.make_model("MLP", K)
        m.set_params(mlpclassifier__hidden_layer_sizes=best[0],
                     mlpclassifier__alpha=best[1], mlpclassifier__max_iter=300,
                     mlpclassifier__n_iter_no_change=10)
        m.fit(Xv[tr], yn[tr],
              mlpclassifier__sample_weight=cm.class_weights(yn[tr], K)[yn[tr]])
        proba[te] = m.predict_proba(Xv[te])
        f1s.append(f1_score(yn[te], proba[te].argmax(1), average="macro",
                            labels=range(K)))
        chosen.append([list(best[0]), best[1]])
    res["nested"] = short(cm.metrics(yn, proba, K))
    res["nested"]["fold_f1_mean"] = float(np.mean(f1s))
    res["nested"]["fold_f1_std"] = cm.sd(f1s)
    res["nested"]["chosen"] = chosen
    print("  MLP:", {k: round(v["f1_macro"], 3) for k, v in res.items()})
    return res


def perm_groups(X, yn, folds, feats, name, n_repeats=3):
    groups = {}
    for f in feats:
        groups.setdefault(ld.feature_group(f), []).append(f)
    drops = {g: [] for g in groups}
    single = {}
    rng = np.random.RandomState(cm.SEED)
    for tr, te in folds:
        m = cm.make_model(name, K)
        cm.fit_model(name, m, X.values[tr], yn[tr], K)
        Xte = X.iloc[te].copy()
        base = f1_score(yn[te], m.predict(Xte.values), average="macro",
                        labels=range(K))
        for g, cols in groups.items():
            d = []
            for _ in range(n_repeats):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                Xp[cols] = Xp[cols].values[perm]
                d.append(base - f1_score(yn[te], m.predict(Xp.values),
                                         average="macro", labels=range(K)))
            drops[g].append(np.mean(d))
    return {g: [float(np.mean(v)), cm.sd(v)] for g, v in drops.items()}, \
        {g: len(c) for g, c in groups.items()}


def shap_oof(X, yn, folds, feats):
    import shap
    S = np.zeros(len(feats))
    per_class = np.zeros((K, len(feats)))
    n = 0
    for tr, te in folds:
        m = cm.make_model("XGBoost", K)
        cm.fit_model("XGBoost", m, X.values[tr], yn[tr], K)
        ex = shap.TreeExplainer(m)
        sv = np.asarray(ex.shap_values(X.values[te]))
        if sv.shape[0] == K:
            sv = np.moveaxis(sv, 0, -1)
        a = np.abs(sv)
        S += a.sum(axis=(0, 2))
        per_class += a.sum(axis=0).T
        n += len(te)
    S /= n
    per_class /= n
    order = np.argsort(-S)
    return {"mean_abs": {feats[i]: float(S[i]) for i in order[:40]},
            "per_class_top": {CL[k]: [feats[i] for i in
                                      np.argsort(-per_class[k])[:5]]
                              for k in range(K)},
            "group_share": pd.Series(S, index=feats).groupby(
                [ld.feature_group(f) for f in feats]).sum()
            .div(S.sum()).sort_values(ascending=False).to_dict()}


def builtin_importance(X, yn, feats):
    res = {}
    for name in ("Random Forest", "XGBoost"):
        m = cm.make_model(name, K)
        cm.fit_model(name, m, X.values, yn, K)
        if name == "XGBoost":
            g = m.get_booster().get_score(importance_type="gain")
            imp = np.array([g.get(f"f{i}", 0.0) for i in range(len(feats))])
            imp = imp / imp.sum()
        else:
            imp = m.feature_importances_
        o = np.argsort(-imp)[:15]
        res[name] = {feats[i]: float(imp[i]) for i in o}
    return res


def session_level(yn, proba, meta, width=None):
    g = meta["group"].values.astype(str)
    if width is not None:
        t = pd.to_datetime(meta["ts"]).astype("int64") // 10**9
        g = np.char.add(np.char.add(g, "_"),
                        (t // width).astype(str).values.astype(str))
    d = pd.DataFrame(proba, columns=range(K))
    d["g"], d["y"] = g, yn
    P = d.groupby("g")[list(range(K))].mean()
    Y = d.groupby("g")["y"].agg(lambda s: s.value_counts().index[0])
    pred = P.values.argmax(1)
    p, r, f, s = precision_recall_fscore_support(Y.values, pred,
                                                 labels=range(K),
                                                 zero_division=0)
    cover = [float(((d["y"] == k) & d["g"].map(Y).eq(k)).sum() /
                   max((d["y"] == k).sum(), 1)) for k in range(K)]
    return {"n_units": int(len(Y)), "precision": p.tolist(),
            "recall": r.tolist(), "support": s.tolist(),
            "f1_macro": float(f.mean()), "coverage": cover}


def stand_study(X, y, yn, meta):
    res = {}
    iscx = meta["year"].isin([2015, 2016]).values
    src = iscx.astype(int)
    folds = cm.folds_for(src, meta["group"].values)
    p, f1, ba, _ = cm.oof_predict("XGBoost", X, src, folds, 2)
    res["stand_separability"] = {
        "bacc": float(balanced_accuracy_score(src, p.argmax(1))),
        "n_iscx": int(iscx.sum()), "n_2020": int((~iscx).sum())}
    tor_iscx = (y == "Tor").values & iscx
    tr = np.where(~tor_iscx)[0]
    m = cm.make_model("XGBoost", K)
    cm.fit_model("XGBoost", m, X.values[tr], yn[tr], K)
    pr = m.predict(X.values[tor_iscx])
    res["tor_cross_stand"] = {
        "n": int(tor_iscx.sum()),
        "pred_counts": {CL[k]: int((pr == k).sum()) for k in range(K)}}
    tor20 = (y == "Tor").values & ~iscx
    tr = np.where(~tor20)[0]
    m = cm.make_model("XGBoost", K)
    cm.fit_model("XGBoost", m, X.values[tr], yn[tr], K)
    pr = m.predict(X.values[tor20])
    res["tor_cross_stand_rev"] = {
        "n": int(tor20.sum()),
        "pred_counts": {CL[k]: int((pr == k).sum()) for k in range(K)}}
    s20 = ~iscx
    cl4 = ["Tor", "I2P", "ZeroNet", "Freenet"]
    y4 = y[s20].map({c: i for i, c in enumerate(cl4)}).values
    X4 = X[s20].reset_index(drop=True)
    g4 = meta["group"].values[s20]
    f4 = cm.folds_for(y4, g4)
    p4, f1_4, _, _ = cm.oof_predict("XGBoost", X4, y4, f4, 4)
    m4 = cm.metrics(y4, p4, 4)
    res["within_2020"] = short(m4)
    res["within_2020"]["fold_f1_mean"] = float(f1_4.mean())
    res["within_2020"]["fold_f1_std"] = cm.sd(f1_4)
    res["within_2020"]["classes"] = cl4
    return res


def prior_shift(prec_rec, fpr):
    R, F = prec_rec
    return {str(pi): float(pi * R / (pi * R + (1 - pi) * fpr))
            for pi in (0.114, 0.01, 0.001, 0.0001)}


def fig_perm(P, sizes, path):
    order = sorted(P["XGBoost"], key=lambda g: -P["XGBoost"][g][0])
    fig, ax = plt.subplots(figsize=(9.5, 6))
    yy = np.arange(len(order))
    for j, (name, col, lab) in enumerate([("Random Forest", "#3b7dd8",
                                           "Случайный лес"),
                                          ("XGBoost", "#e8853b", "XGBoost")]):
        v = [P[name][g][0] for g in order]
        e = [P[name][g][1] for g in order]
        ax.barh(yy + (j - 0.5) * 0.38, v, 0.38, xerr=e, color=col, label=lab,
                error_kw={"lw": 0.8, "capsize": 2})
    ax.set_yticks(yy, [f"{g} ({sizes[g]})" for g in order])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Снижение макроусреднённой F1-меры при перестановке группы "
                  "признаков")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}".replace(".", ",")))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_shap(S, path, names):
    items = list(S["mean_abs"].items())[:15]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    lab = [names.get(k, k) for k, _ in items]
    ax.barh(range(len(items)), [v for _, v in items], color="#e8853b")
    ax.set_yticks(range(len(items)), lab, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Среднее |значение Шепли| (логарифм шансов, сумма по классам)")
    ax.grid(alpha=0.3, axis="x")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}".replace(".", ",")))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT + "results_extra.json")
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    R = json.load(open(a.out)) if os.path.exists(a.out) else {}
    X, y, feats, meta, info = ld.load("multi")
    yn = enc(y)
    folds = cm.folds_for(yn, meta["group"].values)
    oof = np.load(OUT + "oof_S3.npy")
    todo = a.only.split(",")

    def want(k):
        return "all" in todo or k in todo

    if want("session"):
        R["session"] = {}
        for nm, i in (("XGBoost", 1), ("Random Forest", 0)):
            R["session"][nm] = {
                "pair": session_level(yn, oof[i], meta),
                "w60": session_level(yn, oof[i], meta, 60),
                "w300": session_level(yn, oof[i], meta, 300)}
        pred = oof[1].argmax(1)
        tor = CL.index("Tor")
        rec = float(((pred == tor) & (yn == tor)).sum() / (yn == tor).sum())
        fpr = float(((pred == tor) & (yn != tor)).sum() / (yn != tor).sum())
        R["prior_shift_tor"] = {"recall": rec, "fpr": fpr,
                                "precision_at": prior_shift((rec, None), fpr)}
        save(R, a.out)
        print("session:", R["session"]["XGBoost"]["w300"])
    if want("group_by"):
        R["group_by"] = {}
        for how in ("src", "dst"):
            X2, y2, _, meta2, _ = ld.load("multi", group_by=how, verbose=False)
            y2n = enc(y2)
            f2 = cm.folds_for(y2n, meta2["group"].values)
            m, _ = run("XGBoost", X2, y2n, f2)
            R["group_by"][how] = short(m)
            print(f"  group {how}: {m['f1_macro']:.3f}", flush=True)
        save(R, a.out)
    if want("stand"):
        R["stand"] = stand_study(X, y, yn, meta)
        save(R, a.out)
        print("stand:", {k: v for k, v in R["stand"].items()
                         if k != "within_2020"},
              R["stand"]["within_2020"]["f1_macro"])
    if want("alpha"):
        R["alpha"] = alpha_study(X, yn, folds)
        save(R, a.out)
    if want("builtin"):
        R["builtin"] = builtin_importance(X, yn, feats)
        save(R, a.out)
    if want("perm"):
        P, sizes = {}, None
        for name in ("Random Forest", "XGBoost"):
            P[name], sizes = perm_groups(X, yn, folds, feats, name)
            print(" perm", name, flush=True)
        R["perm_groups"] = P
        R["group_sizes"] = sizes
        fig_perm(P, sizes, OUT + "fig_perm.png")
        save(R, a.out)
    if want("shap"):
        R["shap"] = shap_oof(X, yn, folds, feats)
        save(R, a.out)
    if want("mlp"):
        R["mlp"] = mlp_study(X, yn, folds, meta["group"].values)
        save(R, a.out)
    if want("conflicts"):
        R["conflicts"] = {}
        for how in ("majority", "drop"):
            X2, y2, _, meta2, info2 = ld.load("multi", conflicts=how,
                                              verbose=False)
            y2n = enc(y2)
            f2 = cm.folds_for(y2n, meta2["group"].values)
            m, _ = run("XGBoost", X2, y2n, f2)
            R["conflicts"][how] = short(m)
            R["conflicts"][how]["n"] = int(len(y2))
            R["conflicts"][how]["class_counts"] = info2["class_counts"]
            R["conflicts"][how]["n_relabeled"] = info2.get(
                "n_conflict_relabeled", 0)
            print(f"  conflicts {how}: {m['f1_macro']:.3f}", flush=True)
        save(R, a.out)
    if want("literature"):
        from sklearn.model_selection import StratifiedKFold
        X0, y0, _, meta0, _ = ld.load("multi", filter_bg=False,
                                      fix_iat_artifact=False, verbose=False)
        y0n = enc(y0)
        skf = StratifiedKFold(5, shuffle=True, random_state=cm.SEED)
        f0 = list(skf.split(np.zeros(len(y0n)), y0n))
        m, _ = run("XGBoost", X0, y0n, f0)
        R["literature_random_raw"] = short(m)
        print("  исходный набор, случайная CV:", m["f1_macro"], flush=True)
        save(R, a.out)
    if want("timing"):
        m = cm.make_model("XGBoost", K)
        cm.fit_model("XGBoost", m, X.values, yn, K)
        t0 = time.time()
        for _ in range(5):
            m.predict_proba(X.values)
        R["timing_flows_per_s"] = float(5 * len(X) / (time.time() - t0))
        save(R, a.out)
    print("готово")


if __name__ == "__main__":
    main()
