import argparse
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

import common as cm
import load_bccc as ld

OUT = "./results/"
COLORS = {"Random Forest": "#3b7dd8", "XGBoost": "#e8853b", "MLP": "#5aae5e"}
plt.rcParams.update({"font.size": 11, "axes.formatter.use_locale": False})


def comma(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def _fmt_axis(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT + name, dpi=200)
    plt.close(fig)


def summarize(y, proba, n_classes, fold_f1=None, fold_bacc=None, t=None):
    m = cm.metrics(y, proba, n_classes)
    if fold_f1 is not None:
        m["fold_f1"] = fold_f1.tolist()
        m["fold_f1_mean"] = float(fold_f1.mean())
        m["fold_f1_std"] = cm.sd(fold_f1)
        m["fold_bacc"] = fold_bacc.tolist()
    if t is not None:
        m["time_s"] = t
    return m


def scheme_group(X, yn, folds, classes, models, feats=None, alpha=1.0):
    res, probas = {}, {}
    for name in models:
        p, f1, ba, t = cm.oof_predict(name, X, yn, folds, len(classes),
                                      alpha=alpha, feats=feats)
        res[name] = summarize(yn, p, len(classes), f1, ba, t)
        probas[name] = p
        print(f"  {name}: F1={comma(res[name]['f1_macro'])} "
              f"BAcc={comma(res[name]['bacc'])} ({t:.0f} c)", flush=True)
    return res, probas


def scheme_S1(X, y, meta, classes, models, n_repeats=10):
    out = {n: [] for n in models}
    first = {}
    for r in range(n_repeats):
        Xb, yb, _ = ld.undersample(X, y, meta, seed=cm.SEED + r)
        ybn = yb.map({c: i for i, c in enumerate(classes)}).values
        tr, te = train_test_split(np.arange(len(yb)), test_size=0.3,
                                  stratify=ybn, random_state=cm.SEED)
        for name in models:
            mdl = cm.make_model(name, len(classes))
            cm.fit_model(name, mdl, Xb.values[tr], ybn[tr], len(classes))
            p = mdl.predict_proba(Xb.values[te])
            m = cm.metrics(ybn[te], p, len(classes))
            out[name].append(m)
            if r == 0:
                first[name] = m
    res = {}
    for name in models:
        f1s = [m["f1_macro"] for m in out[name]]
        rep = {k: float(np.mean([m[k] for m in out[name]]))
               for k in ("accuracy", "bacc", "f1_macro", "pr_auc_macro")}
        rep["f1_per_class"] = np.mean([m["per_class"]["f1"] for m in out[name]],
                                      axis=0).tolist()
        res[name] = dict(first[name], rep_f1_mean=float(np.mean(f1s)),
                         rep_f1_std=cm.sd(f1s), rep_mean=rep,
                         n_train=int(len(tr)), n_test=int(len(te)))
        print(f"  S1 {name}: F1={comma(res[name]['f1_macro'])}, "
              f"повторы {comma(np.mean(f1s))} ± {comma(cm.sd(f1s))}")
    return res


def scheme_S2(X, yn, classes, models):
    tr, te = train_test_split(np.arange(len(yn)), test_size=0.3, stratify=yn,
                              random_state=cm.SEED)
    res = {}
    for name in models:
        mdl = cm.make_model(name, len(classes))
        cm.fit_model(name, mdl, X.values[tr], yn[tr], len(classes))
        res[name] = cm.metrics(yn[te], mdl.predict_proba(X.values[te]),
                               len(classes))
        res[name]["n_train"] = int(len(tr))
        print(f"  S2 {name}: F1={comma(res[name]['f1_macro'])}")
    skf = StratifiedKFold(5, shuffle=True, random_state=cm.SEED)
    folds = list(skf.split(np.zeros(len(yn)), yn))
    p, f1, ba, t = cm.oof_predict("XGBoost", X, yn, folds, len(classes))
    res["XGBoost"]["random_cv_f1_mean"] = float(f1.mean())
    res["XGBoost"]["random_cv_f1_std"] = cm.sd(f1)
    return res


def learning_curve(X, yn, folds, n_classes,
                   fracs=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0)):
    rows = []
    rng = np.random.RandomState(cm.SEED)
    for fr in fracs:
        te_s, tr_s = [], []
        for tr, te in folds:
            if fr < 1.0:
                sub, _ = train_test_split(tr, train_size=fr, stratify=yn[tr],
                                          random_state=cm.SEED)
            else:
                sub = tr
            m = cm.make_model("XGBoost", n_classes)
            cm.fit_model("XGBoost", m, X.values[sub], yn[sub], n_classes)
            te_s.append(f1_score(yn[te], m.predict(X.values[te]),
                                 average="macro", labels=range(n_classes)))
            tr_s.append(f1_score(yn[sub], m.predict(X.values[sub]),
                                 average="macro", labels=range(n_classes)))
        rows.append({"frac": fr, "n_train": int(np.mean(
            [len(tr) * fr for tr, _ in folds])),
            "test_mean": float(np.mean(te_s)), "test_std": cm.sd(te_s),
            "train_mean": float(np.mean(tr_s)), "train_std": cm.sd(tr_s)})
        print(f"  LC {fr}: {comma(rows[-1]['test_mean'])}", flush=True)
    return rows


def fig_conf(res, classes, labels, fname, models):
    n = len(models)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 5.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, name in zip(axes, models):
        cmx = np.array(res[name]["confusion"])
        norm = cmx / cmx.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                col = "white" if norm[i, j] > 0.55 else "black"
                ax.text(j, i, f"{cmx[i, j]}\n{comma(100 * norm[i, j], 1)}%",
                        ha="center", va="center", fontsize=11.5, color=col)
        ax.set_xticks(range(len(classes)), labels)
        ax.set_yticks(range(len(classes)), labels)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_title(f"{name} (сбал. точность {comma(100 * res[name]['bacc'], 1)}%)",
                     fontsize=13)
    _save(fig, fname)


def fig_pr(yn, proba, labels, fname):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    for k, lab in enumerate(labels):
        pr, rc, _ = precision_recall_curve(yn == k, proba[:, k])
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(yn == k, proba[:, k])
        ax.plot(rc, pr, color=cols[k], lw=2, label=f"{lab} (PR-AUC = {comma(ap)})")
        ax.axhline((yn == k).mean(), color=cols[k], ls=":", lw=1)
    ax.set_xlabel("Полнота (Recall)")
    ax.set_ylabel("Точность (Precision)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    _fmt_axis(ax, "x")
    _fmt_axis(ax)
    _save(fig, fname)


def fig_cv(res, models, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(models):
        v = np.array(res[name]["fold_f1"])
        ax.bar(i, v.mean(), 0.5, color=COLORS[name], alpha=0.75)
        ax.errorbar(i, v.mean(), yerr=cm.sd(v), color="black", capsize=8, lw=1.5)
        ax.scatter([i] * len(v), v, color="black", s=18, zorder=3)
        ax.text(i, v.max() + 0.03, comma(v.mean()), ha="center",
                fontweight="bold")
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel("F1-мера (macro)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    _fmt_axis(ax)
    _save(fig, fname)


def fig_lc(lc, fname):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    n = [r["n_train"] for r in lc]
    for key, lab, col in [("train", "Обучающая выборка", "#3b7dd8"),
                          ("test", "Контрольные блоки", "#e8853b")]:
        m = np.array([r[key + "_mean"] for r in lc])
        s = np.array([r[key + "_std"] for r in lc])
        ax.plot(n, m, "o-", color=col, label=lab)
        ax.fill_between(n, m - s, m + s, color=col, alpha=0.15)
    ax.set_xlabel("Объём обучающей выборки, потоков")
    ax.set_ylabel("F1-мера (macro)")
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    _fmt_axis(ax)
    _save(fig, fname)


def fig_schemes(R, fname):
    schemes = [("S1", "Сокращение выборки +\nслучайное разбиение"),
               ("S2", "Взвешивание +\nслучайное разбиение"),
               ("S3", "Взвешивание +\nгрупповое разбиение"),
               ("S4", "S3 без\nпризнаков среды"),
               ("S5", "S3 без энтропийных\nпризнаков")]
    models = cm.MODEL_NAMES
    fig, ax = plt.subplots(figsize=(12, 5.4))
    w = 0.27
    for j, name in enumerate(models):
        for i, (s, _) in enumerate(schemes):
            if name in R.get(s, {}):
                r = R[s][name]
                v = r.get("rep_f1_mean", r["f1_macro"])
                e = r.get("rep_f1_std", 0.0)
                ax.bar(i + (j - 1) * w, v, w, color=COLORS[name],
                       label=name if i == 1 else None)
                if e:
                    ax.errorbar(i + (j - 1) * w, v, yerr=e, color="black",
                                capsize=3, lw=1)
                ax.text(i + (j - 1) * w, v + e + 0.012, comma(v), ha="center",
                        fontsize=7.5)
    ax.set_xticks(range(len(schemes)),
                  [f"{s}\n{d}" for s, d in schemes], fontsize=9)
    ax.set_ylabel("F1-мера (macro)")
    ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncol=3, loc="upper right")
    _fmt_axis(ax)
    _save(fig, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT + "results_schemes.json")
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    classes = ld.CLASSES
    labels = classes
    R = {"env": {"python": platform.python_version(), "numpy": np.__version__,
                 "pandas": pd.__version__, "sklearn": sklearn.__version__,
                 "xgboost": xgboost.__version__}}
    X, y, feats, meta, info = ld.load("multi")
    R["info"] = {k: v for k, v in info.items() if k != "dropped_const"}
    R["info"]["dropped_const"] = info["dropped_const"]
    yn = y.map({c: i for i, c in enumerate(classes)}).values
    folds = cm.folds_for(yn, meta["group"].values)
    R["folds"] = [{"n_train": int(len(tr)), "n_test": int(len(te)),
                   "test_counts": np.bincount(yn[te], minlength=5).tolist()}
                  for tr, te in folds]
    R["groups"] = ld.group_diagnostics(y, meta["group"]).to_dict("records")
    print("S3 (основная схема)")
    R["S3"], probas = scheme_group(X, yn, folds, classes, cm.MODEL_NAMES)
    np.save(OUT + "oof_S3.npy", np.stack([probas[n] for n in cm.MODEL_NAMES]))
    np.save(OUT + "y_S3.npy", yn)
    json.dump(R, open(a.out, "w"), ensure_ascii=False, indent=1)
    fig_conf(R["S3"], classes, labels, "fig_conf.png", cm.MODEL_NAMES)
    fig_pr(yn, probas["XGBoost"], labels, "fig_pr.png")
    fig_cv(R["S3"], cm.MODEL_NAMES, "fig_cv_group.png")
    b = cm.group_bootstrap_f1(yn, {n: probas[n].argmax(1) for n in
                                   ["XGBoost", "Random Forest"]},
                              meta["group"].values, len(classes))
    n_tr = np.mean([len(tr) for tr, _ in folds])
    n_te = np.mean([len(te) for _, te in folds])
    R["stat_xgb_vs_rf"] = {"bootstrap": b, "nadeau_bengio": cm.nadeau_bengio(
        R["S3"]["XGBoost"]["fold_f1"], R["S3"]["Random Forest"]["fold_f1"],
        n_tr, n_te)}
    print("  стат. сравнение:", R["stat_xgb_vs_rf"])
    print("S4 (без признаков среды)")
    env = ld.env_features(feats)
    R["S4"], _ = scheme_group(X, yn, folds, classes, ["XGBoost"],
                              feats=[f for f in feats if f not in env])
    R["S4"]["XGBoost"]["n_removed"] = len(env)
    print("S5 (без признаков полезной нагрузки)")
    pay = ld.payload_features(feats)
    R["S5"], _ = scheme_group(X, yn, folds, classes, ["XGBoost"],
                              feats=[f for f in feats if f not in pay])
    R["S5"]["XGBoost"]["n_removed"] = len(pay)
    print("S1 (андерсэмплинг)")
    R["S1"] = scheme_S1(X, y, meta, classes, cm.MODEL_NAMES, a.repeats)
    print("S2 (случайное разбиение)")
    R["S2"] = scheme_S2(X, yn, classes, cm.MODEL_NAMES)
    json.dump(R, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("Влияние очистки набора (XGBoost, групповая кросс-валидация)")
    R["cleaning"] = {}
    for tag, fb, fi in [("C1_no_iat_fix", True, False),
                        ("C2_no_bg_filter", False, True),
                        ("C3_raw", False, False)]:
        X0, y0, f0, meta0, info0 = ld.load("multi", filter_bg=fb,
                                           fix_iat_artifact=fi)
        y0n = y0.map({c: i for i, c in enumerate(classes)}).values
        folds0 = cm.folds_for(y0n, meta0["group"].values)
        r0, p0 = scheme_group(X0, y0n, folds0, classes, ["XGBoost"])
        r0 = r0["XGBoost"]
        r0["n_after_clean"] = int(len(y0))
        r0["class_counts"] = info0["class_counts"]
        if not fb:
            bg = meta0["background"].values
            pred0 = p0["XGBoost"].argmax(1)
            r0["bg_n"] = int(bg.sum())
            r0["bg_acc"] = float((pred0[bg] == y0n[bg]).mean())
        R["cleaning"][tag] = r0
        if tag == "C3_raw":
            print("S1 на исходном наборе (типовая постановка литературы)")
            R["S1_raw"] = scheme_S1(X0, y0, meta0, classes, ["XGBoost"],
                                    a.repeats)
        json.dump(R, open(a.out, "w"), ensure_ascii=False, indent=1)
    fig_schemes(R, "fig_schemes.png")
    json.dump(R, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("Кривая обучения")
    R["learning_curve"] = learning_curve(X, yn, folds, len(classes))
    fig_lc(R["learning_curve"], "fig_learning_curve.png")
    json.dump(R, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("готово")


if __name__ == "__main__":
    sys.exit(main())
