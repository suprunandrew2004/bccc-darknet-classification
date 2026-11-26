import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import common as cm
import load_bccc as ld

OUT = "./results/"


def comma(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def short(m):
    return {k: m[k] for k in ("accuracy", "bacc", "f1_macro", "pr_auc_macro",
                              "roc_auc_macro", "per_class", "confusion",
                              "fold_f1_mean", "fold_f1_std") if k in m}


def binary_level(R):
    X, y, feats, meta, info = ld.load("binary")
    R["binary_info"] = {k: v for k, v in info.items() if k != "dropped_const"}
    yn = (y == "Anon").astype(int).values
    folds = cm.folds_for(yn, meta["group"].values)
    R["binary"] = {}
    for name in cm.MODEL_NAMES:
        p, f1, ba, t = cm.oof_predict(name, X, yn, folds, 2)
        m = cm.metrics(yn, p, 2)
        m.update(fold_f1_mean=float(f1.mean()), fold_f1_std=cm.sd(f1))
        R["binary"][name] = short(m)
        print(f"  binary {name}: F1={comma(m['f1_macro'])}", flush=True)
    tr, te = train_test_split(np.arange(len(yn)), test_size=0.3, stratify=yn,
                              random_state=cm.SEED)
    mdl = cm.make_model("XGBoost", 2)
    cm.fit_model("XGBoost", mdl, X.values[tr], yn[tr], 2)
    R["binary_random"] = short(cm.metrics(yn[te], mdl.predict_proba(
        X.values[te]), 2))
    for tag, drop in (("no_env", ld.env_features(feats)),
                      ("no_payload", ld.payload_features(feats))):
        keep = [f for f in feats if f not in drop]
        p, f1, ba, t = cm.oof_predict("XGBoost", X, yn, folds, 2, feats=keep)
        R["binary_" + tag] = short(cm.metrics(yn, p, 2))

    s = meta["year"].isin([2015, 2016]).values
    Xs, ys, gs = X[s].reset_index(drop=True), yn[s], meta["group"].values[s]
    fs = cm.folds_for(ys, gs)
    p, f1, ba, t = cm.oof_predict("XGBoost", Xs, ys, fs, 2)
    R["binary_iscx_only"] = short(cm.metrics(ys, p, 2))
    R["binary_iscx_only"]["n"] = [int((ys == 0).sum()), int((ys == 1).sum())]
    R["binary_years"] = {str(k): v for k, v in
                         (meta.groupby(["year", y.values]).size()
                          .unstack(fill_value=0).to_dict("index").items())}
    print("  ISCX only:", comma(R["binary_iscx_only"]["f1_macro"]))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _inner_oof(Xv, y, groups, n_classes):
    folds = cm.folds_for(y, groups, n_splits=3)
    p = np.zeros((len(y), n_classes))
    for itr, iva in folds:
        m = cm.make_model("XGBoost", n_classes)
        cm.fit_model("XGBoost", m, Xv[itr], y[itr], n_classes)
        p[iva] = m.predict_proba(Xv[iva])
    return p


def calibrators(Xv, yn, groups, tr, an):
    yb = (yn[tr] > 0).astype(int)
    p1 = _inner_oof(Xv[tr], yb, groups[tr], 2)[:, 1]
    c1 = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p1)[:, None], yb)
    p2 = _inner_oof(Xv[an], yn[an] - 1, groups[an], 5)
    c2 = LogisticRegression(C=1e6, max_iter=2000).fit(
        np.log(np.clip(p2, 1e-6, 1)), yn[an] - 1)
    return c1, c2


def ece(y, p, n_bins=10):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        s = bins == b
        if s.any():
            e += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(e)


def two_level(R):
    CL6 = ld.UNION_CLASSES
    X, y, feats, meta, info = ld.load("union")
    yn = y.map({c: i for i, c in enumerate(CL6)}).values
    folds = cm.folds_for(yn, meta["group"].values)
    Xv = X.values
    p_flat, f1f, _, _ = cm.oof_predict("XGBoost", X, yn, folds, 6)
    p_casc = np.zeros((len(yn), 6))
    p_cal = np.zeros((len(yn), 6))
    p_seq = np.zeros(len(yn), dtype=int)
    pa_all, pa_cal_all = np.zeros(len(yn)), np.zeros(len(yn))
    groups = meta["group"].values
    f1c = []
    lvl1 = {"tp": 0, "fp": 0, "fn": 0}
    for tr, te in folds:
        yb = (yn[tr] > 0).astype(int)
        m1 = cm.make_model("XGBoost", 2)
        cm.fit_model("XGBoost", m1, Xv[tr], yb, 2)
        pa = m1.predict_proba(Xv[te])[:, 1]
        an = tr[yn[tr] > 0]
        m2 = cm.make_model("XGBoost", 5)
        cm.fit_model("XGBoost", m2, Xv[an], yn[an] - 1, 5)
        p2 = m2.predict_proba(Xv[te])
        p_casc[te, 0] = 1 - pa
        p_casc[te, 1:] = pa[:, None] * p2
        p_seq[te] = np.where(pa >= 0.5, 1 + p2.argmax(1), 0)
        c1, c2 = calibrators(Xv, yn, groups, tr, an)
        pa_c = c1.predict_proba(_logit(pa)[:, None])[:, 1]
        p2_c = c2.predict_proba(np.log(np.clip(p2, 1e-6, 1)))
        p_cal[te, 0] = 1 - pa_c
        p_cal[te, 1:] = pa_c[:, None] * p2_c
        pa_all[te], pa_cal_all[te] = pa, pa_c
        f1c.append(f1_score(yn[te], p_casc[te].argmax(1), average="macro",
                            labels=range(6)))
    R["union_info"] = {"class_counts": info["class_counts"],
                       "n": int(len(yn))}
    mf = cm.metrics(yn, p_flat, 6)
    mf.update(fold_f1_mean=float(f1f.mean()), fold_f1_std=cm.sd(f1f))
    mc = cm.metrics(yn, p_casc, 6)
    mc.update(fold_f1_mean=float(np.mean(f1c)), fold_f1_std=cm.sd(f1c))
    R["flat6"], R["cascade"] = short(mf), short(mc)
    yb_all = (yn > 0).astype(int)
    Y6 = np.eye(6)[yn]
    pr_c = p_casc.argmax(1)
    thr = 1.0 / (1.0 + p_casc[:, 1:].max(1) / np.maximum(p_casc[:, 1:].sum(1), 1e-12))
    R["calibration"] = {
        "ece_level1_raw": ece(yb_all, pa_all),
        "ece_level1_cal": ece(yb_all, pa_cal_all),
        "brier_level1_raw": float(np.mean((pa_all - yb_all) ** 2)),
        "brier_level1_cal": float(np.mean((pa_cal_all - yb_all) ** 2)),
        "brier6_raw": float(np.mean(((p_casc - Y6) ** 2).sum(1))),
        "brier6_cal": float(np.mean(((p_cal - Y6) ** 2).sum(1))),
        "mean_pa_raw_regular": float(pa_all[yb_all == 0].mean()),
        "mean_pa_cal_regular": float(pa_cal_all[yb_all == 0].mean()),
        "cascade_cal": short(cm.metrics(yn, p_cal, 6)),
        "sequential_f1": float(f1_score(yn, p_seq, average="macro",
                                        labels=range(6))),
        "sequential_vs_argmax_diff": int((p_seq != pr_c).sum()),
        "effective_threshold_range": [float(thr.min()), float(thr.max())],
        "effective_threshold_median": float(np.median(thr)),
    }
    print("  calibration", {k: v for k, v in R["calibration"].items()
                            if k != "cascade_cal"},
          comma(R["calibration"]["cascade_cal"]["f1_macro"]))
    pred = p_casc.argmax(1)
    R["cascade_level1"] = {
        "precision": float(((pred > 0) & (yn > 0)).sum() / (pred > 0).sum()),
        "recall": float(((pred > 0) & (yn > 0)).sum() / (yn > 0).sum())}
    print("  flat6", comma(mf["f1_macro"]), "cascade", comma(mc["f1_macro"]))
    cmx = np.array(mc["confusion"])
    norm = cmx / cmx.sum(1, keepdims=True)
    lab = ["Обычный"] + ld.CLASSES
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cmx[i, j]}\n{comma(100 * norm[i, j], 1)}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(6), lab)
    ax.set_yticks(range(6), lab)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    fig.savefig(OUT + "fig_cascade.png", dpi=200)
    plt.close(fig)


def main():
    R = {}
    binary_level(R)
    json.dump(R, open(OUT + "results_binary.json", "w"), ensure_ascii=False,
              indent=1)
    two_level(R)
    json.dump(R, open(OUT + "results_binary.json", "w"), ensure_ascii=False,
              indent=1)
    print("готово")


if __name__ == "__main__":
    main()
