import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
N_SPLITS = 5
MODEL_NAMES = ["Random Forest", "XGBoost", "MLP"]

warnings.filterwarnings("ignore")


def sd(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def class_weights(y_num, n_classes, alpha=1.0):
    counts = np.bincount(y_num, minlength=n_classes).astype(float)
    w = (len(y_num) / (n_classes * np.maximum(counts, 1))) ** alpha
    return w


def make_model(name, n_classes, seed=SEED):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, objective=obj, tree_method="hist",
            n_jobs=-1, random_state=seed, verbosity=0)
    if name == "MLP":
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0,
                          add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                          alpha=1e-4, early_stopping=True,
                          validation_fraction=0.1, n_iter_no_change=5,
                          max_iter=100, random_state=seed))
    raise ValueError(name)


def fit_model(name, model, X_tr, y_tr, n_classes, weights=True, alpha=1.0,
              seed=SEED):
    if name == "Random Forest":
        if not weights:
            model.set_params(class_weight=None)
        model.fit(X_tr, y_tr)
    elif name == "XGBoost":
        sw = None
        if weights:
            sw = class_weights(y_tr, n_classes, alpha)[y_tr]
        model.fit(X_tr, y_tr, sample_weight=sw)
    else:
        sw = class_weights(y_tr, n_classes)[y_tr] if weights else None
        model.fit(X_tr, y_tr, mlpclassifier__sample_weight=sw)
    return model


def predict_proba(model, X):
    p = model.predict_proba(X)
    return p


def folds_for(y_num, groups, n_splits=N_SPLITS, seed=SEED):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    return list(sgkf.split(np.zeros(len(y_num)), y_num, groups))


def oof_predict(name, X, y_num, folds, n_classes, weights=True, alpha=1.0,
                feats=None):
    Xv = X if feats is None else X[feats]
    Xv = Xv.values if hasattr(Xv, "values") else Xv
    proba = np.zeros((len(y_num), n_classes))
    fold_f1, fold_bacc = [], []
    t0 = time.time()
    for tr, te in folds:
        m = make_model(name, n_classes)
        fit_model(name, m, Xv[tr], y_num[tr], n_classes, weights, alpha)
        proba[te] = m.predict_proba(Xv[te])
        pr = proba[te].argmax(1)
        fold_f1.append(f1_score(y_num[te], pr, average="macro",
                                labels=range(n_classes), zero_division=0))
        fold_bacc.append(balanced_accuracy_score(y_num[te], pr))
    return proba, np.array(fold_f1), np.array(fold_bacc), time.time() - t0


def metrics(y, proba, n_classes):
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=range(n_classes), zero_division=0)
    Y = np.eye(n_classes)[y]
    ap = [average_precision_score(Y[:, k], proba[:, k]) for k in range(n_classes)]
    if n_classes == 2:
        roc = roc_auc_score(y, proba[:, 1])
    else:
        roc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    return {
        "accuracy": accuracy_score(y, pred),
        "bacc": balanced_accuracy_score(y, pred),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(np.mean(f)),
        "pr_auc_macro": float(np.mean(ap)),
        "roc_auc_macro": float(roc),
        "per_class": {"precision": p.tolist(), "recall": r.tolist(),
                      "f1": f.tolist(), "support": s.tolist(), "ap": ap},
        "confusion": confusion_matrix(y, pred, labels=range(n_classes)).tolist(),
    }


def nadeau_bengio(a, b, n_tr, n_te):
    from scipy import stats
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_te / n_tr) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(d.mean()), float(t), float(p)


def group_bootstrap_f1(y, preds, groups, n_classes, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    g = pd.Series(groups)
    dom = pd.Series(y).groupby(g.values).agg(lambda s: s.value_counts().index[0])
    idx_by_g = g.groupby(g.values).indices
    by_cls = {c: dom[dom == c].index.values for c in dom.unique()}
    res = {k: [] for k in preds}
    res["diff"] = []
    names = list(preds)
    for _ in range(n_boot):
        sel = np.concatenate([rng.choice(v, len(v), replace=True)
                              for v in by_cls.values()])
        idx = np.concatenate([idx_by_g[s] for s in sel])
        vals = {}
        for k in names:
            vals[k] = f1_score(y[idx], preds[k][idx], average="macro",
                               labels=range(n_classes), zero_division=0)
            res[k].append(vals[k])
        res["diff"].append(vals[names[0]] - vals[names[1]])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in res.items()}
