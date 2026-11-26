import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

import common as cm
import load_bccc as ld

OUT = "./results/"
CL = ld.CLASSES
K = len(CL)


def pair_key(meta):
    return [f"{min(a, b)}|{max(a, b)}" for a, b in zip(meta["src"], meta["dst"])]


def run(Xc, yc, metac, fold_c, variant, random_cv=False):
    Xv, yv, fv, metav, _ = ld.load("multi", verbose=False, **variant)
    yvn = yv.map({c: i for i, c in enumerate(CL)}).values
    ycn = yc.map({c: i for i, c in enumerate(CL)}).values
    if random_cv:
        skf = StratifiedKFold(5, shuffle=True, random_state=cm.SEED)
        fold_v = np.zeros(len(yvn), int)
        for k, (_, te) in enumerate(skf.split(np.zeros(len(yvn)), yvn)):
            fold_v[te] = k
        pos = {r: i for i, r in enumerate(metav["raw_idx"])}
        fold_test = np.array([fold_v[pos[r]] for r in metac["raw_idx"]])
    else:
        pf = dict(zip(pair_key(metac), fold_c))
        fold_v = np.array([pf.get(p, -1) for p in pair_key(metav)])
        fold_test = fold_c
    pos = {r: i for i, r in enumerate(metav["raw_idx"])}
    own = np.array([pos.get(r, -1) for r in metac["raw_idx"]])
    Xtr_all = Xv.reindex(columns=Xc.columns)
    pc = np.zeros((len(ycn), K)); po = np.full((len(ycn), K), np.nan)
    for k in range(5):
        tr = fold_v != k
        m = cm.make_model("XGBoost", K)
        cm.fit_model("XGBoost", m, Xtr_all.values[tr], yvn[tr], K)
        te = fold_test == k
        pc[te] = m.predict_proba(Xc.values[te])
        sel = te & (own >= 0)
        po[sel] = m.predict_proba(Xtr_all.values[own[sel]])
    mc = cm.metrics(ycn, pc, K)
    ok = ~np.isnan(po[:, 0])
    mo = cm.metrics(ycn[ok], po[ok], K)
    return {"n_variant": int(len(yvn)), "f1_clean_test": mc["f1_macro"],
            "f1_own_features_same_flows": mo["f1_macro"], "n_own": int(ok.sum()),
            "freenet_f1_clean": mc["per_class"]["f1"][3]}


VARIANTS = {"C0_clean": ({}, False), "C1_no_iat_fix": ({"fix_iat_artifact": False}, False),
            "C2_no_bg_filter": ({"filter_bg": False}, False),
            "C3_raw": ({"filter_bg": False, "fix_iat_artifact": False}, False),
            "C3_raw_random": ({"filter_bg": False, "fix_iat_artifact": False}, True)}


def main():
    import sys, os
    Xc, yc, fc, metac, _ = ld.load("multi", verbose=False)
    ycn = yc.map({c: i for i, c in enumerate(CL)}).values
    folds = cm.folds_for(ycn, metac["group"].values)
    fold_c = np.zeros(len(ycn), int)
    for k, (_, te) in enumerate(folds):
        fold_c[te] = k
    for name in sys.argv[1:]:
        kw, rnd = VARIANTS[name]
        r = run(Xc, yc, metac, fold_c, kw, random_cv=rnd)
        json.dump(r, open(OUT + f"clean_test_{name}.json", "w"), indent=1)
        print(name, r, flush=True)


if __name__ == "__main__":
    main()
