import json
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import common as cm
import load_bccc as ld

OUT = "./results/"


def env_composition(feats):
    env = ld.env_features(feats)
    parts = {p: [f for f in env if p in f] for p in ld.ENV_PATTERNS}
    return {"n_env": len(env), "by_pattern": {p: len(v) for p, v in parts.items()},
            "groups": pd.Series([ld.feature_group(f) for f in env]).value_counts().to_dict()}


def tor_iscx(meta, y):
    raw = pd.read_csv(ld.MULTI_PATH, low_memory=False, usecols=["flow_id", "label"])
    ts = pd.to_datetime(raw["flow_id"].str.rsplit("_", n=1).str[1])
    raw_n = int(((raw["label"] == "TOR") & ts.dt.year.isin([2015, 2016])).sum())
    clean_n = int(((y == "Tor").values & meta["year"].isin([2015, 2016]).values).sum())
    return {"raw": raw_n, "after_clean": clean_n}


def union_check():
    b = pd.read_csv(ld.BINARY_PATH, low_memory=False)
    m = pd.read_csv(ld.MULTI_PATH, low_memory=False)
    ids_anon = set(b.loc[b["label"] == "Encrypted", "flow_id"])
    ids_reg = set(b.loc[b["label"] == "Non-Encrypted", "flow_id"])
    ids_m = set(m["flow_id"])
    Xb, yb, fb, mb, _ = ld.load("binary", verbose=False)
    Xm, ym, fm, mm, _ = ld.load("multi", verbose=False)
    reg = Xb[(yb == "Regular").values].fillna(-9e99)
    key = lambda X: pd.util.hash_pandas_object(X[fm], index=False)
    common_vec = len(set(key(reg)) & set(key(Xm.fillna(-9e99))))
    lab = m.drop_duplicates("flow_id").set_index("flow_id")["label"]
    a_iscx = b[(b["label"] == "Encrypted")]
    a_ts = pd.to_datetime(a_iscx["flow_id"].str.rsplit("_", n=1).str[1])
    a_iscx = a_iscx[a_ts.dt.year.isin([2015, 2016]).values]
    return {"anon_ids_in_multi": len(ids_anon & ids_m), "anon_ids_total": len(ids_anon),
            "regular_ids_in_multi": len(ids_reg & ids_m),
            "identical_vectors_regular_vs_multi": int(common_vec),
            "iscx_anon_raw_by_multi_label": a_iscx["flow_id"].map(lab).fillna("нет в многоклассовом").value_counts().to_dict(),
            "iscx_anon_raw_n": int(len(a_iscx))}


def threshold_table():
    X, y, feats, meta, _ = ld.load("binary", verbose=False)
    yn = (y == "Anon").astype(int).values
    folds = cm.folds_for(yn, meta["group"].values)
    p, _, _, _ = cm.oof_predict("XGBoost", X, yn, folds, 2)
    pa = p[:, 1]
    rows = {}
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
        pred = pa >= t
        tp = int((pred & (yn == 1)).sum()); fp = int((pred & (yn == 0)).sum())
        rows[str(t)] = {"fpr": fp / int((yn == 0).sum()), "fp": fp,
                        "recall": tp / int((yn == 1).sum()),
                        "precision": tp / max(tp + fp, 1)}
    return rows


def pca_cluster():
    X, y, feats, meta, _ = ld.load("multi", verbose=False)
    Z = StandardScaler().fit_transform(np.sign(X.fillna(0)) * np.log1p(np.abs(X.fillna(0))))
    P = PCA(n_components=2, random_state=42).fit_transform(Z)
    left = P[:, 0] < -10
    pay = X["payload_bytes_max"].fillna(0).values if "payload_bytes_max" in X else None
    return {"n_left": int(left.sum()), "share_left": float(left.mean()),
            "left_zero_payload": float((pay[left] == 0).mean()) if pay is not None else None,
            "rest_zero_payload": float((pay[~left] == 0).mean()) if pay is not None else None,
            "left_by_class": y[left].value_counts().to_dict()}


def main():
    R = {}
    X, y, feats, meta, info = ld.load("multi", verbose=False)
    R["env"] = env_composition(feats)
    R["tor_iscx"] = tor_iscx(meta, y)
    R["pca"] = pca_cluster()
    print(json.dumps(R, ensure_ascii=False, indent=1), flush=True)
    R["union"] = union_check()
    print(json.dumps(R["union"], ensure_ascii=False, indent=1), flush=True)
    if "tau" in sys.argv:
        R["tau"] = threshold_table()
        print(json.dumps(R["tau"], indent=1))
    json.dump(R, open(OUT + "results_review.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
