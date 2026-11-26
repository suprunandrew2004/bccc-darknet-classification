import numpy as np
import pandas as pd

RANDOM_SEED = 42
MULTI_PATH = "./data/MultiTotalDS.csv"
BINARY_PATH = "./data/Binary -2DSCombined.csv"

ID_COLS = ["flow_id", "timestamp", "src_ip", "src_port", "dst_ip",
           "dst_port", "protocol", "label"]
TEXT_NUM_COLS = ["delta_start", "handshake_duration"]

CLASSES = ["Tor", "I2P", "ZeroNet", "Freenet", "VPN"]
BIN_CLASSES = ["Regular", "Anon"]
UNION_CLASSES = ["Regular"] + CLASSES
MULTI_MAP = {"TOR": "Tor", "I2P": "I2P", "ZERONET": "ZeroNet",
             "FREENET": "Freenet", "VPN": "VPN"}
BIN_MAP = {"Non-Encrypted": "Regular", "Encrypted": "Anon"}


CAPTURE_HOST = "172.24.216.110"
BG_PORTS = (22, 3389)


ENV_PATTERNS = ("init_win_bytes", "flag_counts", "flag_percentage",
                "header_bytes", "segment_size_min")
PAYLOAD_PATTERNS = ("entropy", "mutual_information")


FEATURE_GROUPS = [
    ("Энтропия полезной нагрузки",
     lambda f: "entropy" in f or "mutual_information" in f),
    ("Начальное окно TCP", lambda f: "init_win" in f),
    ("Флаги TCP и рукопожатие",
     lambda f: "flag" in f or "handshake" in f or f == "delta_start"),
    ("Приращения длин и интервалов", lambda f: "delta_" in f),
    ("Заголовки", lambda f: "header_bytes" in f),
    ("Межпакетные интервалы", lambda f: "IAT" in f),
    ("Активность и простой",
     lambda f: f.startswith("active_") or f.startswith("idle_")),
    ("Интенсивность и блоки", lambda f: "rate" in f or "bulk" in f),
    ("Размеры полезной нагрузки",
     lambda f: "payload_bytes" in f or "segment_size" in f),
    ("Длительность и объём", lambda f: True),
]


def feature_group(f):
    for name, cond in FEATURE_GROUPS:
        if cond(f):
            return name


def env_features(feats):
    return [f for f in feats if any(p in f for p in ENV_PATTERNS)]


def payload_features(feats):
    return [f for f in feats if any(p in f for p in PAYLOAD_PATTERNS)]


def _group_key(src, dst, how):
    src = src.astype(str).values.astype(object)
    dst = dst.astype(str).values.astype(object)
    if how == "src":
        key = pd.Series(src)
    elif how == "dst":
        key = pd.Series(dst)
    elif how == "pair":
        a, b = np.minimum(src, dst), np.maximum(src, dst)
        key = pd.Series([f"{x}|{y}" for x, y in zip(a, b)])
    else:
        raise ValueError(f"неизвестный способ группировки: {how}")
    return pd.factorize(key)[0]


def _read(kind):
    if kind == "multi":
        df = pd.read_csv(MULTI_PATH, low_memory=False)
        y = df["label"].map(MULTI_MAP)
    elif kind == "binary":
        df = pd.read_csv(BINARY_PATH, low_memory=False)
        y = df["label"].map(BIN_MAP)
    elif kind == "union":
        b = pd.read_csv(BINARY_PATH, low_memory=False)
        m = pd.read_csv(MULTI_PATH, low_memory=False)
        b = b[b["label"] == "Non-Encrypted"]
        df = pd.concat([b, m[b.columns]], ignore_index=True)
        y = df["label"].map({**MULTI_MAP, "Non-Encrypted": "Regular"})
    else:
        raise ValueError(kind)
    return df, y


def is_background(df):
    return ((df["dst_ip"].astype(str) == CAPTURE_HOST)
            & df["dst_port"].isin(BG_PORTS)).values


def fix_iat(X):
    cols = [c for c in X.columns if "IAT" in c]
    bad = X[cols].gt(X["duration"] + 1e-6, axis=0)
    X[cols] = X[cols].mask(bad)
    return X, int(bad.any(axis=1).sum())


def load(kind="multi", filter_bg=True, fix_iat_artifact=True, dedup=True,
         group_by="pair", conflicts="first", verbose=True):
    df, y = _read(kind)
    info = {"kind": kind, "n_raw": len(df),
            "raw_counts": y.value_counts().to_dict()}
    ts = pd.to_datetime(df["flow_id"].str.extract(
        r"_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.?\d*)$")[0],
        errors="coerce", format="mixed")
    meta = pd.DataFrame({
        "src": df["src_ip"].astype(str).values,
        "dst": df["dst_ip"].astype(str).values,
        "dport": df["dst_port"].values,
        "ts": ts.values,
        "year": ts.dt.year.values,
        "raw_label": df["label"].astype(str).values,
        "background": is_background(df),
        "group": _group_key(df["src_ip"], df["dst_ip"], group_by),
        "raw_idx": np.arange(len(df)),
    })
    feats = [c for c in df.columns if c not in ID_COLS]
    X = df[feats].copy()
    for c in TEXT_NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan)
    if fix_iat_artifact:
        X, info["n_iat_fixed"] = fix_iat(X)
    info["n_background"] = int(meta["background"].sum())
    info["background_by_class"] = y[meta["background"]].value_counts().to_dict()
    if filter_bg:
        keep = ~meta["background"].values
        X, y, meta = X[keep], y[keep], meta[keep]

    na_share = X.isna().mean()
    sparse = na_share[na_share >= 1.0].index.tolist()
    X = X.drop(columns=sparse)
    info["dropped_sparse"] = sparse
    info["rows_with_nan"] = int(X.isna().any(axis=1).sum())
    info["cols_with_nan"] = int((X.isna().sum() > 0).sum())
    if dedup:
        dup = X.duplicated()
        info["n_dup"] = int(dup.sum())
        info["dup_by_class"] = y[dup].value_counts().to_dict()
        key = pd.util.hash_pandas_object(X.fillna(-9.99e99), index=False)
        lab_per_key = pd.Series(y.values).groupby(key.values).nunique()
        confl = lab_per_key[lab_per_key > 1].index
        info["n_conflict_vectors"] = int(len(confl))
        info["n_conflict_rows"] = int(np.isin(key.values, confl).sum())
        in_confl = np.isin(key.values, confl)
        if conflicts == "majority":
            lab = pd.Series(y.values, index=key.values)
            major = {k: lab.loc[[k]].value_counts(sort=False).idxmax()
                     for k in confl}
            yv = y.values.copy()
            first = (~dup.values) & in_confl
            yv[first] = [major[k] for k in key.values[first]]
            info["n_conflict_relabeled"] = int((yv != y.values).sum())
            y = pd.Series(yv, index=y.index)
        keep = ~dup.values
        if conflicts == "drop":
            keep &= ~in_confl
        info["n_conflict_kept"] = int((keep & in_confl).sum())
        X, y, meta = X[keep], y[keep], meta[keep]
    nun = X.nunique(dropna=False)
    const = nun[nun <= 1].index.tolist()
    X = X.drop(columns=const)
    info["dropped_const"] = const
    feats = list(X.columns)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    info.update(n_features=len(feats), n_after_clean=len(X),
                class_counts=y.value_counts().to_dict(),
                n_groups=int(meta["group"].nunique()), group_by=group_by)
    if verbose:
        print(f"[{kind}] исходно {info['n_raw']}, фоновых {info['n_background']}"
              f", дублей {info.get('n_dup', 0)}, осталось {len(X)} потоков, "
              f"{len(feats)} признаков; классы {info['class_counts']}")
    return X, y, feats, meta, info


def group_diagnostics(y, groups):
    d = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(groups)})
    rows = []
    for c in d["y"].unique():
        sizes = d[d["y"] == c].groupby("g").size().sort_values(ascending=False)
        rows.append({"class": c, "n_flows": int(sizes.sum()),
                     "n_groups": int(len(sizes)),
                     "max_group_share": float(sizes.iloc[0] / sizes.sum()),
                     "top5_share": float(sizes.iloc[:5].sum() / sizes.sum())})
    return pd.DataFrame(rows)


def undersample(X, y, meta, seed=RANDOM_SEED):
    n = y.value_counts().min()
    idx = np.concatenate([
        y[y == c].sample(n, random_state=seed).index.values
        for c in sorted(y.unique())])
    idx = np.random.RandomState(seed).permutation(idx)
    return (X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True),
            meta.loc[idx].reset_index(drop=True))


if __name__ == "__main__":
    for k in ("multi", "binary"):
        X, y, feats, meta, info = load(k)
        print({a: b for a, b in info.items() if a not in ("dropped_const",)})
        print(group_diagnostics(y, meta["group"]).to_string(index=False))
