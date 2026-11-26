import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import common as cm
import load_bccc as lb

OUT = "./results/"
L1_PATH = "./data_ss/SafeSurf Dataset Layer 1.csv"
L23_PATH = "./data_ss/SafeSurf Dataset Layer 2 & 3.csv"
BCCC_MULTI = "./data/MultiTotalDS.csv"

ID_COLS = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
           "Label", "Label_2"]
CLASSES = ["Tor", "I2P", "Zeronet", "Freenet", "VPN"]
HOSTS_2020 = ("172.24.216.110", "10.0.2.15")
BG_PORTS = (22, 3389)
KEY = ["Src IP", "Src Port", "Dst IP", "Dst Port"]
COMMON = {"dur": ("duration", "Flow Duration", 1e-6),
          "pkts": ("packets_count", ("Tot Fwd Pkts", "Tot Bwd Pkts"), 1),
          "bytes": ("total_payload_bytes", ("TotLen Fwd Pkts", "TotLen Bwd Pkts"), 1),
          "len_max": ("payload_bytes_max", "Pkt Len Max", 1),
          "len_mean": ("payload_bytes_mean", "Pkt Len Mean", 1),
          "len_std": ("payload_bytes_std", "Pkt Len Std", 1),
          "pkt_rate": ("packets_rate", "Flow Pkts/s", 1),
          "byte_rate": ("bytes_rate", "Flow Byts/s", 1),
          "iat_mean": ("packets_IAT_mean", "Flow IAT Mean", 1e-6),
          "iat_std": ("packet_IAT_std", "Flow IAT Std", 1e-6),
          "iat_max": ("packet_IAT_max", "Flow IAT Max", 1e-6),
          "iat_min": ("packet_IAT_min", "Flow IAT Min", 1e-6)}


def read(path):
    df = pd.read_csv(path, low_memory=False)
    df["Src IP"] = df["Src IP"].astype(str)
    df["Dst IP"] = df["Dst IP"].astype(str)
    return df


def pair_groups(df):
    s, d = df["Src IP"].values.astype(object), df["Dst IP"].values.astype(object)
    a, b = np.minimum(s, d), np.maximum(s, d)
    return pd.factorize(pd.Series([f"{x}|{y}" for x, y in zip(a, b)]))[0]


def features(df):
    X = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
    return X.replace([np.inf, -np.inf], np.nan)


def clean(df, info, tag):
    info[tag] = {"n_raw": int(len(df)),
                 "raw_counts": df["Label"].value_counts().to_dict()}
    bg = (df["Dst IP"].isin(HOSTS_2020) & df["Dst Port"].isin(BG_PORTS)).values
    info[tag]["n_background"] = int(bg.sum())
    df = df[~bg].reset_index(drop=True)
    X = features(df)
    dup = X.duplicated()
    key = pd.util.hash_pandas_object(X.fillna(-9.99e99), index=False)
    lab = df["Label"].values
    nun = pd.Series(lab).groupby(key.values).nunique()
    info[tag]["n_dup"] = int(dup.sum())
    info[tag]["n_conflict_vectors"] = int((nun > 1).sum())
    df, X = df[~dup.values].reset_index(drop=True), X[~dup.values].reset_index(drop=True)
    const = [c for c in X if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=const)
    info[tag].update(n_after=int(len(X)), n_features=int(X.shape[1]),
                     dropped_const=const,
                     class_counts=df["Label"].value_counts().to_dict())
    return df, X


def run(X, y, folds, k):
    p, f1, ba, t = cm.oof_predict("XGBoost", X, y, folds, k)
    m = cm.metrics(y, p, k)
    return {"f1_macro": m["f1_macro"], "bacc": m["bacc"],
            "accuracy": m["accuracy"], "per_class_f1": m["per_class"]["f1"],
            "per_class_precision": m["per_class"]["precision"],
            "per_class_recall": m["per_class"]["recall"],
            "fold_f1_mean": float(f1.mean()), "fold_f1_std": cm.sd(f1),
            "n": int(len(y)), "time_s": t}


def random_folds(y):
    skf = StratifiedKFold(5, shuffle=True, random_state=cm.SEED)
    return list(skf.split(np.zeros(len(y)), y))


def overlap(R):
    ss = read(L23_PATH)
    bc = pd.read_csv(BCCC_MULTI, low_memory=False,
                     usecols=["src_ip", "src_port", "dst_ip", "dst_port", "label"])
    bc = bc.rename(columns={"src_ip": "Src IP", "src_port": "Src Port",
                            "dst_ip": "Dst IP", "dst_port": "Dst Port"})
    bc["Src IP"], bc["Dst IP"] = bc["Src IP"].astype(str), bc["Dst IP"].astype(str)
    k = ["Src IP", "Src Port", "Dst IP", "Dst Port"]
    j = bc.merge(ss[k + ["Label"]], on=k)
    hosts = (ss["Src IP"].isin(HOSTS_2020) | ss["Dst IP"].isin(HOSTS_2020))
    R["overlap"] = {
        "n_tuples": int(len(j[k].drop_duplicates())), "n_rows": int(len(j)),
        "crosstab": pd.crosstab(j["label"], j["Label"]).to_dict(),
        "agree_share": float((j["label"].str.lower() == j["Label"].str.lower()).mean()),
        "rows_2020_hosts": int(hosts.sum()),
        "rows_2020_hosts_by_class": ss[hosts]["Label"].value_counts().to_dict(),
        "protocol_by_class": pd.crosstab(ss["Label"], ss["Protocol"]).to_dict()}
    print("overlap", R["overlap"]["n_tuples"], R["overlap"]["agree_share"], flush=True)


def parse_ts(s):
    s = s.astype(str)
    ampm = s.str.endswith("M")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[ampm] = pd.to_datetime(s[ampm], format="%d/%m/%Y %I:%M:%S %p")
    out[~ampm] = pd.to_datetime(s[~ampm], format="%m/%d/%Y %H:%M")
    return out


def matched():
    ss = read(L23_PATH)
    bc = pd.read_csv(BCCC_MULTI, low_memory=False)
    bc = bc.rename(columns={"src_ip": "Src IP", "src_port": "Src Port",
                            "dst_ip": "Dst IP", "dst_port": "Dst Port"})
    bc["Src IP"], bc["Dst IP"] = bc["Src IP"].astype(str), bc["Dst IP"].astype(str)
    j = bc.merge(ss, on=KEY, suffixes=("_b", "_s"))
    j["t_b"] = pd.to_datetime(j["flow_id"].str.rsplit("_", n=1).str[1])
    j["t_s"] = parse_ts(j["Timestamp"])
    j["pk_s"] = j["Tot Fwd Pkts"] + j["Tot Bwd Pkts"]
    j["exact"] = (((j["Flow Duration"] - (j["duration"] * 1e6).round()).abs() <= 1)
                  & (j["pk_s"] == j["packets_count"]))
    return j


def evidence(R):
    j = matched()
    shift = ((j["t_s"].dt.hour * 60 + j["t_s"].dt.minute)
             - (j["t_b"].dt.hour * 60 + j["t_b"].dt.minute))
    back = j["t_s"] - pd.Timedelta(hours=7)
    same_day = (((back.dt.month == j["t_b"].dt.month) & (back.dt.day == j["t_b"].dt.day))
                | ((back.dt.month == j["t_b"].dt.day) & (back.dt.day == j["t_b"].dt.month)))
    j["shift7"] = shift.between(415, 425) & same_day
    first = j.sort_values("t_s").groupby(KEY).first()
    per = j.groupby(KEY)["exact"].any()
    ex = j[j["exact"]].drop_duplicates("flow_id")
    iat_b = [c for c in ex.columns if "IAT" in c and c[0].islower()]
    iat_s = [c for c in ex.columns if "IAT" in c and c[0].isupper()]
    art = (ex[iat_b].astype(float) > 1e9).any(axis=1)
    a = ex[art]
    stamp = pd.to_datetime(a[iat_b].astype(float).max(axis=1), unit="s")
    R["evidence"] = {
        "n_tuples": int(len(first)),
        "years_bccc": j["t_b"].dt.year.value_counts().to_dict(),
        "years_ss": j["t_s"].dt.year.value_counts().to_dict(),
        "shift7_share_tuples": float(first["shift7"].mean()),
        "exact_tuples": int(per.sum()),
        "exact_share": float(per.mean()),
        "exact_flows": int(len(ex)),
        "capture_host_tuples": int(first.reset_index()[["Src IP", "Dst IP"]]
                                   .isin(HOSTS_2020).any(axis=1).sum()),
        "rdp_ssh_tuples": int(first.reset_index().eval(
            "`Dst Port` in [22, 3389] or `Src Port` in [22, 3389]").sum()),
        "iat_artifact_flows": int(art.sum()),
        "iat_artifact_stamp_4h": float(((stamp - a["t_b"]).dt.total_seconds()
                                        .between(14399, 14401)).mean()),
        "iat_ss_above_duration": int((a[iat_s].max(axis=1) > a["Flow Duration"]).sum()),
        "examples": ex[ex["packets_count"] >= 15][KEY + [
            "t_b", "Timestamp", "duration", "Flow Duration", "packets_count",
            "pk_s", "label", "Label"]].drop_duplicates(KEY)
        .sample(6, random_state=cm.SEED).astype(str).to_dict("records")}
    print("evidence", R["evidence"]["shift7_share_tuples"],
          R["evidence"]["exact_tuples"], flush=True)


def cic_defects(R):
    raw = read(L23_PATH)
    Xr = raw.drop(columns=[c for c in ID_COLS if c in raw.columns])
    Xr = Xr.apply(pd.to_numeric, errors="coerce")
    iat = [c for c in Xr if "IAT" in c]
    tcp = raw[(raw["Protocol"] == 6) & (raw["Tot Fwd Pkts"] + raw["Tot Bwd Pkts"] >= 3)]
    flags = ["FIN Flag Cnt", "SYN Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
             "RST Flag Cnt"]
    same = [(a, b) for i, a in enumerate(Xr.columns) for b in Xr.columns[i + 1:]
            if Xr[a].equals(Xr[b]) and Xr[a].nunique() > 1]
    dur = Xr["Flow Duration"] / 1e6
    R["cic_defects"] = {
        "inf_rows": int(np.isinf(Xr).any(axis=1).sum()),
        "nan_rows": int(Xr.isna().any(axis=1).sum()),
        "negative_rows": int((Xr[["Flow Duration"] + iat] < 0).any(axis=1).sum()),
        "iat_above_duration_rows": int((Xr[iat].max(axis=1) > Xr["Flow Duration"]).sum()),
        "iat_max_s": float(Xr[iat].max().max() / 1e6),
        "identical_columns": same,
        "duration_q99_s": float(dur.quantile(0.99)),
        "duration_over_120s": int((dur > 120).sum()),
        "flag_max": {f: int(tcp[f].max()) for f in flags},
        "flag_zero_share_tcp3": {f: float((tcp[f] == 0).mean()) for f in flags},
        "n_tcp3": int(len(tcp))}
    print("defects", R["cic_defects"]["inf_rows"], R["cic_defects"]["flag_max"], flush=True)


def common_bccc(X):
    return pd.DataFrame({k: X[b] for k, (b, _, _) in COMMON.items()})


def common_ss(df):
    out = {}
    for k, (_, s, mult) in COMMON.items():
        v = df[list(s)].sum(axis=1) if isinstance(s, tuple) else df[s]
        out[k] = pd.to_numeric(v, errors="coerce").astype(float) * mult
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def transfer(R):
    Xb, yb, _, meta, _ = lb.load("multi", verbose=False)
    cl = lb.CLASSES
    Xb = common_bccc(Xb)
    yb = yb.map({c: i for i, c in enumerate(cl)}).values
    df, _ = clean(read(L23_PATH), {}, "l2")
    Xs = common_ss(df)
    ys = df["Label"].replace({"Zeronet": "ZeroNet"}).map(
        {c: i for i, c in enumerate(cl)}).values
    new = ~(df["Src IP"].isin(HOSTS_2020) | df["Dst IP"].isin(HOSTS_2020)).values
    new4 = new & (df["Label"] != "I2P").values

    def score(y, p):
        lab = sorted(set(y))
        return {"f1_macro": float(f1_score(y, p, labels=lab, average="macro")),
                "bacc": float(balanced_accuracy_score(y, p)), "n": int(len(y)),
                "per_class_f1": dict(zip([cl[i] for i in lab], f1_score(
                    y, p, labels=lab, average=None).round(3).tolist()))}

    def fit_predict(Xtr, ytr, Xte):
        mdl = cm.make_model("XGBoost", 5)
        cm.fit_model("XGBoost", mdl, Xtr.values, ytr, 5)
        return mdl.predict(Xte.values)

    pb, _, _, _ = cm.oof_predict("XGBoost", Xb, yb, cm.folds_for(yb, meta["group"].values), 5)
    ps, _, _, _ = cm.oof_predict("XGBoost", Xs, ys, cm.folds_for(ys, pair_groups(df)), 5)
    R["transfer"] = {
        "features": list(COMMON),
        "bccc_group": score(yb, pb.argmax(1)),
        "ss_group": score(ys, ps.argmax(1)),
        "bccc_to_ss_new": score(ys[new4], fit_predict(Xb, yb, Xs[new4])),
        "bccc_to_ss_all": score(ys, fit_predict(Xb, yb, Xs)),
        "ss_new_to_bccc": score(yb, fit_predict(Xs[new], ys[new], Xb))}
    print("transfer", {k: round(v["f1_macro"], 3) for k, v in R["transfer"].items()
                       if isinstance(v, dict)}, flush=True)


def layer2(R):
    info = {}
    raw = read(L23_PATH)
    yr = raw["Label"].map({c: i for i, c in enumerate(CLASSES)}).values
    Xr = features(raw)
    R["l2_random_raw"] = run(Xr, yr, random_folds(yr), 5)
    print("L2 random raw", R["l2_random_raw"]["f1_macro"], flush=True)
    df, X = clean(raw, info, "l2")
    y = df["Label"].map({c: i for i, c in enumerate(CLASSES)}).values
    g = pair_groups(df)
    info["l2"]["n_groups"] = int(len(np.unique(g)))
    R["l2_info"] = info["l2"]
    R["l2_random_clean"] = run(X, y, random_folds(y), 5)
    print("L2 random clean", R["l2_random_clean"]["f1_macro"], flush=True)
    R["l2_group"] = run(X, y, cm.folds_for(y, g), 5)
    print("L2 group", R["l2_group"]["f1_macro"], flush=True)
    new = ~(df["Src IP"].isin(HOSTS_2020) | df["Dst IP"].isin(HOSTS_2020)).values
    keep = new & (df["Label"] != "I2P").values
    cl4 = ["Tor", "Zeronet", "Freenet", "VPN"]
    y4 = df["Label"][keep].map({c: i for i, c in enumerate(cl4)}).values
    X4 = X[keep].reset_index(drop=True)
    R["l2_new_hosts"] = run(X4, y4, cm.folds_for(y4, g[keep]), 4)
    R["l2_new_hosts"]["classes"] = cl4
    R["l2_new_hosts"]["class_counts"] = df["Label"][keep].value_counts().to_dict()
    print("L2 new hosts", R["l2_new_hosts"]["f1_macro"], flush=True)
    old = ~new
    st = old.astype(int)
    p, f1, ba, _ = cm.oof_predict("XGBoost", X, st, cm.folds_for(st, g), 2)
    R["l2_stand_bacc"] = float(balanced_accuracy_score(st, p.argmax(1)))
    tor = (df["Label"] == "Tor").values
    res = {}
    for name, test in (("new_to_old", tor & old), ("old_to_new", tor & new)):
        tr = np.where(~test)[0]
        mdl = cm.make_model("XGBoost", 5)
        cm.fit_model("XGBoost", mdl, X.values[tr], y[tr], 5)
        pr = mdl.predict(X.values[test])
        res[name] = {"n": int(test.sum()),
                     "pred": {CLASSES[k]: int((pr == k).sum()) for k in range(5)}}
    R["l2_tor_cross_host"] = res
    print("stand", R["l2_stand_bacc"], res, flush=True)


def layer1(R):
    info = {}
    raw = read(L1_PATH)
    yr = (raw["Label"] == "Darknet").astype(int).values
    R["l1_random_raw"] = run(features(raw), yr, random_folds(yr), 2)
    print("L1 random raw", R["l1_random_raw"]["f1_macro"], flush=True)
    df, X = clean(raw, info, "l1")
    y = (df["Label"] == "Darknet").astype(int).values
    g = pair_groups(df)
    R["l1_info"] = info["l1"]
    R["l1_random_clean"] = run(X, y, random_folds(y), 2)
    print("L1 random clean", R["l1_random_clean"]["f1_macro"], flush=True)
    R["l1_group"] = run(X, y, cm.folds_for(y, g), 2)
    host = (df["Src IP"].eq("192.168.1.76") | df["Dst IP"].eq("192.168.1.76")).values
    R["l1_normal_host_share"] = float(host[y == 0].mean())
    R["l1_darknet_host_share"] = float(host[y == 1].mean())
    print("L1 group", R["l1_group"]["f1_macro"], flush=True)


def fig_compare(R):
    bc = json.load(open(OUT + "results_schemes.json"))["S3"]["XGBoost"]
    a = bc["per_class"]["f1"]
    b = R["l2_group"]["per_class_f1"]
    lab = ["Tor", "I2P", "ZeroNet", "Freenet", "VPN"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(lab))
    w = 0.38
    for off, v, col, name in ((-w / 2, a, "#3b7dd8", "BCCC-Darknet-2025 (NTLFlowLyzer)"),
                              (w / 2, b, "#e8853b", "SafeSurf Darknet 2025 (CICFlowMeter)")):
        ax.bar(x + off, v, w * 0.94, color=col, label=name)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.015, f"{vi:.2f}".replace(".", ","), ha="center",
                    fontsize=9)
    ax.set_xticks(x, lab)
    ax.set_ylabel("F1-мера класса")
    ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=False)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT + "fig_safesurf.png", dpi=200)
    plt.close(fig)


def main():
    R = json.load(open(OUT + "results_safesurf.json")) if len(sys.argv) > 1 else {}
    todo = sys.argv[1:] or ["overlap", "evidence", "defects", "l2", "l1", "transfer"]
    if "overlap" in todo:
        overlap(R)
    if "evidence" in todo:
        evidence(R)
    if "defects" in todo:
        cic_defects(R)
    if "l2" in todo:
        layer2(R)
    json.dump(R, open(OUT + "results_safesurf.json", "w"), ensure_ascii=False,
              indent=1, default=str)
    if "fig" in todo or "l2" in todo:
        fig_compare(R)
    if "l1" in todo:
        layer1(R)
    if "transfer" in todo:
        transfer(R)
    json.dump(R, open(OUT + "results_safesurf.json", "w"), ensure_ascii=False,
              indent=1, default=str)
    print("готово")


if __name__ == "__main__":
    main()
