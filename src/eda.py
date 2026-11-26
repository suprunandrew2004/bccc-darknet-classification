import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import load_bccc as ld

OUT = "./results/"
PAL = {"Tor": "#7e57c2", "I2P": "#3b7dd8", "ZeroNet": "#5aae5e",
       "Freenet": "#e8b83b", "VPN": "#e8853b"}
BOX = [("fwd_init_win_bytes", "Начальное окно TCP (прямое), байт", False),
       ("bwd_init_win_bytes", "Начальное окно TCP (обратное), байт", False),
       ("packets_count", "Число пакетов в потоке", True),
       ("packets_IAT_mean", "Средний межпакетный интервал, с", True)]
CORR = {"duration": "Длительность", "packets_count": "Пакетов",
        "total_payload_bytes": "Объём нагрузки",
        "payload_bytes_mean": "Ср. нагрузка", "payload_bytes_std": "СКО нагрузки",
        "payload_bytes_max": "Макс. нагрузка", "mean_header_bytes": "Ср. заголовок",
        "fwd_init_win_bytes": "Окно TCP fwd", "bwd_init_win_bytes": "Окно TCP bwd",
        "packets_IAT_mean": "Ср. IAT", "packet_IAT_max": "Макс. IAT",
        "idle_max": "Макс. простой", "active_mean": "Ср. активность",
        "bytes_rate": "Байт/с", "packets_rate": "Пакетов/с",
        "down_up_rate": "Down/Up", "syn_flag_counts": "Число SYN",
        "psh_flag_counts": "Число PSH", "handshake_state": "Сост. рукопожатия",
        "mean_packets_delta_len": "Ср. Δ длины",
        "hex_2_gram_entropy": "Энтропия hex-2",
        "mean_hex_per64bytes_entropy": "Энтропия 64 Б"}


def comma(v):
    if float(v).is_integer():
        return f"{v:,.0f}".replace(",", " ")
    s = f"{v:,.0f}" if abs(v) >= 100 else (f"{v:.2f}" if abs(v) >= 0.1
                                            else f"{v:.3f}")
    return s.replace(",", " ").replace(".", ",")


def main():
    X, y, feats, meta, info = ld.load("multi", verbose=False)
    classes = ld.CLASSES
    keys = ["duration", "packets_count", "total_payload_bytes",
            "payload_bytes_mean", "fwd_init_win_bytes", "bwd_init_win_bytes",
            "packets_IAT_mean"]
    tab = {"median": X[keys].groupby(y).median().loc[classes].to_dict(),
           "mean": X[keys].groupby(y).mean().loc[classes].to_dict(),
           "zero_payload": (X["total_payload_bytes"] == 0).groupby(y).mean()
           .loc[classes].to_dict(),
           "win_top": {c: X.loc[y == c, "fwd_init_win_bytes"].value_counts(
               normalize=True).head(3).to_dict() for c in classes}}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for ax, (f, title, log) in zip(axes.ravel(), BOX):
        data = [X.loc[y == c, f].dropna().values for c in classes]
        if log:
            data = [np.clip(d, 1e-6, None) for d in data]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops={"color": "black", "lw": 1.6})
        for patch, c in zip(bp["boxes"], classes):
            patch.set_facecolor(PAL[c])
            patch.set_alpha(0.8)
        for i, d in enumerate(data):
            md = np.median(d)
            ax.text(i + 1, md, comma(md), ha="center", va="center", fontsize=8,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.15",
                                                 fc="white", ec="gray", lw=0.5))
        if log:
            ax.set_yscale("log")
        ax.set_xticks(range(1, len(classes) + 1), classes)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT + "fig_box.png", dpi=200)
    plt.close(fig)
    cf = [f for f in CORR if f in X.columns]
    C = X[cf].fillna(0).corr(method="spearman").values
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cf)), [CORR[f] for f in cf], rotation=90)
    ax.set_yticks(range(len(cf)), [CORR[f] for f in cf])
    for i in range(len(cf)):
        for j in range(len(cf)):
            if abs(C[i, j]) >= 0.7 and i != j:
                ax.text(j, i, f"{C[i, j]:.1f}".replace(".", ","), ha="center",
                        va="center", fontsize=6.5,
                        color="white" if abs(C[i, j]) > 0.85 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                      label="Коэффициент корреляции Спирмена")
    from matplotlib.ticker import FuncFormatter
    cb.ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}".replace(".", ",")))
    fig.tight_layout()
    fig.savefig(OUT + "fig_corr.png", dpi=200)
    plt.close(fig)
    Cf = X.fillna(0).corr(method="spearman").abs().values
    iu = np.triu_indices_from(Cf, 1)
    tab["share_pairs_r95"] = float(np.nanmean(Cf[iu] > 0.95))
    Z = StandardScaler().fit_transform(
        np.sign(X.fillna(0)) * np.log1p(np.abs(X.fillna(0))))
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(Z)
    tab["pca_explained"] = pca.explained_variance_ratio_.tolist()
    fig, ax = plt.subplots(figsize=(9, 6.6))
    for c in ["VPN", "ZeroNet", "I2P", "Tor", "Freenet"]:
        s = (y == c).values
        ax.scatter(P[s, 0], P[s, 1], s=7, alpha=0.5, color=PAL[c], label=c)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Главная компонента 1 ({ev[0]:.1f}% дисперсии)".replace(".", ","))
    ax.set_ylabel(f"Главная компонента 2 ({ev[1]:.1f}% дисперсии)".replace(".", ","))
    ax.grid(alpha=0.3)
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT + "fig_pca.png", dpi=200)
    plt.close(fig)
    json.dump(tab, open(OUT + "results_eda.json", "w"), ensure_ascii=False,
              indent=1, default=float)
    print(json.dumps(tab, ensure_ascii=False, default=float)[:3000])


if __name__ == "__main__":
    main()
