import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

OUT = "./results/"

NAMES = {
    "fwd_init_win_bytes": "Нач. окно TCP (fwd)",
    "bwd_init_win_bytes": "Нач. окно TCP (bwd)",
    "duration": "Длительность потока",
    "handshake_state": "Состояние рукопожатия",
    "handshake_duration": "Длительность рукопожатия",
    "delta_start": "Задержка начала передачи",
    "packets_count": "Число пакетов",
    "fwd_packets_count": "Число пакетов fwd",
    "bwd_packets_count": "Число пакетов bwd",
    "down_up_rate": "Отношение Down/Up",
    "fwd_mean_header_bytes": "Ср. длина заголовка (fwd)",
    "bwd_mean_header_bytes": "Ср. длина заголовка (bwd)",
    "mean_header_bytes": "Ср. длина заголовка",
    "packet_IAT_min": "Мин. IAT",
    "packet_IAT_max": "Макс. IAT",
    "packets_IAT_median": "Медиана IAT",
    "cwr_flag_counts": "Число флагов CWR",
    "mode_packets_delta_time": "Мода Δ времени прихода",
    "median_packets_delta_time": "Медиана Δ времени прихода",
    "mode_fwd_packets_delta_time": "Мода Δ времени прихода (fwd)",
    "fwd_packets_rate": "Пакетов в секунду (fwd)",
    "skewness_utf8_per16bytes_entropy": "Асимм. энтропии 16-байтных блоков",
    "binary_3_gram_entropy": "Энтропия 3-грамм (двоичн.)",
}
PARTS = [("fwd_", "fwd "), ("bwd_", "bwd "), ("payload_bytes", "нагрузка"),
         ("header_bytes", "заголовок"), ("segment_size", "сегмент"),
         ("packets_IAT", "IAT"), ("packet_IAT", "IAT"), ("delta_len", "Δ длины"),
         ("delta_time", "Δ времени"), ("_entropy", " энтропия"),
         ("flag_counts", "флагов"), ("flag_percentage_in_total", "доля флагов"),
         ("mean", "ср."), ("median", "медиана"), ("mode", "мода"),
         ("variance", "дисп."), ("std", "СКО"), ("skewness", "асимм."),
         ("cov", "коэф. вар."), ("max", "макс."), ("min", "мин."),
         ("total", "сумма"), ("rate", "интенс.")]


def feat_name(f):
    if f in NAMES:
        return NAMES[f]
    s = f
    for a, b in PARTS:
        s = s.replace(a, b)
    return s.replace("_", " ").strip()


def box(ax, x, y, w, h, text, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="black", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5,
            fontweight="bold")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color="black"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    labels = ["Захват\nтрафика\n(PCAP)", "Разбор на\nпотоки (flow)",
              "Извлечение\nпризнаков\n(NTLFlowLyzer)", "Очистка и\nпредобработка"]
    w, h, y = 1.75, 1.2, 1.1
    xs = [0.1, 2.25, 4.4, 6.55]
    for x, t in zip(xs, labels):
        box(ax, x, y, w, h, t)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y + h / 2), (b, y + h / 2))
    box(ax, 8.7, y, 1.9, h, "Уровень 1:\nобычный /\nанонимизир.", fc="#eaf1fb")
    arrow(ax, (6.55 + w, y + h / 2), (8.7, y + h / 2))
    box(ax, 11.0, 2.0, 1.9, 1.2, "Уровень 2:\nTor, I2P, ZeroNet,\nFreenet, VPN",
        fc="#fdf0e6")
    box(ax, 11.0, 0.15, 1.9, 1.0, "Обычный\nтрафик", fc="#f2f2f2")
    arrow(ax, (10.6, y + h * 0.75), (11.0, 2.6))
    arrow(ax, (10.6, y + h * 0.25), (11.0, 0.65))
    ax.text(10.55, 2.55, "Anon", fontsize=9, ha="right")
    ax.text(10.6, 0.6, "Regular", fontsize=9, ha="right")
    fig.tight_layout()
    fig.savefig(OUT + "fig_pipeline.png", dpi=200)
    plt.close(fig)


def fig_shap(path=OUT + "results_extra.json", n=15):
    S = json.load(open(path))["shap"]["mean_abs"]
    items = list(S.items())[:n]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(range(len(items)), [v for _, v in items], color="#e8853b")
    ax.set_yticks(range(len(items)), [feat_name(k) for k, _ in items], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Среднее |значение Шепли| (логарифм шансов, сумма по классам)")
    ax.grid(alpha=0.3, axis="x")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    fig.tight_layout()
    fig.savefig(OUT + "fig_shap.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    try:
        fig_shap()
    except (FileNotFoundError, KeyError):
        print("нет результатов SHAP")
