import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sch import S, top, bot, left, right, tree_link

OUT = "./results/schemes/"


def fig1():
    p, h = 1.75, 1.5
    W = 16.5
    y0 = 0.2 + h / 2
    leaves_y = [y0 + k * p for k in (0, 1.5, 2.5, 3.5, 4.5)]
    H = leaves_y[-1] + h / 2 + 0.2
    s = S(W, H)
    c2 = [y0 + 0 * p, y0 + 2 * p, y0 + 4 * p]
    root = s.rect(1.9, y0 + 2 * p, 3.4, h, "Методы\nклассификации\nсетевого трафика")
    w2, x2 = 3.9, 4.5 + 3.9 / 2
    n = [s.rect(x2, c2[0], w2, h, "По номерам портов"),
         s.rect(x2, c2[1], w2, h, "Глубокая инспекция\nпакетов (DPI)"),
         s.rect(x2, c2[2], w2, h, "Статистический\nанализ метаданных")]
    w3, x3 = 6.9, W - 0.15 - 6.9 / 2
    L = ["Соответствие порта и сервиса:\nненадёжен при динамических портах\nи туннелировании",
         "Сигнатуры в содержимом пакетов:\nнеприменимы к зашифрованной\nнагрузке",
         "Отпечатки рукопожатия TLS (JA3):\nтолько нешифруемая часть\nсоединения",
         "Потоковые признаки (flow-based)\nи машинное обучение:\nприменимы без дешифровки",
         "Последовательности пакетов\nи глубокое обучение"]
    lv = [s.rect(x3, y, w3, h, t, bold=(i == 3)) for i, (y, t) in enumerate(zip(leaves_y, L))]
    tree_link(s, root, n)
    tree_link(s, n[0], [lv[0]])
    tree_link(s, n[1], lv[1:3])
    tree_link(s, n[2], lv[3:5])
    s.save(OUT + "fig1.png")


def fig2():
    p, h = 1.2, 0.95
    y = lambda r: 0.2 + h / 2 + r * p
    H = y(9.5) + h / 2 + 0.2
    W = 16.5
    s = S(W, H)
    ws = [2.6, 4.0, 4.0, 3.4]
    gap = (W - 0.3 - sum(ws)) / 3
    xs, x = [], 0.15
    for w in ws:
        xs.append(x + w / 2); x += w + gap
    root = s.rect(xs[0], (y(1.5) + y(9)) / 2, ws[0], h, "Машинное\nобучение")
    c2 = [s.rect(xs[1], y(1.5), ws[1], h, "Классическое\nобучение"),
          s.rect(xs[1], y(4), ws[1], h, "Обучение\nс подкреплением"),
          s.rect(xs[1], y(6.5), ws[1], h, "Ансамблевые\nметоды"),
          s.rect(xs[1], y(9), ws[1], h, "Нейронные сети\nи глубокое обучение")]
    sup = s.rect(xs[2], y(0.5), ws[2], h, "С учителем", bold=True)
    uns = s.rect(xs[2], y(2.5), ws[2], h, "Без учителя")
    ens = [s.rect(xs[2], y(5.5), ws[2], h, "Бэггинг\n(Random Forest)", bold=True),
           s.rect(xs[2], y(6.5), ws[2], h, "Бустинг\n(XGBoost)", bold=True),
           s.rect(xs[2], y(7.5), ws[2], h, "Стекинг")]
    nn = [s.rect(xs[2], y(8.5), ws[2], h, "Многослойный\nперцептрон", bold=True),
          s.rect(xs[2], y(9.5), ws[2], h, "Свёрточные\nи рекуррентные сети")]
    l4 = [s.rect(xs[3], y(r), ws[3], h, t, bold=(r == 0)) for r, t in
          ((0, "Классификация"), (1, "Регрессия"), (2, "Кластеризация"), (3, "Снижение\nразмерности"))]
    tree_link(s, root, c2)
    tree_link(s, c2[0], [sup, uns])
    tree_link(s, c2[2], ens)
    tree_link(s, c2[3], nn)
    tree_link(s, sup, l4[:2])
    tree_link(s, uns, l4[2:])
    s.save(OUT + "fig2.png")


def fig3():
    W = 15.0
    w, h, g = 7.6, 1.15, 0.6
    xc = W / 2
    ys = [0.2 + h / 2 + k * (h + g) for k in range(5)]
    ho = 1.35
    yo = ys[-1] + h / 2 + 1.3 + ho / 2
    H = yo + ho / 2 + 0.2
    s = S(W, H)
    T = ["Захват трафика (PCAP)", "Разбор на потоки (flow)",
         "Извлечение признаков (NTLFlowLyzer)", "Очистка и предобработка",
         "Уровень 1: обычный /\nанонимизированный трафик"]
    b = [s.rect(xc, y, w, h, t) for y, t in zip(ys, T)]
    for a, c in zip(b, b[1:]):
        s.line([bot(a), top(c)])
    wo = 6.4
    o1 = s.rect(xc - 3.6, yo, wo, ho, "Обычный трафик")
    o2 = s.rect(xc + 3.6, yo, wo, ho, "Уровень 2: Tor, I2P,\nZeroNet, Freenet, VPN")
    ymid = (bot(b[-1])[1] + top(o1)[1]) / 2
    s.line([bot(b[-1]), (xc, ymid)], arrow=False)
    s.line([(o1["x"], ymid), (o2["x"], ymid)], arrow=False)
    s.line([(o1["x"], ymid), top(o1)])
    s.line([(o2["x"], ymid), top(o2)])
    s.text(o1["x"] - 0.25, ymid + 0.32, "Regular", ha="right")
    s.text(o2["x"] + 0.25, ymid + 0.32, "Anon", ha="left")
    s.save(OUT + "fig3.png")


def column(s, x, y, items, gap):
    """Последовательность фигур одного столбца; возвращает фигуры."""
    out = []
    for kind, w, h, t in items:
        y += h / 2
        out.append(getattr(s, kind)(x, y, w, h, t))
        y += h / 2 + gap
    for a, c in zip(out, out[1:]):
        s.line([bot(a), top(c)])
    return out


def fig4():
    w, g = 13.6, 0.30
    items = [("term", 3.0, 0.75, "Начало"),
             ("io", w, 1.1, "1. Входные данные:\nпотоковые признаки записей трафика (CSV)"),
             ("rect", w, 1.1, "2. Формирование входных данных: проверка набора, исключение\nфоновых соединений, исправление артефактов, удаление повторов"),
             ("rect", w, 1.1, "3. Предварительная обработка: неопределённые значения,\nвеса классов, групповое разбиение, стандартизация"),
             ("lstart", w, 0.65, "Цикл: для каждой модели и каждого блока i = 1, …, 5"),
             ("rect", w, 1.1, "4. Обучение: модели первого уровня (Regular / Anon)\nи второго уровня (пять технологий)"),
             ("lend", w, 0.65, "Конец цикла"),
             ("rect", w, 1.1, "5. Предварительные результаты и их обработка: метрики\nпо объединённым прогнозам блоков, сравнение моделей"),
             ("rect", w, 1.1, "6. Обоснование схемы оценивания и параметров: влияние\nразбиения, очистки, взвешивания, объёма выборки"),
             ("io", w, 1.1, "7. Итоговый результат: двухуровневый\nклассификатор, оценки важности признаков"),
             ("term", 3.0, 0.75, "Конец")]
    H = 0.4 + sum(i[2] for i in items) + g * (len(items) - 1)
    W = w + 0.4
    s = S(W, H)
    column(s, W / 2, 0.2, items, g)
    s.save(OUT + "fig4.png")


def branch(s, d, side, label_yes="да", label_no="нет", join_y=None, main_next_top=None):
    """Ветвь «да» вправо к боковой фигуре и возврат на основную линию."""
    s.line([right(d), left(side)])
    s.text(right(d)[0] + 0.12, d["y"] - 0.22, label_yes, ha="left", size=10)
    s.text(d["x"] + 0.15, bot(d)[1] + 0.25, label_no, ha="left", size=10)
    s.line([bot(side), (side["x"], join_y), (d["x"] + 0.0, join_y)])


def fig5():
    W = 16.5
    xm, wm = 4.55, 8.5
    xs, wsd = 12.75, 6.6
    g = 0.42
    s = S(W, 22)
    y = 0.2
    def put(kind, w, h, t, gap=g):
        nonlocal y
        y += h / 2
        b = getattr(s, kind)(xm, y, w, h, t)
        y += h / 2 + gap
        return b
    b0 = put("term", 3.0, 0.85, "Начало")
    b1 = put("io", wm, 0.95, "Чтение CSV-файла исходного набора")
    b2 = put("rect", wm, 1.25, "Отделение служебных полей и метки класса;\nсохранение пары IP-адресов как ключа группы")
    seq = [b0, b1, b2]
    decs = [("Есть входящие соединения\nк портам 22 и 3389 узла захвата?", "Исключить фоновые\nсоединения"),
            ("Есть значения IAT больше\nдлительности потока?", "Заменить такие значения\nнеопределёнными"),
            ("Есть совпадающие\nвекторы признаков?", "Удалить повторные\nзаписи")]
    nexts = [("rect", wm, 1.25, "Преобразование текстовых признаков\nрукопожатия в числовые"),
             None, ("rect", wm, 0.95, "Удаление константных признаков")]
    for (q, a), nx in zip(decs, nexts):
        d = put("dec", wm, 1.7, q, gap=1.0)
        sd = s.rect(xs, d["y"], wsd, 1.25, a)
        jy = y - 0.45
        s.line([bot(seq[-1]), top(d)])
        s.line([right(d), left(sd)])
        s.text(right(d)[0] + 0.12, d["y"] - 0.3, "да", ha="left", size=10)
        s.text(d["x"] + 0.15, bot(d)[1] + 0.25, "нет", ha="left", size=10)
        s.line([bot(sd), (xs, jy), (xm, jy)], arrow=False)
        s.head(xm, jy, -1, 0)
        seq = [d]
        if nx:
            b = put(*nx)
            s.line([bot(d), top(b)])
            seq = [b]
        else:
            seq = [d]
    out = put("io", wm, 1.25, "Очищенная выборка:\nпризнаки X, метки Y, группы G")
    end = put("term", 3.0, 0.85, "Конец")
    s.line([bot(seq[-1]), top(out)])
    s.line([bot(out), top(end)])
    for a, c in zip([b0, b1], [b1, b2]):
        s.line([bot(a), top(c)])
    H = y - g + 0.2
    s.ax.set_ylim(H, 0); s.fig.set_size_inches(W / 2.54, H / 2.54)
    s.save(OUT + "fig5.png")
    return H


def fig6():
    W = 16.5
    xm, wm = 4.55, 8.5
    xs, wsd = 12.75, 6.6
    g = 0.425
    s = S(W, 20)
    y = 0.2
    def put(kind, w, h, t, gap=g):
        nonlocal y
        y += h / 2
        b = getattr(s, kind)(xm, y, w, h, t)
        y += h / 2 + gap
        return b
    bs = [put("term", 3.0, 0.85, "Начало"),
          put("io", wm, 1.25, "Очищенная выборка:\nX, Y, G"),
          put("rect", wm, 1.25, "Групповое стратифицированное разбиение\nна 5 блоков (группа – пара IP-адресов)"),
          put("lstart", wm, 0.95, "Цикл: для каждого блока i = 1, …, 5"),
          put("rect", wm, 1.25, "Расчёт весов классов\nпо обучающей части блока")]
    for a, c in zip(bs, bs[1:]):
        s.line([bot(a), top(c)])
    d = put("dec", wm, 1.9, "Модель чувствительна\nк масштабу (MLP)?", gap=1.1)
    s.line([bot(bs[-1]), top(d)])
    sd = s.rect(xs, d["y"], wsd, 1.6, "Замена неопределённых значений\nнулём, добавление индикаторов,\nстандартизация по обучающей части")
    jy = y - 0.45
    s.line([right(d), left(sd)])
    s.text(right(d)[0] + 0.12, d["y"] - 0.3, "да", ha="left", size=10)
    s.text(d["x"] + 0.15, bot(d)[1] + 0.25, "нет", ha="left", size=10)
    s.line([bot(sd), (xs, jy), (xm, jy)], arrow=False)
    s.head(xm, jy, -1, 0)
    o = put("io", wm, 1.25, "Обучающая и контрольная части блока\nпереданы модулю обучения")
    le = put("lend", wm, 0.95, "Конец цикла по i")
    e = put("term", 3.0, 0.85, "Конец")
    s.line([bot(d), top(o)]); s.line([bot(o), top(le)]); s.line([bot(le), top(e)])
    H = y - g + 0.2
    s.ax.set_ylim(H, 0); s.fig.set_size_inches(W / 2.54, H / 2.54)
    s.save(OUT + "fig6.png")


def fig7():
    W = 15.0
    xm, wm = 4.55, 8.5
    xs, wsd = 11.9, 5.4
    g = 0.55
    s = S(W, 20)
    y = 0.2
    def put(kind, w, h, t, gap=g):
        nonlocal y
        y += h / 2
        b = getattr(s, kind)(xm, y, w, h, t)
        y += h / 2 + gap
        return b
    bs = [put("term", 3.0, 0.85, "Начало"),
          put("io", wm, 0.95, "Вектор признаков потока x"),
          put("rect", wm, 1.25, "Модель первого уровня:\nрасчёт P1(Anon | x)")]
    d = put("dec", wm, 1.6, "P1(Anon | x) ≥ τ ?")
    rest = [put("rect", wm, 1.25, "Модель второго уровня: расчёт P2(k | x)\nдля Tor, I2P, ZeroNet, Freenet, VPN"),
            put("rect", wm, 1.25, "Выбор технологии\nс наибольшей вероятностью P2(k | x)"),
            put("io", wm, 1.25, "Класс технологии\nанонимизации k"),
            put("term", 3.0, 0.85, "Конец")]
    seq = bs + [d] + rest
    for a, c in zip(seq, seq[1:]):
        s.line([bot(a), top(c)])
    sd = s.io(xs, d["y"], wsd, 1.25, "Класс Regular\n(обычный трафик)")
    s.line([right(d), left(sd)])
    s.text(right(d)[0] + 0.12, d["y"] - 0.3, "нет", ha="left", size=10)
    s.text(d["x"] + 0.15, bot(d)[1] + 0.25, "да", ha="left", size=10)
    end = rest[-1]
    s.line([bot(sd), (xs, end["y"]), right(end)])
    H = y - g + 0.2
    s.ax.set_ylim(H, 0); s.fig.set_size_inches(W / 2.54, H / 2.54)
    s.save(OUT + "fig7.png")


import sys
for f in [globals()[n] for n in sys.argv[1:]] or (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
    f()
print("ok")
