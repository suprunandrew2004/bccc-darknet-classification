# Классификация зашифрованного трафика средств анонимизации (BCCC-Darknet-2025)

Программы и результаты к курсовой работе «Применение машинного обучения для классификации
зашифрованного трафика в задачах цифровой криминалистики» (Супрун А. А., гр. БИТ-221).

Реализован двухуровневый классификатор потоков:

1. **Первый уровень** – обычный трафик (Regular) или трафик средств анонимизации (Anon).
2. **Второй уровень** – конкретная технология: Tor, I2P, ZeroNet, Freenet, VPN.

Качество оценивается групповой кросс-валидацией: 5 блоков, группировка по паре IP-адресов, начальное значение
генератора 42. Сравниваются модели XGBoost, случайный лес и многослойный перцептрон.

## Структура репозитория

| Путь | Назначение |
|---|---|
| `src/load_bccc.py` | загрузка и очистка набора: служебные поля, фоновые соединения, артефакт IAT, повторы, константные признаки |
| `src/common.py` | модели, групповое разбиение, кросс-валидация, метрики |
| `src/eda.py` | разведочный анализ: медианы признаков, распределения, корреляции, PCA |
| `src/eval_schemes.py` | основная схема и сопоставительные схемы S1–S5, кривая обучения, матрицы ошибок, PR-кривые |
| `src/extra.py` | сила взвешивания α, перестановочная важность, значения Шепли, разделение стендов, оценка по окнам 300 с |
| `src/binary_two_level.py` | первый уровень и двухуровневая схема |
| `src/review.py` | дополнительные проверки: состав признаков среды, PCA, пересечения файлов, таблица порогов τ |
| `src/clean_test.py` | влияние очистки на одних и тех же очищенных контрольных потоках |
| `src/safesurf.py` | проверка выводов на наборе SafeSurf Darknet 2025 |
| `src/figs.py` | диаграмма значений Шепли |
| `src/check_data.py` | проверка контрольных сумм SHA-256 исходных файлов |
| `schemes/` | построение схем и блок-схем (рисунки 1–7 пояснительной записки) |
| `results/` | полученные результаты (JSON) и рисунки |
| `data/SHA256SUMS.txt` | контрольные суммы исходных CSV-файлов |
| `requirements.txt` | версии библиотек |

## Данные

Исходные наборы в репозиторий не включены: их распространяют авторы.

* **BCCC-Darknet-2025.** Сайт:
  https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/encrypted-traffic-dataset-bccc-darknet-2025/
  Файлы `MultiTotalDS.csv` и `Binary -2DSCombined.csv` положите в каталог `data/`.
* **SafeSurf Darknet 2025.** Источник: https://data.mendeley.com/datasets/kcrnj6z4rm/2
  Файлы `SafeSurf Dataset Layer 1.csv` и `SafeSurf Dataset Layer 2 & 3.csv` положите в каталог `data_ss/`.

Проверка, что используются те же файлы, что и в работе:

```
python src/check_data.py
```

## Установка

Использовались Python 3.11 и библиотеки следующих версий: numpy 2.4.4, pandas 3.0.2,
scikit-learn 1.8.0, xgboost 3.2.0, scipy 1.17.1, shap 0.51.0, matplotlib 3.10.9.

```
python -m venv .venv
.venv\Scripts\activate          # Windows; в Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Все команды выполняются из корня репозитория. Результаты записываются в `results/`.

```
python src/eda.py                       # разведочный анализ
python src/eval_schemes.py              # основная и сопоставительные схемы
python src/extra.py                     # использует results/oof_S3.npy из предыдущего шага
python src/binary_two_level.py          # первый уровень и двухуровневая схема
python src/figs.py                      # диаграмма значений Шепли
python src/review.py tau                # дополнительные проверки и таблица порогов τ
python src/clean_test.py C1_no_iat_fix C2_no_bg_filter C3_raw C3_raw_random
python src/safesurf.py                  # требует каталог data_ss/
python schemes/block_diagrams.py        # схемы и блок-схемы
```

Полный расчёт на обычном компьютере занимает от десятков минут до нескольких часов. Больше всего времени
уходит на повторы схем S1–S2 и на `clean_test.py`.

## Соответствие рисунков пояснительной записке

| Рисунок | Файл | Программа |
|---|---|---|
| 1–7 | `results/schemes/fig1.png` … `fig7.png` | `schemes/block_diagrams.py` |
| 8 | `results/fig_box.png` | `src/eda.py` |
| 9 | `results/fig_conf.png` | `src/eval_schemes.py` |
| 10 | `results/fig_pr.png` | `src/eval_schemes.py` |
| 11 | `results/fig_cv_group.png` | `src/eval_schemes.py` |
| 12 | `results/fig_cascade.png` | `src/binary_two_level.py` |
| 13 | `results/fig_schemes.png` | `src/eval_schemes.py` |
| 14 | `results/fig_learning_curve.png` | `src/eval_schemes.py` |
| 15 | `results/fig_perm.png` | `src/extra.py` |
| 16 | `results/fig_shap.png` | `src/extra.py`, `src/figs.py` |
| 17 | `results/fig_safesurf.png` | `src/safesurf.py` |

Числовые значения таблиц хранятся в файлах `results/results_*.json` и `results/clean_test_*.json`.

## Лицензия

Код распространяется по лицензии MIT (см. `LICENSE`). Наборы данных распространяются на условиях их авторов.
