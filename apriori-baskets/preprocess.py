#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Предобработка Census Income: очистка, дискретизация, кодирование.
"""

import csv
import json
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/processed")

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

CONTINUOUS_INDICES = [0, 2, 4, 10, 11, 12]
N_BINS = 5


def load_csv(path, skip_header=False):
    """Чтение CSV, пропуск строки заголовка при необходимости."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if skip_header and i == 0:
                continue
            row = [v.strip() for v in row]
            if len(row) >= 15:
                rows.append(row)
    return rows


def remove_missing(rows):
    """Удаление строк с '?'."""
    return [r for r in rows if "?" not in r]


def discretize_continuous(rows):
    """Равноширинная дискретизация непрерывных признаков."""
    for col_idx in CONTINUOUS_INDICES:
        values = [float(r[col_idx]) for r in rows]
        mn, mx = min(values), max(values)
        if mx == mn:
            continue
        width = (mx - mn) / N_BINS
        for r in rows:
            idx = min(int((float(r[col_idx]) - mn) / width), N_BINS - 1)
            r[col_idx] = f"bin_{idx}"
    return rows


def encode_categorical(rows):
    """Кодирование категориальных признаков в целые индексы."""
    n_cols = len(rows[0])
    maps = [{} for _ in range(n_cols)]
    counters = [0] * n_cols

    for r in rows:
        for i in range(n_cols):
            if r[i] not in maps[i]:
                maps[i][r[i]] = counters[i]
                counters[i] += 1

    encoded = [[maps[i][r[i]] for i in range(n_cols)] for r in rows]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(COLUMNS):
        (OUTPUT_DIR / f"mapping_{name}.json").write_text(
            json.dumps(maps[i], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    return encoded


def save_csv(rows, path):
    """Сохранение в CSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Обработка обучающей выборки...")
    train = load_csv(DATA_DIR / "adult.data")
    train = remove_missing(train)
    train = discretize_continuous(train)
    train = encode_categorical(train)
    save_csv(train, OUTPUT_DIR / "adult_train.csv")

    print("Обработка тестовой выборки...")
    test = load_csv(DATA_DIR / "adult.test", skip_header=True)
    test = remove_missing(test)
    test = discretize_continuous(test)
    test = encode_categorical(test)
    save_csv(test, OUTPUT_DIR / "adult_test.csv")

    print("Готово.")


if __name__ == "__main__":
    main()