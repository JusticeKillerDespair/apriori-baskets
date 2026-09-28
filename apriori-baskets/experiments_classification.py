#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Эксперименты с деревом решений на Census Income."""

import csv
import json
import time
from pathlib import Path

import numpy as np

from decision_tree import build_tree, predict
from evaluate import evaluate_all
from visualize_tree import export_tree
from plot_metrics import plot_metrics

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results")

FEATURE_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country"
]
CLASS_NAMES = ["<=50K", ">50K"]

CRITERIA = ["information_gain", "gain_ratio", "gini"]
TRAIN_RATIOS = [0.60, 0.70, 0.80, 0.90]


def read_csv(path):
    """Чтение подготовленного CSV."""
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 15:
                continue
            X.append([int(v) for v in row[:14]])
            y.append(int(row[14]))
    return np.array(X), np.array(y)


def split_data(X, y, ratio, seed=42):
    """Разбиение на обучающую и тестовую части."""
    np.random.seed(seed)
    idx = np.random.permutation(len(y))
    split = int(len(y) * ratio)
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


def experiment_1():
    """100% обучающей выборки: построение и визуализация деревьев."""
    print("=" * 50)
    print("Эксперимент 1: 100% обучающей выборки")
    print("=" * 50)

    X_train, y_train = read_csv(DATA_DIR / "adult_train.csv")
    X_test, y_test = read_csv(DATA_DIR / "adult_test.csv")
    print(f"Обучающая: {len(y_train)}, тестовая: {len(y_test)}")

    for criterion in CRITERIA:
        print(f"\nКритерий: {criterion}")
        t0 = time.perf_counter()
        tree = build_tree(X_train, y_train, criterion=criterion,
                          max_depth=4, min_samples_split=50)
        print(f"  Время построения: {time.perf_counter() - t0:.2f} с")

        y_pred = predict(tree, X_test)
        m = evaluate_all(y_test, y_pred, positive_label=1)
        print(f"  Accuracy={m['accuracy']:.4f}  "
              f"Precision={m['precision']:.4f}  "
              f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}")

        export_tree(tree, FEATURE_NAMES, CLASS_NAMES,
                    filename=f"results/tree_{criterion}", max_depth=3)


def experiment_2():
    """Разные доли обучающей выборки."""
    print("\n" + "=" * 50)
    print("Эксперимент 2: доли обучающей выборки 60-90%")
    print("=" * 50)

    X_full, y_full = read_csv(DATA_DIR / "adult_train.csv")
    all_results = {}

    for criterion in CRITERIA:
        print(f"\nКритерий: {criterion}")
        results = []
        for ratio in TRAIN_RATIOS:
            X_tr, y_tr, X_te, y_te = split_data(X_full, y_full, ratio)
            t0 = time.perf_counter()
            tree = build_tree(X_tr, y_tr, criterion=criterion,
                              max_depth=6, min_samples_split=30)
            elapsed = time.perf_counter() - t0

            y_pred = predict(tree, X_te)
            m = evaluate_all(y_te, y_pred, positive_label=1)
            results.append({"train_ratio": ratio, "time_sec": elapsed, **m})

            print(f"  {ratio*100:.0f}%: acc={m['accuracy']:.4f}, "
                  f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, "
                  f"f1={m['f1']:.4f}")

        all_results[criterion] = results
        plot_metrics(results, criterion)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "classification_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("\nРезультаты сохранены: results/classification_results.json")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    experiment_1()
    experiment_2()


if __name__ == "__main__":
    main()