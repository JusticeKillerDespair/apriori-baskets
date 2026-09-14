#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проведение экспериментов с Apriori на baskets.csv и визуализация результатов.

Запуск:
    python experiments.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Для сохранения графиков без графического окна
import matplotlib.pyplot as plt

from apriori import apriori, read_transactions

plt.rcParams["font.family"] = "DejaVu Sans"

THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15)
INPUT_CSV = "baskets.csv"
OUTPUT_DIR = Path("results")
REPEATS = 3  # Повторяем замер времени, чтобы уменьшить случайные колебания


def run_experiments() -> List[Dict]:
    """Запускает Apriori для всех порогов и собирает статистику."""
    transactions = read_transactions(INPUT_CSV)
    results: List[Dict] = []

    for thr in THRESHOLDS:
        times = []
        frequent = []

        for _ in range(REPEATS):
            t0 = time.perf_counter()
            frequent = apriori(transactions, thr, sort_by="support")
            times.append(time.perf_counter() - t0)

        elapsed = sum(times) / len(times)

        lengths: Dict[int, int] = {}
        for itemset, _, _ in frequent:
            k = len(itemset)
            lengths[k] = lengths.get(k, 0) + 1

        results.append(
            {
                "threshold": thr,
                "threshold_percent": thr * 100.0,
                "time_sec": elapsed,
                "total_frequent": len(frequent),
                "lengths": lengths,
            }
        )

        print(
            f"threshold={thr:.2f} ({thr * 100:.0f}%), "
            f"time={elapsed:.4f} s, total={len(frequent)}, lengths={lengths}"
        )

    return results


def save_results(results: List[Dict]) -> None:
    """Сохраняет результаты экспериментов в JSON и CSV."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    (OUTPUT_DIR / "experiment_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(
        OUTPUT_DIR / "experiment_results.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "threshold_percent",
                "time_sec",
                "total_frequent",
                "lengths",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r["threshold"],
                    r["threshold_percent"],
                    f"{r['time_sec']:.6f}",
                    r["total_frequent"],
                    json.dumps(r["lengths"], ensure_ascii=False),
                ]
            )


def plot_time(results: List[Dict]) -> None:
    """Строит график зависимости времени работы от порога поддержки."""
    x = [r["threshold_percent"] for r in results]
    y = [r["time_sec"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Порог поддержки, %")
    plt.ylabel("Время работы, с")
    plt.title("Зависимость времени работы Apriori от порога поддержки")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_vs_support.png", dpi=200)
    plt.close()


def plot_lengths(results: List[Dict]) -> None:
    """Строит график количества частых наборов разной длины."""
    thresholds = [r["threshold_percent"] for r in results]

    max_len = max(
        [max(r["lengths"].keys()) if r["lengths"] else 0 for r in results]
        + [1]
    )

    # Линейный график по длинам.
    plt.figure(figsize=(9, 5))
    for k in range(1, max_len + 1):
        vals = [r["lengths"].get(k, 0) for r in results]
        plt.plot(thresholds, vals, marker="o", label=f"Длина {k}")

    plt.xlabel("Порог поддержки, %")
    plt.ylabel("Количество частых наборов")
    plt.title("Количество частых наборов различной длины")
    plt.xticks(thresholds, [f"{t:.0f}%" for t in thresholds])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lengths_vs_support.png", dpi=200)
    plt.close()

    # Дополнительно столбчатая диаграмма.
    plt.figure(figsize=(9, 5))
    width = 0.8 / max_len
    x_base = range(len(thresholds))

    for k in range(1, max_len + 1):
        vals = [r["lengths"].get(k, 0) for r in results]
        plt.bar(
            [i + (k - 1) * width for i in x_base],
            vals,
            width=width,
            label=f"Длина {k}",
        )

    plt.xticks(
        [i + 0.4 - width / 2 for i in x_base],
        [f"{t:.0f}%" for t in thresholds],
    )
    plt.xlabel("Порог поддержки")
    plt.ylabel("Количество частых наборов")
    plt.title("Количество частых наборов различной длины")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lengths_vs_support_bars.png", dpi=200)
    plt.close()


def main() -> None:
    """Запускает эксперименты и сохраняет графики."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = run_experiments()
    save_results(results)
    plot_time(results)
    plot_lengths(results)

    print(f"Результаты и графики сохранены в каталоге: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()