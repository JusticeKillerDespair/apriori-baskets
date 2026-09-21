#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперименты по поиску ассоциативных правил.

Фиксируется поддержка 10%, варьируется достоверность:
70%, 75%, 80%, 85%, 90%, 95%.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apriori import apriori, read_transactions
from association_rules import generate_rules, sort_rules

plt.rcParams["font.family"] = "DejaVu Sans"

INPUT_CSV = "baskets.csv"
OUTPUT_DIR = Path("results")
MIN_SUPPORT = 0.01
CONFIDENCES = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
MAX_ITEMS = 7
REPEATS = 1  # можно увеличить до 3, если позволяет время


def run_experiments() -> List[Dict]:
    """Запускает эксперименты и собирает статистику."""
    transactions = read_transactions(INPUT_CSV)
    results: List[Dict] = []

    for conf in CONFIDENCES:
        times = []
        rules_count = 0

        for _ in range(REPEATS):
            t0 = time.perf_counter()
            frequent = apriori(transactions, MIN_SUPPORT, sort_by="support")
            rules = generate_rules(frequent, conf, MAX_ITEMS)
            rules = sort_rules(rules, "support")
            times.append(time.perf_counter() - t0)
            rules_count = len(rules)

        elapsed = sum(times) / len(times)

        results.append(
            {
                "min_confidence": conf,
                "min_confidence_percent": conf * 100.0,
                "time_sec": elapsed,
                "rules_count": rules_count,
            }
        )

        print(
            f"confidence={conf:.2f} ({conf*100:.0f}%), "
            f"time={elapsed:.6f} s, rules={rules_count}"
        )

    return results


def save_results(results: List[Dict]) -> None:
    """Сохраняет результаты в JSON и CSV."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    (OUTPUT_DIR / "rules_experiment_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(
        OUTPUT_DIR / "rules_experiment_results.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "min_confidence",
                "min_confidence_percent",
                "time_sec",
                "rules_count",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["min_confidence"],
                    r["min_confidence_percent"],
                    f"{r['time_sec']:.6f}",
                    r["rules_count"],
                ]
            )


def plot_time(results: List[Dict]) -> None:
    """График времени работы от порога достоверности."""
    x = [r["min_confidence_percent"] for r in results]
    y = [r["time_sec"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Порог достоверности, %")
    plt.ylabel("Время работы, с")
    plt.title("Зависимость времени поиска правил от порога достоверности")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rules_time_vs_confidence.png", dpi=200)
    plt.close()


def plot_rules_count(results: List[Dict]) -> None:
    """График количества правил от порога достоверности."""
    x = [r["min_confidence_percent"] for r in results]
    y = [r["rules_count"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Порог достоверности, %")
    plt.ylabel("Количество правил")
    plt.title("Количество найденных правил при различных порогах достоверности")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rules_count_vs_confidence.png", dpi=200)
    plt.close()


def main() -> None:
    """Запускает эксперименты и сохраняет графики."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = run_experiments()
    save_results(results)
    plot_time(results)
    plot_rules_count(results)
    print(f"Результаты сохранены в каталоге: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()