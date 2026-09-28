#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Графики метрик качества."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"
OUTPUT_DIR = Path("results")


def plot_metrics(results, criterion_name):
    """Построение графика метрик в зависимости от доли обучающей выборки."""
    ratios = [r["train_ratio"] * 100 for r in results]
    titles = {"accuracy": "Accuracy", "precision": "Precision",
              "recall": "Recall", "f1": "F1-мера"}

    plt.figure(figsize=(10, 6))
    for m in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(ratios, [r[m] for r in results], marker="o", label=titles[m])

    plt.xlabel("Доля обучающей выборки, %")
    plt.ylabel("Значение показателя")
    plt.title(f"Показатели качества ({criterion_name})")
    plt.xticks(ratios, [f"{r:.0f}%" for r in ratios])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"metrics_{criterion_name}.png", dpi=200)
    plt.close()
    print(f"График сохранён: results/metrics_{criterion_name}.png")