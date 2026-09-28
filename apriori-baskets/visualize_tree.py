#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация дерева решений с помощью matplotlib.
Не требует установки Graphviz.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"


def _get_depth(node):
    """Максимальная глубина поддерева."""
    if node.label is not None or not node.children:
        return 1
    return 1 + max(_get_depth(c) for c in node.children.values())


def _layout(node, depth, x_start, x_end, positions, edges,
            feature_names, class_names, max_depth):
    """Рекурсивное размещение узлов дерева."""
    x = (x_start + x_end) / 2
    y = -depth

    if node.label is not None:
        label = class_names[node.label] if node.label < len(class_names) else str(node.label)
        positions.append((x, y, label, "leaf"))
        return

    if depth >= max_depth:
        positions.append((x, y, "...", "leaf"))
        return

    fname = feature_names[node.feature] if node.feature < len(feature_names) else f"f{node.feature}"
    positions.append((x, y, fname, "internal"))

    n_children = len(node.children)
    width = (x_end - x_start) / n_children
    for i, (val, child) in enumerate(node.children.items()):
        cx_start = x_start + i * width
        cx_end = cx_start + width
        cx = (cx_start + cx_end) / 2
        edges.append((x, y, cx, y - 1, str(val)))
        _layout(child, depth + 1, cx_start, cx_end,
                positions, edges, feature_names, class_names, max_depth)


def export_tree(node, feature_names, class_names, filename="tree", max_depth=3):
    """
    Рисует дерево решений и сохраняет в PNG.

    Параметры:
        node          : корневой узел
        feature_names : список имён признаков
        class_names   : список имён классов
        filename      : имя файла без расширения
        max_depth     : максимальная отображаемая глубина
    """
    positions, edges = [], []
    _layout(node, 0, 0, 1, positions, edges,
            feature_names, class_names, max_depth)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis("off")

    # Рёбра
    for x1, y1, x2, y2, label in edges:
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label,
                fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1",
                          facecolor="white", edgecolor="none"))

    # Узлы
    for x, y, label, kind in positions:
        color = "lightgreen" if kind == "leaf" else "lightblue"
        ax.text(x, y, label, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=color, edgecolor="black"))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-max_depth - 1, 1)
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Дерево сохранено: {filename}.png")