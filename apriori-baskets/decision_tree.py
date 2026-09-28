#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дерево решений: ID3 (Information gain), C4.5 (Gain ratio), CART (Gini index).
"""

import numpy as np


class Node:
    """Узел дерева решений."""

    def __init__(self, feature=None, children=None, label=None, majority=None):
        """
        Параметры:
            feature  : индекс признака для разбиения (None для листа)
            children : словарь {значение признака: Node}
            label    : метка класса (для листа)
            majority : метка класса, преобладающая в подвыборке узла
        """
        self.feature = feature
        self.children = children if children is not None else {}
        self.label = label
        self.majority = majority


def entropy(y):
    """Энтропия Шеннона."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))


def information_gain(y, y_subs):
    """Прирост информации (ID3)."""
    n = len(y)
    weighted = sum(len(s) / n * entropy(s) for s in y_subs if len(s) > 0)
    return entropy(y) - weighted


def gain_ratio(y, y_subs):
    """Отношение прироста информации (C4.5)."""
    n = len(y)
    ig = information_gain(y, y_subs)
    split_info = 0.0
    for s in y_subs:
        if len(s) > 0:
            p = len(s) / n
            split_info -= p * np.log2(p + 1e-12)
    if split_info == 0:
        return 0.0
    return ig / split_info


def gini_index(y, y_subs):
    """Индекс Джини (CART), возвращает отрицательное взвешенное значение."""
    def gini(arr):
        if len(arr) == 0:
            return 0.0
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / len(arr)
        return 1.0 - np.sum(probs ** 2)

    n = len(y)
    weighted = sum(len(s) / n * gini(s) for s in y_subs if len(s) > 0)
    return -weighted


def majority_label(y):
    """Метка класса, встречающаяся чаще всего."""
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]


def build_tree(X, y, criterion="information_gain", max_depth=None,
               min_samples_split=2, min_samples_leaf=1, depth=0):
    """
    Рекурсивное построение дерева решений.
    """
    n_samples, n_features = X.shape
    maj = majority_label(y)

    if len(np.unique(y)) == 1:
        return Node(label=y[0], majority=maj)
    if max_depth is not None and depth >= max_depth:
        return Node(label=maj, majority=maj)
    if n_samples < min_samples_split:
        return Node(label=maj, majority=maj)

    best_feature, best_gain, best_vals, best_subs = None, -np.inf, None, None

    for feature in range(n_features):
        unique_vals = np.unique(X[:, feature])
        if len(unique_vals) <= 1:
            continue

        y_subs = [y[X[:, feature] == v] for v in unique_vals]

        if criterion == "information_gain":
            gain = information_gain(y, y_subs)
        elif criterion == "gain_ratio":
            gain = gain_ratio(y, y_subs)
        elif criterion == "gini":
            gain = gini_index(y, y_subs)
        else:
            raise ValueError(f"Неизвестный критерий: {criterion}")

        if gain > best_gain:
            best_gain, best_feature = gain, feature
            best_vals, best_subs = unique_vals, y_subs

    if best_feature is None:
        return Node(label=maj, majority=maj)

    children = {}
    for val, y_sub in zip(best_vals, best_subs):
        if len(y_sub) < min_samples_leaf:
            children[val] = Node(label=majority_label(y_sub),
                                 majority=majority_label(y_sub))
        else:
            mask = X[:, best_feature] == val
            children[val] = build_tree(
                X[mask], y[mask], criterion, max_depth,
                min_samples_split, min_samples_leaf, depth + 1
            )

    return Node(feature=best_feature, children=children, majority=maj)


def predict_one(node, x):
    """Предсказание для одного объекта."""
    if node.label is not None:
        return node.label
    val = x[node.feature]
    if val in node.children:
        return predict_one(node.children[val], x)
    # Неизвестное значение признака: возвращаем преобладающий класс узла
    return node.majority


def predict(node, X):
    """Предсказание для набора объектов."""
    return np.array([predict_one(node, x) for x in X])