#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Метрики качества классификации."""

import numpy as np


def accuracy(y_true, y_pred):
    """Доля верных ответов."""
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, positive_label):
    """Precision, Recall, F1 для положительного класса."""
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def evaluate_all(y_true, y_pred, positive_label=1):
    """Все метрики в одном словаре."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision_recall_f1(y_true, y_pred, positive_label)[0],
        "recall": precision_recall_f1(y_true, y_pred, positive_label)[1],
        "f1": precision_recall_f1(y_true, y_pred, positive_label)[2],
    }