#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Поиск ассоциативных правил на основе частых наборов.

Модуль использует функции из apriori.py:
    - read_transactions
    - apriori
    - format_itemset
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

from apriori import apriori, format_itemset, read_transactions

Itemset = FrozenSet[str]
Rule = Tuple[Itemset, Itemset, float, float]  # (antecedent, consequent, support, confidence)


def generate_rules(
    frequent_itemsets: List[Tuple[Itemset, int, float]],
    min_confidence: float,
    max_total_items: int = 7,
) -> List[Rule]:
    """
    Генерирует ассоциативные правила из частых наборов.

    Параметры:
        frequent_itemsets: список (набор, абсолютная поддержка, относительная поддержка),
                           полученный из функции apriori.
        min_confidence: минимальная достоверность правила (0..1).
        max_total_items: максимальное суммарное число объектов в антецеденте
                         и консеквенте (по умолчанию 7).

    Возвращает:
        Список правил: (antecedent, consequent, support, confidence).
    """
    # Словарь поддержек: набор -> относительная поддержка
    support_dict: Dict[Itemset, float] = {
        itemset: support for itemset, _, support in frequent_itemsets
    }

    rules: List[Rule] = []

    for itemset, _, support in frequent_itemsets:
        if len(itemset) < 2:
            continue

        items = list(itemset)

        # Перебираем все непустые собственные подмножества как антецеденты
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                if not consequent:
                    continue

                if len(antecedent) + len(consequent) > max_total_items:
                    continue

                ant_support = support_dict.get(antecedent)
                if ant_support is None or ant_support == 0.0:
                    continue

                confidence = support / ant_support

                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, support, confidence))

    return rules


def sort_rules(rules: List[Rule], sort_by: str) -> List[Rule]:
    """
    Сортирует список правил.

    Параметры:
        rules: список правил.
        sort_by:
            'support' — по убыванию поддержки, затем по убыванию достоверности;
            'lex' — лексикографически по антецеденту и консеквенту.

    Возвращает:
        Отсортированный список правил.
    """
    if sort_by == "support":
        return sorted(
            rules,
            key=lambda r: (
                -r[2],  # support по убыванию
                -r[3],  # confidence по убыванию
                tuple(sorted(r[0])),
                tuple(sorted(r[1])),
            ),
        )
    elif sort_by == "lex":
        return sorted(
            rules,
            key=lambda r: (tuple(sorted(r[0])), tuple(sorted(r[1]))),
        )
    else:
        raise ValueError("sort_by должен быть 'support' или 'lex'")


def format_rule(rule: Rule) -> str:
    """Возвращает удобочитаемое представление правила."""
    antecedent, consequent, support, confidence = rule
    return (
        f"{format_itemset(antecedent)} -> {format_itemset(consequent)} "
        f"(support={support:.6f}, confidence={confidence:.6f})"
    )


def main() -> None:
    """Точка входа для командной строки."""
    parser = argparse.ArgumentParser(
        description="Поиск ассоциативных правил (Apriori + правила)."
    )
    parser.add_argument(
        "--input",
        default="baskets.csv",
        help="путь к CSV-файлу с транзакциями",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.10,
        help="относительный порог поддержки, например 0.10 = 10%%",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        help="относительный порог достоверности, например 0.70 = 70%%",
    )
    parser.add_argument(
        "--sort",
        choices=["support", "lex"],
        default="support",
        help="support — по убыванию поддержки, lex — лексикографически",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=7,
        help="максимальное суммарное число объектов в правиле",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="файл для сохранения правил",
    )
    args = parser.parse_args()

    transactions = read_transactions(args.input)

    t0 = time.perf_counter()
    frequent = apriori(transactions, args.min_support, sort_by="support")
    rules = generate_rules(frequent, args.min_confidence, args.max_items)
    rules = sort_rules(rules, args.sort)
    elapsed = time.perf_counter() - t0

    lines = [
        f"Файл: {args.input}",
        f"Число транзакций: {len(transactions)}",
        f"Порог поддержки: {args.min_support:.4f} ({args.min_support * 100:.2f}%)",
        f"Порог достоверности: {args.min_confidence:.4f} ({args.min_confidence * 100:.2f}%)",
        f"Сортировка: {args.sort}",
        f"Максимум объектов в правиле: {args.max_items}",
        f"Время работы: {elapsed:.6f} с",
        f"Найдено правил: {len(rules)}",
        "",
        "Правила:",
    ]

    for rule in rules:
        lines.append(format_rule(rule))

    text = "\n".join(lines)
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()