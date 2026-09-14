#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для поиска частых наборов объектов с помощью алгоритма Apriori.

Входные данные: CSV-файл, в каждой строке которого через запятую перечислены
товары одной покупки. Поддерживается кодировка UTF-8 (в т.ч. с BOM).
Выходные данные: список частых наборов с абсолютной и относительной поддержкой.

Пример запуска:
    python apriori.py --input baskets.csv --min-support 0.05 --sort support
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple, Union

Itemset = FrozenSet[str]
Transaction = FrozenSet[str]
FrequentResult = List[Tuple[Itemset, int, float]]


def read_transactions(path: Union[str, Path]) -> List[Transaction]:
    """
    读取购物篮 CSV 文件。

    文件可能不是 UTF-8 编码（例如 Windows-1251），因此依次尝试多种编码。

    参数:
        path: CSV 文件路径。

    返回:
        交易列表。每个交易是一个 frozenset[str]。
        空项忽略，同一行中重复商品只算一次。
    """
    transactions: List[Transaction] = []
    encodings = ("utf-8-sig", "cp1251", "cp866", "latin-1")

    last_error: Exception | None = None
    for enc in encodings:
        try:
            transactions = []
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    items = {item.strip() for item in row if item.strip()}
                    if items:
                        transactions.append(frozenset(items))
            print(f"[read_transactions] 成功使用编码: {enc}")
            return transactions
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"无法解码文件 {path}，尝试过的编码: {encodings}"
    ) from last_error


def format_itemset(itemset: Itemset) -> str:
    """Возвращает строковое представление набора в лексикографическом порядке."""
    return "{" + ", ".join(sorted(itemset)) + "}"


def generate_candidates(prev_frequent: Dict[Itemset, int]) -> Set[Itemset]:
    """
    Генерирует кандидаты длины k на основе частых наборов длины k-1.

    Параметры:
        prev_frequent: словарь {набор длины k-1: абсолютная поддержка}.

    Возвращает:
        Множество кандидатов длины k.
    """
    sorted_itemsets = [tuple(sorted(s)) for s in prev_frequent.keys()]
    groups: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = defaultdict(list)

    # Группируем по префиксу длины k-2.
    for s in sorted_itemsets:
        groups[s[:-1]].append(s)

    candidates: Set[Itemset] = set()
    for group in groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]

                # Соединяем два набора, отличающиеся только последним элементом.
                candidate_tuple = a + (b[-1],)
                candidate = frozenset(candidate_tuple)

                # Проверка антимонотонности: все подмножества длины k-1
                # должны быть частыми.
                if all(
                    frozenset(sub) in prev_frequent
                    for sub in combinations(candidate, len(candidate) - 1)
                ):
                    candidates.add(candidate)

    return candidates


def count_candidates(
    transactions: List[Transaction],
    candidates: Set[Itemset],
) -> Dict[Itemset, int]:
    """
    Подсчитывает абсолютную поддержку кандидатов.

    Параметры:
        transactions: список транзакций.
        candidates: множество наборов-кандидатов.

    Возвращает:
        Словарь {набор: число транзакций, содержащих набор}.
    """
    counts: Dict[Itemset, int] = defaultdict(int)

    for tr in transactions:
        for candidate in candidates:
            if candidate.issubset(tr):
                counts[candidate] += 1

    return counts


def apriori(
    transactions: List[Transaction],
    min_support: float,
    sort_by: str = "support",
) -> FrequentResult:
    """
    Ищет частые наборы алгоритмом Apriori.

    Параметры:
        transactions: список транзакций.
        min_support: относительный порог поддержки, например 0.05 = 5%.
        sort_by: способ сортировки результата:
                 'support' — по убыванию поддержки, затем лексикографически;
                 'lex' — лексикографически.

    Возвращает:
        Список кортежей (набор, абсолютная поддержка, относительная поддержка).
    """
    if not transactions:
        return []

    if not (0.0 < min_support <= 1.0):
        raise ValueError("min_support должен быть в интервале (0, 1].")

    n = len(transactions)
    min_count = min_support * n

    # Частые наборы длины 1.
    item_counts: Dict[Itemset, int] = defaultdict(int)
    for tr in transactions:
        for item in tr:
            item_counts[frozenset([item])] += 1

    frequent_by_length: Dict[int, Dict[Itemset, int]] = {}
    frequent_by_length[1] = {
        itemset: cnt
        for itemset, cnt in item_counts.items()
        if cnt >= min_count
    }

    all_frequent: FrequentResult = []
    k = 1

    while frequent_by_length.get(k):
        for itemset, cnt in frequent_by_length[k].items():
            all_frequent.append((itemset, cnt, cnt / n))

        k += 1
        candidates = generate_candidates(frequent_by_length[k - 1])
        if not candidates:
            break

        counts = count_candidates(transactions, candidates)
        frequent_by_length[k] = {
            itemset: cnt
            for itemset, cnt in counts.items()
            if cnt >= min_count
        }

    if sort_by == "support":
        all_frequent.sort(key=lambda x: (-x[1], tuple(sorted(x[0]))))
    elif sort_by == "lex":
        all_frequent.sort(key=lambda x: tuple(sorted(x[0])))
    else:
        raise ValueError("sort_by должен быть 'support' или 'lex'.")

    return all_frequent


def main() -> None:
    """Точка входа для командной строки."""
    parser = argparse.ArgumentParser(
        description="Поиск частых наборов в CSV-файле с покупками (Apriori)."
    )
    parser.add_argument(
        "--input",
        default="baskets.csv",
        help="путь к CSV-файлу с транзакциями",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="относительный порог поддержки, например 0.05 = 5%%",
    )
    parser.add_argument(
        "--sort",
        choices=["support", "lex"],
        default="support",
        help="support — по убыванию поддержки, lex — лексикографически",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="файл для сохранения результата",
    )
    args = parser.parse_args()

    transactions = read_transactions(args.input)

    t0 = time.perf_counter()
    result = apriori(transactions, args.min_support, args.sort)
    elapsed = time.perf_counter() - t0

    lines = [
        f"Файл: {args.input}",
        f"Число транзакций: {len(transactions)}",
        f"Порог поддержки: {args.min_support:.4f} ({args.min_support * 100:.2f}%)",
        f"Сортировка: {args.sort}",
        f"Время работы Apriori: {elapsed:.6f} с",
        f"Найдено частых наборов: {len(result)}",
        "",
        "Набор\tАбс. поддержка\tОтн. поддержка",
    ]

    for itemset, cnt, support in result:
        lines.append(f"{format_itemset(itemset)}\t{cnt}\t{support:.6f}")

    text = "\n".join(lines)
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()