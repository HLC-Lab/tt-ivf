#!/usr/bin/env python3
"""Compare original-query-order result dumps from two transport configurations."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read(path: Path, queries: int | None = None, k: int | None = None) -> dict[tuple[int, int], tuple[int, float]]:
    results = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"query", "rank", "index", "score"}:
            raise ValueError(f"{path}: expected query,rank,index,score")
        for row in reader:
            key = int(row["query"]), int(row["rank"])
            value = int(row["index"]), float(row["score"])
            if key in results or min(key) < 0 or value[0] < -1 or not math.isfinite(value[1]):
                raise ValueError(f"{path}: duplicate/invalid key, invalid ID or non-finite score at {key}")
            results[key] = value
    if not results:
        raise ValueError(f"{path}: empty result dump")
    by_query: dict[int, list[tuple[int, int, float]]] = {}
    for (query, rank), (index, score) in results.items():
        by_query.setdefault(query, []).append((rank, index, score))
    count = len(by_query) if queries is None else queries
    if sorted(by_query) != list(range(count)):
        raise ValueError(f"{path}: missing queries or unexpected query count (expected {count})")
    ranks = len(by_query[0]) if k is None else k
    if not 1 <= ranks <= 32:
        raise ValueError(f"{path}: k must be in [1,32]")
    for query, entries in by_query.items():
        entries.sort()
        if [entry[0] for entry in entries] != list(range(ranks)):
            raise ValueError(f"{path}: missing or unexpected ranks for query {query} (expected k={ranks})")
        ids = [index for _, index, _ in entries if index >= 0]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate result IDs for query {query}")
        invalid_seen, previous = False, math.inf
        for _, index, score in entries:
            if index < 0:
                invalid_seen = True
            else:
                if invalid_seen or score > previous:
                    raise ValueError(f"{path}: invalid padding order or unsorted scores for query {query}")
                previous = score
    return results


def compare(left: Path, right: Path, tolerance: float, allow_ties: bool,
            queries: int | None = None, k: int | None = None) -> tuple[int, int]:
    reference, actual = read(left, queries, k), read(right, queries, k)
    if reference.keys() != actual.keys():
        raise ValueError("Query/rank keys differ: a query was lost, duplicated or reordered incorrectly")
    failures, ties = [], 0
    for key, (index, score) in reference.items():
        other_index, other_score = actual[key]
        if abs(score - other_score) > tolerance or (index < 0) != (other_index < 0):
            failures.append(f"{key}: {reference[key]} versus {actual[key]}")
        elif index != other_index:
            # Same-score IDs can reorder or cross the k-th boundary because
            # BF16 local_sort ties depend on reduction order. These are reported
            # separately; score agreement cannot prove alternative IDs correct.
            ties += 1
            if not allow_ties:
                failures.append(f"{key}: IDs differ at the same score ({index} versus {other_index})")
    if failures:
        raise ValueError(f"{len(failures)} mismatched result entries:\n  " + "\n  ".join(failures[:12]))
    return len(reference), ties


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--score-atol", type=float, default=0.0)
    parser.add_argument("--allow-ties", action="store_true", help="Report, but accept, different IDs at equal ranked scores")
    parser.add_argument("--queries", type=int, help="Require this many consecutive original queries")
    parser.add_argument("--k", type=int, help="Require this many ranks for every query")
    args = parser.parse_args()
    if not math.isfinite(args.score_atol) or args.score_atol < 0:
        parser.error("--score-atol must be finite and non-negative")
    if args.queries is not None and args.queries <= 0:
        parser.error("--queries must be positive")
    if args.k is not None and not 1 <= args.k <= 32:
        parser.error("--k must be in [1,32]")
    try:
        count, ties = compare(args.reference, args.actual, args.score_atol, args.allow_ties, args.queries, args.k)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Result comparison failed: {error}\n")
    print(f"Compared {count} entries: scores and valid-result counts agree; {ties} equal-score ID differences.")
    if ties:
        print("Inspect recall and tie differences before accepting a hardware result; alternative IDs were not independently rescored.")


if __name__ == "__main__":
    main()
