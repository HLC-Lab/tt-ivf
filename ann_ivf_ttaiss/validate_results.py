#!/usr/bin/env python3
"""Validate a cluster-major result dump against a batch-major reference.

Both dumps come from `--results-output` (query,rank,index,score in original
query order). Checks:
  1. same queries/ranks, valid padding, no duplicate IDs, sorted scores;
  2. identical score at every rank (exact BF16 equality by default);
  3. every returned ID in both dumps is rescored on the host from the same
     L2-normalized, round-to-nearest-even BF16 inputs the device receives.
     The reported score must equal the rescored value within one BF16 ulp
     (device fp32 accumulation and packer rounding). This independently
     validates IDs that differ only through equal-score ties.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np


def read_dump(path: Path) -> dict[int, list[tuple[int, float]]]:
    rows: dict[int, dict[int, tuple[int, float]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["query", "rank", "index", "score"]:
            raise ValueError(f"{path}: expected header query,rank,index,score")
        for row in reader:
            query, rank = int(row["query"]), int(row["rank"])
            index, score = int(row["index"]), float(row["score"])
            if rank in rows.setdefault(query, {}) or index < -1 or not math.isfinite(score):
                raise ValueError(f"{path}: duplicate or invalid entry at query {query}, rank {rank}")
            rows[query][rank] = (index, score)
    if not rows or sorted(rows) != list(range(len(rows))):
        raise ValueError(f"{path}: queries are missing or not consecutive")
    result = {}
    k = len(rows[0])
    for query, ranks in rows.items():
        if sorted(ranks) != list(range(k)):
            raise ValueError(f"{path}: query {query} does not have ranks 0..{k - 1}")
        entries = [ranks[r] for r in range(k)]
        ids = [i for i, _ in entries if i >= 0]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate IDs for query {query}")
        padding, previous = False, math.inf
        for index, score in entries:
            if index < 0:
                padding = True
            elif padding or score > previous:
                raise ValueError(f"{path}: invalid padding order or unsorted scores for query {query}")
            else:
                previous = score
        result[query] = entries
    return result


def load_bin(path: Path, dtype) -> np.ndarray:
    with path.open("rb") as handle:
        n, d = np.frombuffer(handle.read(8), dtype=np.uint32)
        data = np.frombuffer(handle.read(), dtype=dtype)
    if data.size != int(n) * int(d):
        raise ValueError(f"{path}: size does not match its [N][D] header")
    return data.reshape(int(n), int(d))


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 (ties to even) and widen back to float32."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def normalize(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float32, copy=True)
    norms = np.sqrt(np.sum(rows * rows, axis=1, dtype=np.float32))
    nonzero = norms > 0
    rows[nonzero] /= norms[nonzero, None]
    return rows


def bf16_ulp(value: float) -> float:
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7) if value else 2.0 ** -133


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="batch-major (ann_ivf_mem_log) dump")
    parser.add_argument("actual", type=Path, help="cluster-major dump")
    parser.add_argument("--dataset", default="glove-100-angular")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("ANN_DATA_DIR", Path(__file__).resolve().parents[1] / "data")))
    parser.add_argument("--score-atol", type=float, default=0.0, help="allowed reference/actual score difference")
    parser.add_argument("--no-rescore", action="store_true", help="skip the host rescoring of returned IDs")
    parser.add_argument("--report-only", action="store_true",
                        help="report deviation statistics (e.g. reduced math fidelity) instead of failing")
    args = parser.parse_args()

    reference, actual = read_dump(args.reference), read_dump(args.actual)
    if reference.keys() != actual.keys() or len(reference[0]) != len(actual[0]):
        raise SystemExit("Query count or k differs between the dumps")
    failures, tie_ids = [], 0
    for query, entries in reference.items():
        for rank, ((ref_id, ref_score), (act_id, act_score)) in enumerate(zip(entries, actual[query])):
            if abs(ref_score - act_score) > args.score_atol or (ref_id < 0) != (act_id < 0):
                failures.append(f"query {query} rank {rank}: {ref_id}/{ref_score} versus {act_id}/{act_score}")
            elif ref_id != act_id:
                tie_ids += 1
    entries_total = sum(len(e) for e in reference.values())
    deltas = [abs(r[1] - a[1]) for q in reference for r, a in zip(reference[q], actual[q]) if r[0] >= 0 and a[0] >= 0]
    overlap = sum(len({i for i, _ in reference[q] if i >= 0} & {i for i, _ in actual[q] if i >= 0})
                  for q in reference) / max(1, sum(len([i for i, _ in reference[q] if i >= 0]) for q in reference))
    print(f"[Deviation] max |score delta|={max(deltas, default=0.0):.6g}, mean={sum(deltas) / max(1, len(deltas)):.6g}, "
          f"ranked ID-set overlap with reference={overlap:.6f}")
    print(f"[Compare] {len(reference)} queries, {entries_total} ranked entries; "
          f"score mismatches={len(failures)}; equal-score ID differences={tie_ids}")

    rescore_failures, rescored, worst_ulps = [], 0, 0.0
    if not args.no_rescore:
        datasets = args.data_dir / "datasets"
        train = load_bin(datasets / f"{args.dataset}_train.bin", np.float32)
        queries = load_bin(datasets / f"{args.dataset}_queries.bin", np.float32)[: len(reference)]
        train_bf = to_bf16(normalize(train))
        query_bf = to_bf16(normalize(queries))
        for name, dump in (("reference", reference), ("actual", actual)):
            for query, entries in dump.items():
                ids = np.array([i for i, _ in entries if i >= 0], dtype=np.int64)
                if ids.size and (ids.max() >= train_bf.shape[0]):
                    rescore_failures.append(f"{name} query {query}: ID outside the dataset")
                    continue
                exact = train_bf[ids].astype(np.float64) @ query_bf[query].astype(np.float64)
                for (index, score), value in zip([e for e in entries if e[0] >= 0], exact):
                    rescored += 1
                    worst_ulps = max(worst_ulps, abs(score - value) / bf16_ulp(value)) if name == "actual" else worst_ulps
                    if abs(score - value) > bf16_ulp(value):
                        rescore_failures.append(
                            f"{name} query {query} id {index}: reported {score} versus rescored {value:.6f}")
        print(f"[Rescore] {rescored} returned IDs rescored from BF16 inputs; "
              f"outside one BF16 ulp={len(rescore_failures)}; actual worst={worst_ulps:.3f} ulp")

    if args.report_only:
        return 0
    for message in (failures + rescore_failures)[:12]:
        print("  " + message, file=sys.stderr)
    return 1 if failures or rescore_failures else 0


if __name__ == "__main__":
    sys.exit(main())
