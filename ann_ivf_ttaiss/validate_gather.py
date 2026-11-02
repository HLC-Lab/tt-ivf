#!/usr/bin/env python3
"""Validate ann_ivf_ttaiss results against an exact per-query IVF search.

The gather implementation compares every query only with the nprobe lists the
device coarse search selected for it. Inputs come from one run:
--results-output (query,rank,index,score), --selections-output ([Q][nprobe]
list IDs) and --assignment-output ([N][1] list per vector). Checks:
  1. every returned ID belongs to one of the query's selected lists;
  2. completeness: each ranked device score equals the exact ranked score of
     the query's own lists within one BF16 ulp (host: L2-normalized,
     round-to-nearest-even BF16 inputs, FP32 products/sums per list);
  3. every returned ID rescored on the host is within one BF16 ulp.
  4. with --centroids: the selected lists are the query's top-nprobe centroids,
     which checks the device coarse top-32/top-64 and the nprobe choice. The
     best unselected score may exceed the worst selected one by at most 1.5
     BF16 ulp: each side is a separately rounded BF16 score (0.5 ulp each)
     and host/C++ normalization sums differ in order, which can move single
     BF16 inputs by one step.
With --reranked (results of --rerank: FP32 scores of the device top-32), the
host uses the unrounded normalized FP32 vectors: returned scores must match an
FP32 recomputation within 1e-5, and ranks below the exact per-query search
("re-ranking misses": a true neighbour outside the device BF16 top-32) are
counted and reported, not failed.
Recall@k against the dataset ground truth is reported for the device result
and for the exact per-query search.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_results import bf16_ulp, load_bin, normalize, read_dump, to_bf16  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="--results-output CSV")
    parser.add_argument("selections", type=Path, help="--selections-output binary")
    parser.add_argument("assignment", type=Path, help="--assignment-output binary")
    parser.add_argument("--dataset", default="glove-100-angular")
    parser.add_argument("--reranked", action="store_true", help="results come from --rerank (FP32 scores)")
    parser.add_argument("--centroids", type=Path, help="[nlist][D] float32 centroid file: also check coarse selections")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("ANN_DATA_DIR", Path(__file__).resolve().parents[1] / "data")))
    args = parser.parse_args()

    results = read_dump(args.results)
    selections = load_bin(args.selections, np.uint32).astype(np.int64)
    assignment = load_bin(args.assignment, np.uint32).reshape(-1).astype(np.int64)
    queries, nprobe = selections.shape
    if len(results) != queries:
        raise SystemExit(f"Result dump has {len(results)} queries; selections have {queries}")
    k = len(results[0])
    datasets = args.data_dir / "datasets"
    # Device BF16 inputs; with --reranked the FP32 vectors the host re-scores with.
    rounding = (lambda rows: rows) if args.reranked else to_bf16
    train = rounding(normalize(load_bin(datasets / f"{args.dataset}_train.bin", np.float32)))
    query_rows = rounding(normalize(load_bin(datasets / f"{args.dataset}_queries.bin", np.float32)[:queries]))
    coarse_queries = to_bf16(normalize(load_bin(datasets / f"{args.dataset}_queries.bin", np.float32)[:queries]))
    truth = load_bin(datasets / f"{args.dataset}_neighbors.bin", np.uint32)[:queries, :k]
    if train.shape[0] != assignment.shape[0]:
        raise SystemExit("Assignment size differs from the dataset")

    # Exact per-query search, list by list: top-k of every (query, list), then
    # top-k of the query's nprobe candidates.
    order = np.argsort(assignment, kind="stable")
    bounds = np.searchsorted(assignment[order], np.arange(int(assignment.max()) + 2))
    pool_scores = np.full((queries, nprobe * k), -np.inf, dtype=np.float32)
    pool_ids = np.full((queries, nprobe * k), -1, dtype=np.int64)
    q_index, j_index = np.nonzero(np.ones_like(selections, dtype=bool))
    lists = selections[q_index, j_index]
    by_list = np.argsort(lists, kind="stable")
    q_index, j_index, lists = q_index[by_list], j_index[by_list], lists[by_list]
    starts = np.flatnonzero(np.r_[True, lists[1:] != lists[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(lists)]):
        c = lists[s]
        members = order[bounds[c]:bounds[c + 1]] if c + 1 < len(bounds) else np.empty(0, dtype=np.int64)
        if members.size == 0:
            continue
        rows_q, rows_j = q_index[s:e], j_index[s:e]
        scores = train[members] @ query_rows[rows_q].T  # (members, requesters), FP32
        take = min(k, members.size)
        top = np.argpartition(-scores, take - 1, axis=0)[:take] if take < members.size else \
            np.broadcast_to(np.arange(members.size)[:, None], scores.shape)
        top_scores = np.take_along_axis(scores, top, axis=0)
        for col, (q, j) in enumerate(zip(rows_q, rows_j)):
            pool_scores[q, j * k:j * k + take] = top_scores[:, col]
            pool_ids[q, j * k:j * k + take] = members[top[:, col]]
    exact_order = np.argsort(-pool_scores, axis=1, kind="stable")[:, :k]
    exact_scores = np.take_along_axis(pool_scores, exact_order, axis=1)
    exact_ids = np.take_along_axis(pool_ids, exact_order, axis=1)

    coarse_violations, coarse_over_one, worst_coarse_ulps = 0, 0, 0.0
    if args.centroids:
        centroids = to_bf16(normalize(load_bin(args.centroids, np.float32)))
        coarse = coarse_queries @ centroids.T  # (queries, nlist), FP32 of the device BF16 inputs
        selected_mask = np.zeros_like(coarse, dtype=bool)
        np.put_along_axis(selected_mask, selections, True, axis=1)
        worst_selected = np.where(selected_mask, coarse, np.inf).min(axis=1)
        best_unselected = np.where(selected_mask, -np.inf, coarse).max(axis=1)
        for q in np.flatnonzero(np.isfinite(best_unselected)):
            gap = (best_unselected[q] - worst_selected[q]) / bf16_ulp(float(best_unselected[q]))
            worst_coarse_ulps = max(worst_coarse_ulps, gap)
            coarse_over_one += gap > 1.0
            if gap > 1.5:
                coarse_violations += 1
        print(f"[Coarse] {queries} queries: selected lists not the top-{nprobe} centroids (beyond 1.5 BF16 ulp)="
              f"{coarse_violations}; margins above 1 ulp={coarse_over_one}; "
              f"worst unselected-over-selected margin {worst_coarse_ulps:.3f} ulp")

    tolerance = (lambda value: 1e-5) if args.reranked else bf16_ulp
    membership = completeness = rescore = rescored = 0
    worst_rank_ulps = worst_rescore_ulps = 0.0
    hits_device = hits_exact = 0
    messages = []
    for q in range(queries):
        selected = set(selections[q].tolist())
        truth_set = set(truth[q].tolist())
        for rank, (index, score) in enumerate(results[q]):
            reference = float(exact_scores[q, rank])
            if index < 0:
                if math.isfinite(reference):
                    completeness += 1
                    messages.append(f"query {q} rank {rank}: device returned no ID, exact has {reference:.5f}")
                continue
            if int(assignment[index]) not in selected:
                membership += 1
                messages.append(f"query {q} rank {rank}: ID {index} is in list {assignment[index]}, not selected")
            if not math.isfinite(reference):
                completeness += 1
                continue
            ulps = abs(score - reference) / tolerance(reference)
            worst_rank_ulps = max(worst_rank_ulps, ulps)
            if ulps > 1.0:
                completeness += 1
                messages.append(f"query {q} rank {rank}: device {score:.6f} vs exact {reference:.6f} ({ulps:.2f} ulp)")
            value = float(train[index].astype(np.float64) @ query_rows[q].astype(np.float64))
            rescored += 1
            rescore_ulps = abs(score - value) / tolerance(value)
            worst_rescore_ulps = max(worst_rescore_ulps, rescore_ulps)
            if rescore_ulps > 1.0:
                rescore += 1
                messages.append(f"query {q} id {index}: reported {score:.6f} vs rescored {value:.6f}")
            hits_device += index in truth_set
        hits_exact += len(truth_set & {int(i) for i in exact_ids[q] if i >= 0})
    unit = "x 1e-5" if args.reranked else "BF16 ulp"
    label = "re-ranking misses (below the exact search)" if args.reranked else "ranks off the exact per-query search"
    print(f"[Gather] {queries} queries, nprobe {nprobe}, k {k}: membership violations={membership}; "
          f"{label} by >1 {unit}={completeness} (worst {worst_rank_ulps:.3f}); "
          f"rescored {rescored} IDs, outside tolerance={rescore} (worst {worst_rescore_ulps:.3f} {unit})")
    print(f"[Recall@{k}] device{' + FP32 re-ranking' if args.reranked else ''}={hits_device / (queries * k):.6f}; "
          f"exact per-query IVF ({'FP32 inputs' if args.reranked else 'BF16 inputs, FP32 ties'})="
          f"{hits_exact / (queries * k):.6f}")
    for message in messages[:12]:
        print("  " + message, file=sys.stderr)
    if args.reranked:  # misses are the method's approximation: reported, not failed
        return 1 if membership or rescore or coarse_violations else 0
    return 1 if membership or completeness or rescore or coarse_violations else 0


if __name__ == "__main__":
    sys.exit(main())
