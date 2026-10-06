import faiss
import numpy as np
import time
import argparse
import sys
import os
import h5py
import csv
from pathlib import Path
# Shared data/result helpers live in the project root's tools/ package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.data_layout import dataset_file, centroid_file, result_file

def append_to_csv(filepath, run_id, program_type, dataset, n, d, nlist, nprobe, k, batch_size, avg_latency_s, qps, recall_fraction):
    """Appends results to the shared comma-separated CSV file."""
    file_exists = os.path.isfile(filepath)
    avg_latency_us = avg_latency_s * 1000000.0
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        # Write headers if the file is being created for the first time
        if not file_exists:
            writer.writerow(['RunID', 'Type', 'Dataset', 'N', 'D', 'NList', 'NProbe', 'K', 'BatchSize', 'AvgLatency_us', 'QPS', 'Recall@K'])
        writer.writerow([
            run_id,
            program_type,
            dataset,
            n,
            d,
            nlist,
            nprobe,
            k,
            batch_size,
            f"{avg_latency_us:.2f}",
            f"{qps:.2f}",
            f"{recall_fraction:.6f}"
        ])

def run_official_faiss_benchmark():
    parser = argparse.ArgumentParser(description="Official FAISS GloVe-100 Benchmark")
    parser.add_argument('--dataset', type=str, default='glove-100-angular.hdf5')
    parser.add_argument('--nlist', type=int, default=512, help="Number of clusters")
    parser.add_argument('--nprobe', type=int, default=1, help="Number of clusters to search")
    parser.add_argument('--k', type=int, default=10, help="Top-K vectors to retrieve")
    parser.add_argument('--num_queries', type=int, default=10000)
    parser.add_argument('--gpu', action='store_true', help="Run FAISS on GPU")
    parser.add_argument('--run_id', type=str, default='run_1', help="Identifier for this specific run")
    parser.add_argument('--runs', type=int, default=1, help="Number of times to repeat the search")
    args = parser.parse_args()

    if Path(args.dataset).name == args.dataset:
        args.dataset = str(dataset_file(args.dataset))
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset '{args.dataset}' not found.")
        sys.exit(1)

    print("Extracting Text")
    # Extract base dataset name for file naming (e.g., 'glove-100-angular')
    dataset_base = os.path.splitext(os.path.basename(args.dataset))[0]

    # FAISS summary uses the same columns as the C++ benchmark results.
    shared_csv = result_file('faiss', f"results_{dataset_base}.csv")

    print(f"\n>>> Loading dataset '{args.dataset}' into RAM...")
    with h5py.File(args.dataset, 'r') as f:
        xb = np.array(f['train']).astype('float32')
        xq_all = np.array(f['test']).astype('float32')
        gt_neighbors_all = np.array(f['neighbors'])

    num_vectors, d = xb.shape
    total_queries = min(args.num_queries, len(xq_all))
    xq_all = xq_all[:total_queries]

    # --- CRUCIAL: L2 Normalization for Cosine Similarity ---
    print(">>> L2 Normalizing vectors for Cosine Similarity...")
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq_all)

    print(f"\n--- Benchmark Configuration ---")
    print(f"Library: Official FAISS")
    print(f"Hardware: {'GPU' if args.gpu else 'CPU'}")
    print(f"Run ID: {args.run_id}")
    print(f"Index: IVF{args.nlist}, Flat | nprobe={args.nprobe}")
    print(f"Task: Recall@{args.k} on {total_queries} queries")
    print(f"Runs: {args.runs}\n")

    # 1. Initialize FAISS Index
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFScalarQuantizer(quantizer, d, args.nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)

    if args.gpu:
        print(">>> Moving index to GPU...")
        res = faiss.StandardGpuResources()
        print(">>> 2")
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # 2. Train and Extract Centroids
    print(">>> Training K-Means centroids (this takes a moment)...")
    train_start = time.perf_counter()
    index.train(xb)
    train_time = time.perf_counter() - train_start

    # --- Extract and Save Centroids ---
    centroid_filename = centroid_file(dataset_base, args.nlist)
    centroid_filename.parent.mkdir(parents=True, exist_ok=True)
    print(">>> Checking centroids...")

    if not os.path.exists(centroid_filename):
        print(">>> Extracting and saving centroids...")
        if args.gpu:
            cpu_index = faiss.index_gpu_to_cpu(index)
            centroids = cpu_index.quantizer.reconstruct_n(0, args.nlist)
        else:
            centroids = index.quantizer.reconstruct_n(0, args.nlist)

        np.save(centroid_filename, centroids)
        print(f"    Saved {args.nlist} centroids to '{centroid_filename}'\n")
    else:
        print(f"    Centroids already exist at '{centroid_filename}'. Skipping save.\n")

    # 3. Add Vectors
    print(">>> Adding vectors to index...")
    add_start = time.perf_counter()
    index.add(xb)
    add_time = time.perf_counter() - add_start

    index.nprobe = args.nprobe
    print(f">>> Index ready! (Train: {train_time:.2f}s | Add: {add_time:.2f}s)\n")

    # --- TEST 2: BATCHED THROUGHPUT ---
    print("=============================================")
    print(f" TEST 2: BATCHED SEARCH ({args.runs} runs)")
    print("=============================================")

    for r_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Run {r_idx + 1} / {args.runs} ---")

        start_time = time.perf_counter()
        _, I_batch = index.search(xq_all, args.k)
        end_time = time.perf_counter()

        batch_total_time = end_time - start_time
        batch_qps = total_queries / batch_total_time
        batch_avg_latency = batch_total_time / total_queries

        # Verify recall
        batch_recalls = []
        for r in range(total_queries):
            gt_indices = gt_neighbors_all[r, :args.k].tolist()
            retrieved_indices = I_batch[r].tolist()
            intersection = set(gt_indices).intersection(set(retrieved_indices))
            batch_recalls.append(len(intersection) / args.k)

        batch_recall = np.mean(batch_recalls)

        print(f" Average Recall@{args.k}:  {batch_recall * 100:.2f}%")
        print(f" Total Search Time:  {batch_total_time:.3f} seconds")
        print(f" Batched Speed:      {batch_qps:.2f} QPS")

        # Log to shared CSV
        program_type_batch = f"{'gpu' if args.gpu else 'cpu'}-batched"
        current_run_id = args.run_id if args.runs == 1 else f"{args.run_id}_{r_idx + 1}"
        append_to_csv(shared_csv, current_run_id, program_type_batch, dataset_base, num_vectors, d, args.nlist, args.nprobe, args.k, total_queries, batch_avg_latency, batch_qps, batch_recall)
        print(f"    -> Logged run to {shared_csv}")

    # Save per-query recall for the last run.
    tmp_csv_path = result_file('faiss', f"recall_{dataset_base}_{args.nlist}_{args.nprobe}.csv")
    print(f"\n>>> Saving per-query recall to {tmp_csv_path}...")
    with open(tmp_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['query_id', 'recall'])
        for q_id, rec in enumerate(batch_recalls):
            writer.writerow([q_id, f"{rec:.4f}"])
    print(f"    -> Successfully saved {len(batch_recalls)} individual queries to {tmp_csv_path}")

if __name__ == "__main__":
    print("Start main")
    run_official_faiss_benchmark()
