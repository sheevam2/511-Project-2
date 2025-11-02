#!/usr/bin/env python3
"""
Part 1 sanity test:
- Loads SIFT HDF5 base/query from repo root
- Uses sift_groundtruth.ivecs if available; otherwise falls back to exact FlatL2
- Builds a small HNSW and reports Recall@1 against ground truth on a sample
"""

import os
import time
import numpy as np
import faiss
import h5py


def read_ivecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype='int32')
    if a.size == 0:
        raise ValueError(f"Empty ivecs file: {path}")
    dim = a[0]
    return a.reshape(-1, dim + 1)[:, 1:]


def load_sift_from_hdf5(root_dir: str):
    h5_path = os.path.join(root_dir, 'sift-128-euclidean.hdf5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing {h5_path}. Place SIFT HDF5 at repo root.")
    with h5py.File(h5_path, 'r') as f:
        train = f['train'][:].astype('float32')
        test = f['test'][:].astype('float32')
    return train, test


def get_ground_truth_indices(root_dir: str, num_queries: int, dim: int) -> np.ndarray:
    ivecs_path = os.path.join(root_dir, 'sift_groundtruth.ivecs')
    if os.path.exists(ivecs_path):
        gt = read_ivecs(ivecs_path)
        if gt.shape[0] < num_queries:
            raise ValueError(f"Ground truth has {gt.shape[0]} queries, requested {num_queries}")
        return gt[:num_queries, :]
    # Fallback: compute exact top-1 with FlatL2
    print("Ground truth ivecs not found, computing exact ground truth with FlatL2...")
    return None


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    train, test = load_sift_from_hdf5(root)
    dim = train.shape[1]

    sample_size = min(1000, len(test))
    test_q = test[:sample_size]

    gt = get_ground_truth_indices(root, sample_size, dim)
    if gt is None:
        flat = faiss.IndexFlatL2(dim)
        flat.add(train)
        _, gt_idx = flat.search(test_q, 1)
    else:
        gt_idx = gt[:, :1]

    # Build a modest HNSW and evaluate
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(train)
    index.hnsw.efSearch = 100

    t0 = time.time()
    _, pred = index.search(test_q, 1)
    q_time = time.time() - t0
    qps = sample_size / q_time if q_time > 0 else float('inf')
    recall = float(np.mean(pred.reshape(-1) == gt_idx.reshape(-1)))

    print("\n=== Part 1 SIFT Sanity Check ===")
    print(f"Queries: {sample_size}")
    print(f"Recall@1: {recall:.4f}")
    print(f"QPS: {qps:.0f}")


if __name__ == '__main__':
    main()


