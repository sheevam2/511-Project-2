#!/usr/bin/env python3
"""
Part 2 sanity test:
- Scans known HDF5 datasets at repo root
- Uses f['neighbors'] as ground-truth indices when available
- Runs a small HNSW search on a sample and reports Recall@1
"""

import os
import time
import numpy as np
import faiss
import h5py


DATASET_FILES = {
    "MNIST": "mnist-784-euclidean.hdf5",
    "NYTimes": "nytimes-256-angular.hdf5",
    "SIFT": "sift-128-euclidean.hdf5",
    "GloVe-100": "glove-100-angular.hdf5",
}

SAMPLE_SIZE = 1000


def maybe_normalize_for_angular(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def main():
    root = os.path.dirname(os.path.dirname(__file__))

    for name, fname in DATASET_FILES.items():
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            continue
        with h5py.File(path, 'r') as f:
            train = f['train'][:].astype('float32')
            test = f['test'][:].astype('float32')
            has_neighbors = 'neighbors' in f
            neighbors = f['neighbors'][:] if has_neighbors else None

        sample = min(SAMPLE_SIZE, len(test))
        test_q = test[:sample]

        # Angular datasets normalized for cosine/IP equivalence
        is_angular = 'angular' in fname
        if is_angular:
            train_use = maybe_normalize_for_angular(train)
            test_use = maybe_normalize_for_angular(test_q)
        else:
            train_use = train
            test_use = test_q

        dim = train.shape[1]
        index = faiss.IndexHNSWFlat(dim, 12)
        index.hnsw.efConstruction = 200
        index.add(train_use)
        index.hnsw.efSearch = 100

        t0 = time.time()
        _, pred = index.search(test_use, 1)
        q_time = time.time() - t0
        qps = sample / q_time if q_time > 0 else float('inf')

        if neighbors is not None:
            gt_idx = neighbors[:sample, :1]
        else:
            # Exact fallback
            base = faiss.IndexFlatIP(dim) if is_angular else faiss.IndexFlatL2(dim)
            base.add(train_use)
            _, gt_idx = base.search(test_use, 1)

        recall = float(np.mean(pred.reshape(-1) == gt_idx.reshape(-1)))

        print(f"\n=== {name} sanity check ===")
        print(f"Queries: {sample}")
        print(f"Recall@1: {recall:.4f}")
        print(f"QPS: {qps:.0f}")


if __name__ == '__main__':
    main()


