import os
import time
import json
from typing import Dict, List, Tuple, Optional

import faiss
import h5py
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
# Expected HDF5 datasets (train/test) from ann-benchmarks
# Put the .hdf5 files at repo root. Script will skip datasets not found.
DATASET_FILES = {
    "MNIST": "mnist-784-euclidean.hdf5",
    "NYTimes": "nytimes-256-angular.hdf5",
    "SIFT": "sift-128-euclidean.hdf5",
    "GloVe-100": "glove-100-angular.hdf5",
}

# Distance type: for angular we use inner product after normalizing vectors
DATASET_METRIC = {
    "MNIST": "l2",
    "NYTimes": "angular",
    "SIFT": "l2",
    "GloVe-100": "angular",
}

M_VALUES = [4, 8, 12, 24, 48]
TUNE_EF_CONSTRUCTION = [100, 200]
TUNE_EF_SEARCH = [50, 100]
SAMPLE_SIZE = 1000  # number of query vectors used for recall/QPS

# ------------------------------
# Utilities
# ------------------------------

def _load_hdf5(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        train = f["train"][:]
        test = f["test"][:]
    return train.astype("float32"), test.astype("float32")


def _maybe_normalize_for_angular(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def _compute_ground_truth(train: np.ndarray, test: np.ndarray, metric: str, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    print(f"  Computing ground truth on {sample_size} queries using exact search ...")
    test_sample = test[:sample_size]

    if metric == "angular":
        # Normalize for cosine/Angular, then use inner product equivalence
        train_n = _maybe_normalize_for_angular(train)
        test_n = _maybe_normalize_for_angular(test_sample)
        index = faiss.IndexFlatIP(train.shape[1])
        index.add(train_n)
        _, gt = index.search(test_n, 1)
    else:
        index = faiss.IndexFlatL2(train.shape[1])
        index.add(train)
        _, gt = index.search(test_sample, 1)

    print("  ✓ Ground truth computed")
    return gt, test_sample


def _evaluate_hnsw(train: np.ndarray, test_q: np.ndarray, gt: np.ndarray, metric: str, M: int, efc: int, efs: int) -> Tuple[float, float, float]:
    # Choose flat quantizer according to metric for distance computation inside HNSW
    # For angular, HNSWFlat uses L2 distances on vectors as-is. Empirically, cosine works well when vectors are normalized.
    if metric == "angular":
        train_use = _maybe_normalize_for_angular(train)
        test_use = _maybe_normalize_for_angular(test_q)
    else:
        train_use = train
        test_use = test_q

    dim = train.shape[1]
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efc

    t0 = time.time()
    index.add(train_use)
    build_time = time.time() - t0

    index.hnsw.efSearch = efs

    t1 = time.time()
    _, pred = index.search(test_use, 1)
    q_time = time.time() - t1

    qps = len(test_use) / q_time if q_time > 0 else float("inf")
    recall = float(np.mean(pred.reshape(-1) == gt.reshape(-1)))
    return recall, qps, build_time


def _tune_params(train: np.ndarray, test_q: np.ndarray, gt: np.ndarray, metric: str) -> Tuple[int, int, Dict[str, float]]:
    print("  Tuning efConstruction and efSearch with a tiny grid ...")
    best = None
    best_stats = {}
    for efc in TUNE_EF_CONSTRUCTION:
        for efs in TUNE_EF_SEARCH:
            recall, qps, build_time = _evaluate_hnsw(train, test_q, gt, metric, M=12, efc=efc, efs=efs)
            key = f"efc={efc}, efs={efs}"
            print(f"    {key}: recall={recall:.4f}, qps={qps:.0f}, build={build_time:.1f}s")
            candidate = (recall, qps, efc, efs)
            if best is None:
                best = candidate
                best_stats = {"recall": recall, "qps": qps, "build_time": build_time}
            else:
                # Prefer higher recall, tie-break on higher QPS
                if (candidate[0] > best[0]) or (candidate[0] == best[0] and candidate[1] > best[1]):
                    best = candidate
                    best_stats = {"recall": recall, "qps": qps, "build_time": build_time}
    _, _, best_efc, best_efs = best
    print(f"  ✓ Chosen params: efConstruction={best_efc}, efSearch={best_efs}")
    return best_efc, best_efs, best_stats


# ------------------------------
# Main benchmark per dataset
# ------------------------------

def run_dataset(name: str, path: str, metric: str, sample_size: int) -> Dict:
    print(f"\n=== Dataset: {name} ===")
    print(f"Path: {path}")
    train, test = _load_hdf5(path)
    print(f"  Loaded train={len(train):,}, test={len(test):,}, dim={train.shape[1]}")

    gt, test_sample = _compute_ground_truth(train, test, metric, sample_size)
    efc, efs, tuning_stats = _tune_params(train, test_sample, gt, metric)

    results = {
        "dataset": name,
        "metric": metric,
        "dim": int(train.shape[1]),
        "num_train": int(len(train)),
        "num_test_sample": int(len(test_sample)),
        "tuned": {"efConstruction": efc, "efSearch": efs, **tuning_stats},
        "sweep": [],  # list of {M, recall, qps, build_time}
    }

    for i, M in enumerate(M_VALUES, 1):
        print(f"  [{i}/{len(M_VALUES)}] Evaluating M={M} ...")
        recall, qps, build_time = _evaluate_hnsw(train, test_sample, gt, metric, M=M, efc=efc, efs=efs)
        print(f"     → recall={recall:.4f}, qps={qps:.0f}, build={build_time:.1f}s")
        results["sweep"].append({"M": M, "recall": recall, "qps": qps, "build_time": build_time})

    return results


# ------------------------------
# Note: Plotting removed by request — this script now ONLY saves results JSON
# ------------------------------


# ------------------------------
# Orchestration
# ------------------------------

def main() -> None:
    print("=== Part 2: HNSW Scalability Benchmark ===")
    print("This will scan for available HDF5 datasets in repo root and run benchmarks on those found.\n")

    found = []
    for name, fname in DATASET_FILES.items():
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), fname)
        if os.path.exists(path):
            found.append((name, path, DATASET_METRIC[name]))
        else:
            print(f"Skipping {name}: {fname} not found (place it at repo root to include)")

    if not found:
        print("No datasets found. Please download HDF5 files from ann-benchmarks and place them at repo root.")
        return

    all_results: List[Dict] = []
    for idx, (name, path, metric) in enumerate(found, 1):
        print(f"\nDataset {idx}/{len(found)}: {name}")
        res = run_dataset(name, path, metric, SAMPLE_SIZE)
        all_results.append(res)

    # Save metrics
    out_json = os.path.join(os.path.dirname(__file__), "part2_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {out_json}")

    print("\n🎉 Part 2 run complete. Use part2/graph2.py to generate all plots from part2_results.json.")


if __name__ == "__main__":
    main()
