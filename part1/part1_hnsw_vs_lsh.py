import os
import time
import json
from typing import List, Dict, Tuple

import faiss
import h5py
import numpy as np


SAMPLE_SIZE = 1000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH_SWEEP = [10, 50, 100, 200]
LSH_NBITS_SWEEP = [32, 64, 512, 768]


def _load_sift(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    h5_path = os.path.join(root_dir, "sift-128-euclidean.hdf5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Missing dataset: {h5_path}. Place 'sift-128-euclidean.hdf5' at the repo root."
        )
    with h5py.File(h5_path, "r") as f:
        train = f["train"][:]
        test = f["test"][:]
    return train.astype(np.float32), test.astype(np.float32)


def _compute_ground_truth(train: np.ndarray, test: np.ndarray, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Computing ground truth on {sample_size} queries with exact L2 search...")
    test_sample = test[:sample_size].astype(np.float32)
    dim = train.shape[1]
    flat = faiss.IndexFlatL2(dim)
    flat.add(train.astype(np.float32))
    _, gt = flat.search(test_sample, 1)
    print("Ground truth computed")
    return gt, test_sample


def _benchmark_hnsw(train: np.ndarray, test_q: np.ndarray, gt: np.ndarray) -> List[Dict]:
    print("Benchmarking HNSW...")
    dim = train.shape[1]

    # Build once (graph depends on M and efConstruction only)
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    t0 = time.time()
    index.add(train)
    build_time = time.time() - t0
    print(f"  Build done in {build_time:.2f}s for M={HNSW_M}, efC={HNSW_EF_CONSTRUCTION}")

    results: List[Dict] = []
    for efs in HNSW_EF_SEARCH_SWEEP:
        index.hnsw.efSearch = efs
        t1 = time.time()
        _, pred = index.search(test_q, 1)
        q_time = time.time() - t1
        qps = len(test_q) / q_time if q_time > 0 else float("inf")
        recall = float(np.mean(pred.reshape(-1) == gt.reshape(-1)))
        print(f"  efSearch={efs:<4} -> recall={recall:.4f}, qps={qps:.0f}")
        results.append({
            "M": HNSW_M,
            "efConstruction": HNSW_EF_CONSTRUCTION,
            "efSearch": efs,
            "recall": recall,
            "qps": qps,
            "build_time": build_time,
        })
    return results


def _benchmark_lsh(train: np.ndarray, test_q: np.ndarray, gt: np.ndarray) -> List[Dict]:
    print("Benchmarking LSH...")
    dim = train.shape[1]

    results: List[Dict] = []
    for nbits in LSH_NBITS_SWEEP:
        index = faiss.IndexLSH(dim, nbits)
        t0 = time.time()
        index.add(train)
        build_time = time.time() - t0

        t1 = time.time()
        _, pred = index.search(test_q, 1)
        q_time = time.time() - t1
        qps = len(test_q) / q_time if q_time > 0 else float("inf")
        recall = float(np.mean(pred.reshape(-1) == gt.reshape(-1)))
        print(f"  nbits={nbits:<4}  -> recall={recall:.4f}, qps={qps:.0f}, build={build_time:.2f}s")
        results.append({
            "nbits": nbits,
            "recall": recall,
            "qps": qps,
            "build_time": build_time,
        })
    return results


def _save_results(hnsw_results: List[Dict], lsh_results: List[Dict], out_path: str) -> None:
    payload = {
        "hnsw_results": hnsw_results,
        "lsh_results": lsh_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


def main() -> None:
    print("=== Part 1: HNSW vs LSH Benchmark (results only) ===")
    root = os.path.dirname(os.path.dirname(__file__))

    train, test = _load_sift(root)
    print(f"Loaded SIFT: train={len(train):,}, test={len(test):,}, dim={train.shape[1]}")

    gt, test_sample = _compute_ground_truth(train, test, SAMPLE_SIZE)

    hnsw_results = _benchmark_hnsw(train, test_sample, gt)
    lsh_results = _benchmark_lsh(train, test_sample, gt)

    out_json = os.path.join(os.path.dirname(__file__), "part1_results.json")
    _save_results(hnsw_results, lsh_results, out_json)

    print("\nRun part1/graph1.py to generate the figure from the saved JSON.")


if __name__ == "__main__":
    main()

import faiss
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os

def load_dataset():
    """
    Load SIFT1M dataset from HDF5 file
    Returns: train_embeddings, test_embeddings
    """
    print("Loading SIFT1M dataset...")
    with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
        train_embeddings = f['train'][:]  # Database vectors
        test_embeddings = f['test'][:]    # Query vectors
    
    print(f"Loaded {len(train_embeddings)} training vectors and {len(test_embeddings)} test vectors")
    print(f"Vector dimension: {train_embeddings.shape[1]}")
    
    return train_embeddings.astype('float32'), test_embeddings.astype('float32')

def compute_ground_truth(train_embeddings, test_embeddings, sample_size=1000):
    """
    Compute ground truth using brute force exact search on a sample
    Returns: ground_truth_indices (array of shape [sample_size, 1])
    """
    print(f"Computing ground truth using brute force on {sample_size} sample queries...")
    
    # Use only a sample of test queries for faster computation
    test_sample = test_embeddings[:sample_size]
    
    # Create exact search index
    print("  Building exact search index...")
    dimension = train_embeddings.shape[1]
    exact_index = faiss.IndexFlatL2(dimension)
    exact_index.add(train_embeddings)
    print("  ✓ Index built")
    
    # Search for exact nearest neighbors
    print("  Searching for exact nearest neighbors...")
    _, ground_truth_indices = exact_index.search(test_sample, 1)
    print("  ✓ Ground truth search complete")
    
    print(f"Ground truth computed for {len(test_sample)} queries")
    return ground_truth_indices, test_sample

def benchmark_hnsw(train_embeddings, test_sample, ground_truth_indices):
    """
    Benchmark HNSW with different efSearch values
    Returns: list of (recall, qps, efSearch) tuples
    """
    print("\n=== HNSW Benchmarking ===")
    
    dimension = train_embeddings.shape[1]
    efSearch_values = [10, 50, 100, 200]
    results = []
    
    for i, efSearch in enumerate(efSearch_values, 1):
        print(f"[{i}/{len(efSearch_values)}] Testing HNSW with efSearch={efSearch}...")
        
        # Create HNSW index with M=32, efConstruction=200
        print("  Creating HNSW index...")
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = efSearch
        
        # Build index
        print("  Building index...")
        start_time = time.time()
        index.add(train_embeddings)
        build_time = time.time() - start_time
        print(f"  ✓ Index built in {build_time:.2f} seconds")
        
        # Perform queries on sample
        print("  Running queries...")
        start_time = time.time()
        _, predicted_indices = index.search(test_sample, 1)  # k=1
        query_time = time.time() - start_time
        
        # Calculate metrics
        qps = len(test_sample) / query_time
        correct_predictions = np.sum(predicted_indices == ground_truth_indices)
        recall = correct_predictions / len(test_sample)
        
        print(f"  ✓ QPS: {qps:.2f}, Recall: {recall:.4f}")
        results.append((recall, qps, efSearch))
    
    print("✓ HNSW benchmarking complete")
    return results

def benchmark_lsh(train_embeddings, test_sample, ground_truth_indices):
    """
    Benchmark LSH with different nbits values
    Returns: list of (recall, qps, nbits) tuples
    """
    print("\n=== LSH Benchmarking ===")
    
    dimension = train_embeddings.shape[1]
    nbits_values = [32, 64, 512, 768]
    results = []
    
    for i, nbits in enumerate(nbits_values, 1):
        print(f"[{i}/{len(nbits_values)}] Testing LSH with nbits={nbits}...")
        
        # Create LSH index
        print("  Creating LSH index...")
        index = faiss.IndexLSH(dimension, nbits)
        
        # Build index
        print("  Building index...")
        start_time = time.time()
        index.add(train_embeddings)
        build_time = time.time() - start_time
        print(f"  ✓ Index built in {build_time:.2f} seconds")
        
        # Perform queries on sample
        print("  Running queries...")
        start_time = time.time()
        _, predicted_indices = index.search(test_sample, 1)  # k=1
        query_time = time.time() - start_time
        
        # Calculate metrics
        qps = len(test_sample) / query_time
        correct_predictions = np.sum(predicted_indices == ground_truth_indices)
        recall = correct_predictions / len(test_sample)
        
        print(f"  ✓ QPS: {qps:.2f}, Recall: {recall:.4f}")
        results.append((recall, qps, nbits))
    
    print("✓ LSH benchmarking complete")
    return results

def plot_results_matplotlib(hnsw_results, lsh_results):
    """
    Create matplotlib plot comparing HNSW vs LSH
    """
    print("Creating matplotlib plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Extract data
    hnsw_recalls = [r[0] for r in hnsw_results]
    hnsw_qps = [r[1] for r in hnsw_results]
    hnsw_params = [r[2] for r in hnsw_results]
    
    lsh_recalls = [r[0] for r in lsh_results]
    lsh_qps = [r[1] for r in lsh_results]
    lsh_params = [r[2] for r in lsh_results]
    
    # Plot HNSW
    plt.plot(hnsw_recalls, hnsw_qps, 'o-', label='HNSW', linewidth=2, markersize=8)
    for i, (recall, qps, param) in enumerate(hnsw_results):
        plt.annotate(f'efSearch={param}', (recall, qps), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot LSH
    plt.plot(lsh_recalls, lsh_qps, 's-', label='LSH', linewidth=2, markersize=8)
    for i, (recall, qps, param) in enumerate(lsh_results):
        plt.annotate(f'nbits={param}', (recall, qps), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('1-Recall@1', fontsize=12)
    plt.ylabel('QPS (Queries Per Second)', fontsize=12)
    plt.title('HNSW vs LSH: QPS vs Recall Comparison', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('part1_qps_vs_recall_matplotlib.png', dpi=300, bbox_inches='tight')
    print("✓ Matplotlib plot saved as 'part1_qps_vs_recall_matplotlib.png'")
    
    # Display plot
    plt.show()


def save_results(hnsw_results, lsh_results):
    """
    Save results to JSON file for later analysis
    """
    print("Saving results to JSON...")
    
    results = {
        'hnsw_results': [
            {'recall': r[0], 'qps': r[1], 'efSearch': r[2]} 
            for r in hnsw_results
        ],
        'lsh_results': [
            {'recall': r[0], 'qps': r[1], 'nbits': r[2]} 
            for r in lsh_results
        ]
    }
    
    with open('part1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to 'part1_results.json'")

def main():
    """
    Main function to run Part 1 benchmarking
    """
    print("=== Part 1: HNSW vs LSH Benchmarking ===")
    print("This will test HNSW and LSH algorithms on SIFT1M dataset")
    print("Using 1000 sample queries for faster computation\n")
    
    # Load dataset
    print("Step 1/5: Loading dataset...")
    train_embeddings, test_embeddings = load_dataset()
    print("✓ Dataset loaded\n")
    
    # Compute ground truth on sample
    print("Step 2/5: Computing ground truth...")
    ground_truth_indices, test_sample = compute_ground_truth(train_embeddings, test_embeddings, sample_size=1000)
    print("✓ Ground truth computed\n")
    
    # Benchmark HNSW
    print("Step 3/5: Benchmarking HNSW...")
    hnsw_results = benchmark_hnsw(train_embeddings, test_sample, ground_truth_indices)
    print("✓ HNSW benchmarking complete\n")
    
    # Benchmark LSH
    print("Step 4/5: Benchmarking LSH...")
    lsh_results = benchmark_lsh(train_embeddings, test_sample, ground_truth_indices)
    print("✓ LSH benchmarking complete\n")
    
    # Generate plot
    print("Step 5/5: Generating plot and saving results...")
    plot_results_matplotlib(hnsw_results, lsh_results)
    
    # Save results
    save_results(hnsw_results, lsh_results)
    
    print("\n🎉 === Part 1 Complete === 🎉")
    print("Generated files:")
    print("- part1_qps_vs_recall_matplotlib.png")
    print("- part1_results.json")
    print("\nYou can now use the plot in your report!")

if __name__ == "__main__":
    main()
