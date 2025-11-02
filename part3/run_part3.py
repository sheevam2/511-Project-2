import os
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil


def load_sift_hdf5(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        # Common ANN-Benchmarks layout
        xb = np.array(f["train"], dtype=np.float32)
        xq = np.array(f["test"], dtype=np.float32)
        gt = np.array(f["neighbors"], dtype=np.int32)
    return xb, xq, gt


def compute_recall_at1(pred_indices: np.ndarray, gt_neighbors: np.ndarray) -> float:
    # pred_indices: (nq, 1) or (nq,)
    pred = pred_indices.reshape(-1)
    gt_top1 = gt_neighbors[:, 0].reshape(-1)
    return float((pred == gt_top1).mean())


def run_single_hnsw_trial(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray,
                          M: int, efSearch: int, efConstruction: int,
                          trial_num: int) -> dict:
    """Run a single HNSW configuration and return metrics."""
    import faiss
    
    print(f"  Running HNSW trial {trial_num}: M={M}, efSearch={efSearch}, efConstruction={efConstruction}")
    
    d = xb.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    
    build_start = time.perf_counter()
    index.add(xb)
    build_time = time.perf_counter() - build_start
    print(f"  HNSW build time: {build_time:.2f}s")
    
    start = time.perf_counter()
    D, I = index.search(xq, 1)
    elapsed_s = time.perf_counter() - start
    
    recall = compute_recall_at1(I, gt)
    avg_latency_ms = (elapsed_s / xq.shape[0]) * 1000.0
    
    del index
    
    return {
        "algo": "HNSW",
        "M": M,
        "efSearch": efSearch,
        "efConstruction": efConstruction,
        "R": None,
        "L": None,
        "latency_ms": avg_latency_ms,
        "recall_at1": recall,
        "one_minus_recall": 1.0 - recall,
    }


def run_single_diskann_trial(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray,
                             R: int, L: int, build_L: int, index_dir_base: str,
                             trial_num: int) -> dict:
    """Run a single DiskANN configuration (DISK-BASED) and return metrics."""
    try:
        import diskannpy as dap
    except Exception as e:
        print(f"diskannpy import failed: {e}. Skipping DiskANN.")
        return None
    
    d = xb.shape[1]
    threads = os.cpu_count() or 4
    
    # Unique index directory for each trial
    index_dir = os.path.join(index_dir_base, f"diskann_disk_trial{trial_num}_R{R}")
    
    # Clean up old index if it exists
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    
    print(f"  Building DiskANN disk index {trial_num}: R={R}, L={L}, build_L={build_L}")
    print(f"    Index directory: {index_dir}")
    
    try:
        # Build DISK index per spec
        # Memory budgets in GB (tweakable). No PQ (pq_disk_bytes=0)
        build_start = time.perf_counter()
        dap.build_disk_index(
            data=xb,
            distance_metric="l2",
            index_directory=index_dir,
            complexity=build_L,
            graph_degree=R,
            search_memory_maximum=8.0,
            build_memory_maximum=8.0,
            num_threads=threads,
            pq_disk_bytes=0,
            vector_dtype=np.float32,
        )
        build_time = time.perf_counter() - build_start
        print(f"  DiskANN build time: {build_time:.2f}s")
    except Exception as e:
        print(f"DiskANN build failed for trial {trial_num}, R={R}: {e}")
        return None
    
    try:
        # Open DISK index for search
        # Cache a fraction of nodes to speed up; keep modest for subset/full
        num_nodes_to_cache = max(10000, xb.shape[0] // 10)
        idx = dap.StaticDiskIndex(
            index_directory=index_dir,
            num_threads=threads,
            num_nodes_to_cache=num_nodes_to_cache,
            cache_mechanism=1,
            distance_metric="l2",
            vector_dtype=np.float32,
            dimensions=d,
            index_prefix="ann",
        )
        
        start = time.perf_counter()
        resp = idx.batch_search(xq, k_neighbors=1, complexity=L, num_threads=threads)
        identifiers = np.asarray(resp.identifiers)
        
        indices = identifiers.reshape(-1, 1)
        elapsed_s = time.perf_counter() - start
        
        recall = compute_recall_at1(indices, gt)
        avg_latency_ms = (elapsed_s / xq.shape[0]) * 1000.0
        
        return {
            "algo": "DiskANN",
            "M": None,
            "efSearch": None,
            "efConstruction": None,
            "R": R,
            "L": L,
            "build_L": build_L,
            "latency_ms": avg_latency_ms,
            "recall_at1": recall,
            "one_minus_recall": 1.0 - recall,
        }
    except Exception as e:
        print(f"DiskANN search failed for trial {trial_num}, R={R}, L={L}: {e}")
        return None


def run_four_trials(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray, out_dir: str):
    """
    Run exactly 4 coordinated trials for each algorithm.
    Each trial changes all parameters together to show the latency-recall tradeoff.
    """
    # 4 trials with coordinated parameter changes
    trials = [
        # Trial 1: Low performance (fast, lower recall)
        {"hnsw": {"M": 16, "efSearch": 10, "efConstruction": 100},
         "diskann": {"R": 16, "L": 20, "build_L": 50}},
        
        # Trial 2: Medium-low performance
        {"hnsw": {"M": 32, "efSearch": 50, "efConstruction": 150},
         "diskann": {"R": 32, "L": 50, "build_L": 100}},
        
        # Trial 3: Medium-high performance
        {"hnsw": {"M": 48, "efSearch": 100, "efConstruction": 200},
         "diskann": {"R": 48, "L": 100, "build_L": 150}},
        
        # Trial 4: High performance (slower, higher recall)
        {"hnsw": {"M": 64, "efSearch": 200, "efConstruction": 300},
         "diskann": {"R": 64, "L": 200, "build_L": 200}},
    ]
    
    hnsw_results = []
    diskann_results = []
    
    print("Running 4 trials...")
    print(f"Dataset size: {xb.shape[0]} training vectors, {xq.shape[0]} query vectors\n")
    
    for i, trial in enumerate(trials, 1):
        print(f"\n{'='*60}")
        print(f"Trial {i}:")
        print(f"  HNSW: M={trial['hnsw']['M']}, efSearch={trial['hnsw']['efSearch']}, efConstruction={trial['hnsw']['efConstruction']}")
        print(f"  DiskANN: R={trial['diskann']['R']}, L={trial['diskann']['L']}, build_L={trial['diskann']['build_L']}")
        
        # Run HNSW trial
        hnsw_result = run_single_hnsw_trial(xb, xq, gt, **trial["hnsw"], trial_num=i)
        hnsw_results.append(hnsw_result)
        print(f"  HNSW Result: Recall={hnsw_result['recall_at1']:.3f}, Latency={hnsw_result['latency_ms']:.2f}ms")
        
        # Run DiskANN trial
        diskann_result = run_single_diskann_trial(
            xb, xq, gt, 
            **trial["diskann"],
            index_dir_base=os.path.join(out_dir, "diskann_indices"),
            trial_num=i
        )
        if diskann_result:
            diskann_results.append(diskann_result)
            print(f"  DiskANN Result: Recall={diskann_result['recall_at1']:.3f}, Latency={diskann_result['latency_ms']:.2f}ms")
        else:
            print(f"  DiskANN: Failed")
    
    return pd.DataFrame(hnsw_results), pd.DataFrame(diskann_results)


def plot_latency_vs_recall(df_hnsw: pd.DataFrame, df_diskann: pd.DataFrame, out_png: str):
    """Plot 2 lines with 4 points each - one for HNSW, one for DiskANN."""
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Latency vs Recall - ONE LINE PER ALGORITHM
    if not df_diskann.empty:
        # Sort by one_minus_recall to create smooth curve
        df_diskann_sorted = df_diskann.sort_values('one_minus_recall')
        ax1.plot(df_diskann_sorted["one_minus_recall"], df_diskann_sorted["latency_ms"],
                marker="o", linestyle="-", label="DiskANN", markersize=10, linewidth=2, color="#1f77b4")

    if not df_hnsw.empty:
        # Sort by one_minus_recall to create smooth curve
        df_hnsw_sorted = df_hnsw.sort_values('one_minus_recall')
        ax1.plot(df_hnsw_sorted["one_minus_recall"], df_hnsw_sorted["latency_ms"],
                marker="s", linestyle="-", label="HNSW", markersize=10, linewidth=2, color="#d62728")

    ax1.set_xlabel("1 – Recall@1")
    ax1.set_ylabel("Average latency per query (ms)")
    ax1.set_title("Latency vs Recall@1 — DiskANN vs HNSW (SIFT1M)")
    ax1.legend()
    # Use log-scale so both sub-ms (HNSW) and multi-ms (DiskANN) latencies are visible
    try:
        all_lat = []
        if not df_diskann.empty:
            all_lat += df_diskann_sorted["latency_ms"].tolist()
        if not df_hnsw.empty:
            all_lat += df_hnsw_sorted["latency_ms"].tolist()
        if all_lat:
            ax1.set_yscale("log")
            ymin = max(min(all_lat) / 2.0, 1e-3)
            ax1.set_ylim(bottom=ymin)
    except Exception:
        pass
    ax1.grid(True, which="both", alpha=0.3)

    # Right plot: Parameter table
    ax2.axis('off')
    ax2.set_title("Parameter Values (4 Trials)")
    
    # Create table data
    table_data = []
    
    if not df_diskann.empty:
        for idx, row in df_diskann.iterrows():
            table_data.append([
                "DiskANN", f"R={row['R']}", f"L={row['L']}", f"build_L={row['build_L']}",
                f"{row['recall_at1']:.3f}", f"{row['latency_ms']:.2f}ms"
            ])
    
    if not df_hnsw.empty:
        for idx, row in df_hnsw.iterrows():
            table_data.append([
                "HNSW", f"M={row['M']}", f"ef={row['efSearch']}", f"efConst={row['efConstruction']}",
                f"{row['recall_at1']:.3f}", f"{row['latency_ms']:.2f}ms"
            ])
    
    if table_data:
        table = ax2.table(cellText=table_data,
                         colLabels=['Algorithm', 'Param1', 'Param2', 'Param3', 'Recall@1', 'Latency'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Color code by algorithm
        for i in range(len(table_data)):
            if table_data[i][0] == "DiskANN":
                table[(i+1, 0)].set_facecolor('#e6f3ff')
            else:
                table[(i+1, 0)].set_facecolor('#ffe6e6')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    data_path = "/home/shivamp6/data/sift1m/sift-128-euclidean.hdf5"
    out_dir = "/home/shivamp6/projects/511-Project-2/part3/out"
    os.makedirs(out_dir, exist_ok=True)

    xb, xq, gt = load_sift_hdf5(data_path)

    # Run full SIFT1M dataset experiment
    print("="*60)
    print("Running FULL SIFT1M dataset experiment...")
    print(f"Training vectors: {xb.shape[0]:,}, Query vectors: {xq.shape[0]:,}")
    print("="*60)

    # Run exactly 4 trials
    df_hnsw, df_diskann = run_four_trials(xb, xq, gt, out_dir)
    
    # Save results
    df_hnsw.to_csv(os.path.join(out_dir, "metrics_hnsw.csv"), index=False)
    if not df_diskann.empty:
        df_diskann.to_csv(os.path.join(out_dir, "metrics_diskann.csv"), index=False)

    # Plot: 2 lines, 4 points each
    plot_latency_vs_recall(df_hnsw, df_diskann, os.path.join(out_dir, "latency_vs_recall.png"))
    
    print("\n" + "="*60)
    print("Results written:")
    print(f"  {os.path.join(out_dir, 'metrics_hnsw.csv')}")
    if not df_diskann.empty:
        print(f"  {os.path.join(out_dir, 'metrics_diskann.csv')}")
    print(f"  {os.path.join(out_dir, 'latency_vs_recall.png')}")
    print("="*60)


if __name__ == "__main__":
    main()
