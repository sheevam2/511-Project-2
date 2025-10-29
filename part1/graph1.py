#!/usr/bin/env python3
"""
Part 1: Enhanced QPS vs Recall Plot
Reads from part1_results.json and creates a better-looking plot
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load results from JSON file"""
    with open('part1_results.json', 'r') as f:
        return json.load(f)

def create_enhanced_plot(hnsw_results, lsh_results):
    """Create an enhanced QPS vs Recall plot"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # HNSW plotting
    hnsw_recalls = [r['recall'] for r in hnsw_results]
    hnsw_qps = [r['qps'] for r in hnsw_results]
    hnsw_efs = [r['efSearch'] for r in hnsw_results]
    
    # LSH plotting
    lsh_recalls = [r['recall'] for r in lsh_results]
    lsh_qps = [r['qps'] for r in lsh_results]
    lsh_nbits = [r['nbits'] for r in lsh_results]
    
    # Plot HNSW with different colors for different efSearch values
    ef_values = sorted(set(hnsw_efs))
    colors_hnsw = plt.cm.viridis(np.linspace(0, 1, len(ef_values)))
    
    for i, ef in enumerate(ef_values):
        mask = [efs == ef for efs in hnsw_efs]
        recalls = [hnsw_recalls[j] for j in range(len(hnsw_recalls)) if mask[j]]
        qps = [hnsw_qps[j] for j in range(len(hnsw_qps)) if mask[j]]
        
        ax.scatter(recalls, qps, c=[colors_hnsw[i]], s=100, alpha=0.8, 
                  label=f'HNSW (efSearch={ef})', marker='o', edgecolors='black', linewidth=0.5)
        
        # Add annotations
        for j, (rec, q) in enumerate(zip(recalls, qps)):
            ax.annotate(f'efS={ef}', (rec, q), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Plot LSH with different colors for different nbits values
    nbits_values = sorted(set(lsh_nbits))
    colors_lsh = plt.cm.plasma(np.linspace(0, 1, len(nbits_values)))
    
    for i, nbits in enumerate(nbits_values):
        mask = [n == nbits for n in lsh_nbits]
        recalls = [lsh_recalls[j] for j in range(len(lsh_recalls)) if mask[j]]
        qps = [lsh_qps[j] for j in range(len(lsh_qps)) if mask[j]]
        
        ax.scatter(recalls, qps, c=[colors_lsh[i]], s=100, alpha=0.8,
                  label=f'LSH (nbits={nbits})', marker='s', edgecolors='black', linewidth=0.5)
        
        # Add annotations
        for j, (rec, q) in enumerate(zip(recalls, qps)):
            ax.annotate(f'nb={nbits}', (rec, q), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Styling
    ax.set_xlabel('1-Recall@1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Queries Per Second (QPS)', fontsize=12, fontweight='bold')
    ax.set_title('Part 1: HNSW vs LSH Performance Comparison\nQPS vs 1-Recall@1 on SIFT1M Dataset', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Grid and formatting
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(max(hnsw_recalls), max(lsh_recalls)) * 1.05)
    ax.set_ylim(0, max(max(hnsw_qps), max(lsh_qps)) * 1.05)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Log scale for better visualization
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def main():
    """Main function"""
    print("Loading Part 1 results...")
    results = load_results()
    
    hnsw_results = results['hnsw_results']
    lsh_results = results['lsh_results']
    
    print(f"Loaded {len(hnsw_results)} HNSW results and {len(lsh_results)} LSH results")
    
    # Create enhanced plot
    fig = create_enhanced_plot(hnsw_results, lsh_results)
    
    # Save plot
    output_file = 'part1_enhanced_qps_vs_recall.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Enhanced plot saved as: {output_file}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
