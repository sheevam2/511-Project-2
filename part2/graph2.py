#!/usr/bin/env python3
"""
Part 2: Enhanced HNSW Scalability Plots
Reads from part2_results.json and creates better-looking plots
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_results():
    """Load results from JSON file"""
    with open('part2_results.json', 'r') as f:
        return json.load(f)

def create_qps_vs_recall_plot(results):
    """Create QPS vs Recall plot with different curves for each dataset"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color palette for datasets
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for i, dataset_result in enumerate(results):
        dataset = dataset_result['dataset']
        sweep_data = dataset_result['sweep']
        
        # Group by M value for this dataset
        by_m = defaultdict(list)
        for d in sweep_data:
            by_m[d['M']].append(d)
        
        # Plot each M value as a separate line
        for M in sorted(by_m.keys()):
            m_data = by_m[M]
            recalls = [d['recall'] for d in m_data]
            qps = [d['qps'] for d in m_data]
            
            # Sort by recall for smooth line
            sorted_data = sorted(zip(recalls, qps))
            recalls, qps = zip(*sorted_data)
            
            ax.plot(recalls, qps, 'o-', color=colors[i], alpha=0.8, 
                   linewidth=2, markersize=6, label=f'{dataset} (M={M})')
            
            # Add M annotations
            for j, (rec, q) in enumerate(zip(recalls, qps)):
                if j % 2 == 0:  # Annotate every other point to avoid clutter
                    ax.annotate(f'M={M}', (rec, q), xytext=(5, 5),
                               textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Styling
    ax.set_xlabel('1-Recall@1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Queries Per Second (QPS)', fontsize=12, fontweight='bold')
    ax.set_title('Part 2: HNSW Scalability - QPS vs Recall\nAcross Different Dataset Sizes and M Values', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def create_buildtime_vs_recall_plot(results):
    """Create Index Build Time vs Recall plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color palette for datasets
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for i, dataset_result in enumerate(results):
        dataset = dataset_result['dataset']
        sweep_data = dataset_result['sweep']
        
        # Group by M value for this dataset
        by_m = defaultdict(list)
        for d in sweep_data:
            by_m[d['M']].append(d)
        
        # Plot each M value as a separate line
        for M in sorted(by_m.keys()):
            m_data = by_m[M]
            recalls = [d['recall'] for d in m_data]
            build_times = [d['build_time'] for d in m_data]
            
            # Sort by recall for smooth line
            sorted_data = sorted(zip(recalls, build_times))
            recalls, build_times = zip(*sorted_data)
            
            ax.plot(recalls, build_times, 's-', color=colors[i], alpha=0.8,
                   linewidth=2, markersize=6, label=f'{dataset} (M={M})')
            
            # Add M annotations
            for j, (rec, bt) in enumerate(zip(recalls, build_times)):
                if j % 2 == 0:  # Annotate every other point
                    ax.annotate(f'M={M}', (rec, bt), xytext=(5, 5),
                               textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Styling
    ax.set_xlabel('1-Recall@1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Index Build Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Part 2: HNSW Scalability - Build Time vs Recall\nAcross Different Dataset Sizes and M Values', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def create_dataset_size_analysis(results):
    """Create analysis plots showing dataset size effects"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group by dataset and M
    by_dataset = defaultdict(lambda: defaultdict(list))
    for dataset_result in results:
        dataset = dataset_result['dataset']
        for sweep_item in dataset_result['sweep']:
            M = sweep_item['M']
            by_dataset[dataset][M].append(sweep_item)
    
    # Extract dataset sizes (approximate from names)
    dataset_sizes = {
        'MNIST': 60000,
        'NYTimes': 290000, 
        'SIFT': 1000000,
        'GloVe-100': 1183514
    }
    
    # Plot 1: QPS vs Dataset Size (for M=16)
    M_target = 16
    if M_target in by_dataset.get('SIFT', {}):
        sizes = []
        qps_values = []
        recalls = []
        
        for dataset, m_data in by_dataset.items():
            if M_target in m_data:
                # Use median QPS for this M value
                m_results = m_data[M_target]
                qps_vals = [r['qps'] for r in m_results]
                recall_vals = [r['recall'] for r in m_results]
                
                sizes.append(dataset_sizes.get(dataset, 0))
                qps_values.append(np.median(qps_vals))
                recalls.append(np.median(recall_vals))
        
        if sizes:
            scatter = ax1.scatter(sizes, qps_values, c=recalls, s=100, 
                                cmap='viridis', alpha=0.8, edgecolors='black')
            ax1.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
            ax1.set_ylabel('QPS (median)', fontsize=12, fontweight='bold')
            ax1.set_title(f'QPS vs Dataset Size (M={M_target})', fontsize=12, fontweight='bold')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Recall', fontsize=10)
    
    # Plot 2: Build Time vs Dataset Size (for M=16)
    if M_target in by_dataset.get('SIFT', {}):
        sizes = []
        build_times = []
        recalls = []
        
        for dataset, m_data in by_dataset.items():
            if M_target in m_data:
                m_results = m_data[M_target]
                bt_vals = [r['build_time'] for r in m_results]
                recall_vals = [r['recall'] for r in m_results]
                
                sizes.append(dataset_sizes.get(dataset, 0))
                build_times.append(np.median(bt_vals))
                recalls.append(np.median(recall_vals))
        
        if sizes:
            scatter = ax2.scatter(sizes, build_times, c=recalls, s=100,
                                cmap='plasma', alpha=0.8, edgecolors='black')
            ax2.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Build Time (seconds, median)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Build Time vs Dataset Size (M={M_target})', fontsize=12, fontweight='bold')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Recall', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    """Main function"""
    print("Loading Part 2 results...")
    results = load_results()
    
    print(f"Loaded {len(results)} results")
    
    # Create plots
    print("Creating QPS vs Recall plot...")
    fig1 = create_qps_vs_recall_plot(results)
    fig1.savefig('part2_enhanced_qps_vs_recall.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: part2_enhanced_qps_vs_recall.png")
    
    print("Creating Build Time vs Recall plot...")
    fig2 = create_buildtime_vs_recall_plot(results)
    fig2.savefig('part2_enhanced_buildtime_vs_recall.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: part2_enhanced_buildtime_vs_recall.png")
    
    print("Creating dataset size analysis...")
    fig3 = create_dataset_size_analysis(results)
    fig3.savefig('part2_dataset_size_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: part2_dataset_size_analysis.png")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
