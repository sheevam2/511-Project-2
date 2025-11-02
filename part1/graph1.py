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

def create_comprehensive_plot(hnsw_results, lsh_results):
    """Create a comprehensive plot with graph on left and table on right"""
    fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(16, 8), 
                                      gridspec_kw={'width_ratios': [2, 1]})
    
    # HNSW plotting - create smooth curves
    hnsw_data = sorted(hnsw_results, key=lambda x: x['recall'])
    hnsw_recalls = [r['recall'] for r in hnsw_data]
    hnsw_qps = [r['qps'] for r in hnsw_data]
    hnsw_efs = [r['efSearch'] for r in hnsw_data]
    
    # LSH plotting - create smooth curves  
    lsh_data = sorted(lsh_results, key=lambda x: x['recall'])
    lsh_recalls = [r['recall'] for r in lsh_data]
    lsh_qps = [r['qps'] for r in lsh_data]
    lsh_nbits = [r['nbits'] for r in lsh_data]
    
    # Plot HNSW as connected line with markers
    ax.plot(hnsw_recalls, hnsw_qps, 'o-', color='#2E86AB', linewidth=3, 
            markersize=8, label='HNSW', markerfacecolor='white', 
            markeredgewidth=2, markeredgecolor='#2E86AB')
    
    # Plot LSH as connected line with markers
    ax.plot(lsh_recalls, lsh_qps, 's-', color='#A23B72', linewidth=3,
            markersize=8, label='LSH', markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='#A23B72')
    
    # Styling for main plot
    ax.set_xlabel('1-Recall@1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Queries Per Second (QPS)', fontsize=14, fontweight='bold')
    ax.set_title('Part 1: HNSW vs LSH Performance Comparison\nQPS vs 1-Recall@1 on SIFT1M Dataset', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(1000, 200000)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Create table data
    table_data = []
    colors = []
    
    # Add HNSW data
    for i, (rec, q, ef) in enumerate(zip(hnsw_recalls, hnsw_qps, hnsw_efs)):
        table_data.append(['HNSW', f'efS={ef}', f'{rec:.3f}', f'{q:.0f}'])
        colors.append('#E8F4F8')
    
    # Add LSH data
    for i, (rec, q, nb) in enumerate(zip(lsh_recalls, lsh_qps, lsh_nbits)):
        table_data.append(['LSH', f'nb={nb}', f'{rec:.3f}', f'{q:.0f}'])
        colors.append('#F8E8F0')
    
    # Create table
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table = ax_table.table(cellText=table_data,
                          colLabels=['Algorithm', 'Parameter', '1-Recall@1', 'QPS'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the cells
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:  # Data rows
                cell.set_facecolor(colors[i-1])
                cell.set_text_props(weight='normal', color='black')
    
    ax_table.set_title('Parameter Values', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def main():
    """Main function"""
    print("Loading Part 1 results...")
    results = load_results()
    
    hnsw_results = results['hnsw_results']
    lsh_results = results['lsh_results']
    
    print(f"Loaded {len(hnsw_results)} HNSW results and {len(lsh_results)} LSH results")
    
    # Create comprehensive plot
    fig = create_comprehensive_plot(hnsw_results, lsh_results)
    
    # Save plot
    output_file = 'part1_comprehensive_qps_vs_recall.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive plot saved as: {output_file}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
