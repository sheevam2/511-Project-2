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

def create_comprehensive_qps_plot(results):
    """Create comprehensive QPS vs Recall plot with table"""
    fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(18, 8), 
                                      gridspec_kw={'width_ratios': [2, 1]})
    
    # Color palette for datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    table_data = []
    table_colors = []
    
    for i, dataset_result in enumerate(results):
        dataset = dataset_result['dataset']
        sweep_data = dataset_result['sweep']
        
        # Sort by recall for smooth curve
        sweep_data_sorted = sorted(sweep_data, key=lambda x: x['recall'])
        recalls = [d['recall'] for d in sweep_data_sorted]
        qps = [d['qps'] for d in sweep_data_sorted]
        M_values = [d['M'] for d in sweep_data_sorted]
        
        # Plot dataset curve
        ax.plot(recalls, qps, marker=markers[i], color=colors[i], alpha=0.8, 
               linewidth=3, markersize=8, label=f'{dataset}', 
               markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i])
        
        # Add to table data
        for j, (rec, q, M) in enumerate(zip(recalls, qps, M_values)):
            table_data.append([dataset, f'M={M}', f'{rec:.3f}', f'{q:.0f}'])
            table_colors.append(colors[i])
    
    # Styling for main plot
    ax.set_xlabel('Recall@1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Queries Per Second (QPS)', fontsize=14, fontweight='bold')
    ax.set_title('Part 2: HNSW Scalability - QPS vs Recall\nAcross Different Dataset Sizes', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_yscale('log')
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Create table
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table = ax_table.table(cellText=table_data,
                          colLabels=['Dataset', 'M Value', 'Recall@1', 'QPS'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color the cells
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:  # Data rows
                cell.set_facecolor(table_colors[i-1])
                cell.set_text_props(weight='normal', color='black')
                cell.set_alpha(0.3)
    
    ax_table.set_title('Parameter Values', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_comprehensive_buildtime_plot(results):
    """Create comprehensive Build Time vs Recall plot with table"""
    fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(18, 8), 
                                      gridspec_kw={'width_ratios': [2, 1]})
    
    # Color palette for datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    table_data = []
    table_colors = []
    
    for i, dataset_result in enumerate(results):
        dataset = dataset_result['dataset']
        sweep_data = dataset_result['sweep']
        
        # Sort by recall for smooth curve
        sweep_data_sorted = sorted(sweep_data, key=lambda x: x['recall'])
        recalls = [d['recall'] for d in sweep_data_sorted]
        build_times = [d['build_time'] for d in sweep_data_sorted]
        M_values = [d['M'] for d in sweep_data_sorted]
        
        # Plot dataset curve
        ax.plot(recalls, build_times, marker=markers[i], color=colors[i], alpha=0.8,
               linewidth=3, markersize=8, label=f'{dataset}',
               markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i])
        
        # Add to table data
        for j, (rec, bt, M) in enumerate(zip(recalls, build_times, M_values)):
            table_data.append([dataset, f'M={M}', f'{rec:.3f}', f'{bt:.2f}'])
            table_colors.append(colors[i])
    
    # Styling for main plot
    ax.set_xlabel('Recall@1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Index Build Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Part 2: HNSW Scalability - Build Time vs Recall\nAcross Different Dataset Sizes', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_yscale('log')
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Create table
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table = ax_table.table(cellText=table_data,
                          colLabels=['Dataset', 'M Value', 'Recall@1', 'Build Time (s)'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color the cells
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:  # Data rows
                cell.set_facecolor(table_colors[i-1])
                cell.set_text_props(weight='normal', color='black')
                cell.set_alpha(0.3)
    
    ax_table.set_title('Parameter Values', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_individual_dataset_plots(results):
    """Create individual plots for each dataset with separate table"""
    for dataset_result in results:
        dataset = dataset_result['dataset']
        sweep_data = dataset_result['sweep']
        
        # Sort by M value
        sweep_sorted = sorted(sweep_data, key=lambda x: x['M'])
        
        # Create figure with 3 subplots: graph1, graph2, table
        fig, (ax1, ax2, ax_table) = plt.subplots(1, 3, figsize=(20, 6), 
                                                gridspec_kw={'width_ratios': [1, 1, 0.8]})
        
        # Extract data
        M_values = [d['M'] for d in sweep_sorted]
        recalls = [d['recall'] for d in sweep_sorted]
        qps = [d['qps'] for d in sweep_sorted]
        build_times = [d['build_time'] for d in sweep_sorted]
        
        # Plot 1: QPS vs Recall
        ax1.plot(recalls, qps, 'o-', color='#2E86AB', linewidth=3, markersize=8,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2E86AB')
        ax1.set_xlabel('Recall@1', fontsize=12, fontweight='bold')
        ax1.set_ylabel('QPS', fontsize=12, fontweight='bold')
        ax1.set_title(f'{dataset} - QPS vs Recall', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Build Time vs Recall
        ax2.plot(recalls, build_times, 's-', color='#A23B72', linewidth=3, markersize=8,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#A23B72')
        ax2.set_xlabel('Recall@1', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Build Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title(f'{dataset} - Build Time vs Recall', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Create table data
        table_data = []
        for d in sweep_sorted:
            table_data.append([f"M={d['M']}", f"{d['recall']:.3f}", f"{d['qps']:.0f}", f"{d['build_time']:.2f}s"])
        
        # Create table
        ax_table.axis('tight')
        ax_table.axis('off')
        
        table = ax_table.table(cellText=table_data,
                              colLabels=['M Value', 'Recall@1', 'QPS', 'Build Time'],
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
                    cell.set_facecolor('#E8F4F8')
                    cell.set_text_props(weight='normal', color='black')
        
        ax_table.set_title(f'{dataset} Results', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save individual plot
        filename = f'part2_{dataset.lower().replace("-", "_")}_detailed.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        
        plt.close(fig)

def print_results_table(results):
    """Print a formatted table of results"""
    print("\n" + "="*80)
    print("PART 2 RESULTS TABLE")
    print("="*80)
    
    for dataset_result in results:
        dataset = dataset_result['dataset']
        sweep_data = dataset_result['sweep']
        
        print(f"\n{dataset} Dataset:")
        print("-" * 60)
        print(f"{'M':<4} {'Recall@1':<10} {'QPS':<12} {'Build Time (s)':<15}")
        print("-" * 60)
        
        # Sort by M value for clean table
        sweep_sorted = sorted(sweep_data, key=lambda x: x['M'])
        for d in sweep_sorted:
            print(f"{d['M']:<4} {d['recall']:<10.3f} {d['qps']:<12.0f} {d['build_time']:<15.2f}")
    
    print("\n" + "="*80)

def main():
    """Main function"""
    print("Loading Part 2 results...")
    results = load_results()
    
    print(f"Loaded {len(results)} results")
    
    # Print results table
    print_results_table(results)
    
    # Create comprehensive plots
    print("\nCreating comprehensive QPS vs Recall plot...")
    fig1 = create_comprehensive_qps_plot(results)
    fig1.savefig('part2_comprehensive_qps_vs_recall.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: part2_comprehensive_qps_vs_recall.png")
    
    print("Creating comprehensive Build Time vs Recall plot...")
    fig2 = create_comprehensive_buildtime_plot(results)
    fig2.savefig('part2_comprehensive_buildtime_vs_recall.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: part2_comprehensive_buildtime_vs_recall.png")
    
    # Create individual dataset plots with tables
    print("\nCreating individual dataset plots with tables...")
    create_individual_dataset_plots(results)
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
