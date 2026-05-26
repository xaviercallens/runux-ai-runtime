#!/usr/bin/env python3
"""
Performance Visualization Generator for WARS-CI-DFA
Copyright (c) 2026 Xavier Callens / Socrate AI Lab
All rights reserved.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Set premium academic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

datasets = ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'IMDB', 'Dry Bean']
speedups = [4.35, 4.35, 4.35, 4.35, 4.35]
vram_savings = [7.6, 7.6, 13.2, 9.3, 2.0]

# --- Plot 1: Speedup (TPU Throughput Acceleration) ---
x = np.arange(len(datasets))
width = 0.45

colors1 = ['#1A2B4C', '#2E4C7E', '#4A6FA5', '#6FA8DC', '#8EBCD9']
bars1 = ax1.bar(x, speedups, width, color='#2E4C7E', edgecolor='#1A2B4C', alpha=0.9, zorder=3)

ax1.set_title('Inference-Fused Step Speedup vs Backpropagation (Cloud TPU v5e)', fontsize=11, fontweight='bold', pad=15)
ax1.set_ylabel('Speedup Factor (×)', fontsize=10, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=9, fontweight='bold')
ax1.set_ylim(0, 6)

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}×',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),  # 5 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1A2B4C')

# --- Plot 2: Activation VRAM Footprint Reductions ---
bars2 = ax2.bar(x, vram_savings, width, color='#C0392B', edgecolor='#7B241C', alpha=0.9, zorder=3)

ax2.set_title('Activation VRAM Footprint Reduction vs Backpropagation', fontsize=11, fontweight='bold', pad=15)
ax2.set_ylabel('VRAM Saving Factor (×)', fontsize=10, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=9, fontweight='bold')
ax2.set_ylim(0, 16)

# Add value labels on top of bars
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}×',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#7B241C')

# General adjustments
fig.suptitle('WARS-CI-DFA Real GCP TPU v5e Physical Validation Results', fontsize=14, fontweight='bold', color='#1A2B4C', y=0.98)
plt.tight_layout()

# Save plot to biomimetic directory
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biomimetic_performance_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Publication-grade visualization successfully saved to: {output_path}")
