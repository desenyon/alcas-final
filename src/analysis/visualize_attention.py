"""
Visualize attention mechanisms in the affinity model
Shows which protein-ligand interactions the model focuses on
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.append('src/models')
from affinity_model import AffinityModel
from train import manual_batch
import matplotlib.pyplot as plt
import seaborn as sns

# Config
GRAPHS_DIR = Path("data/graphs/protein_cluster")
MODELS_DIR = Path("results/models/affinity")
OUTPUT_DIR = Path("results/analysis/attention")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ATTENTION MECHANISM VISUALIZATION")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test graphs
test_graphs = torch.load(GRAPHS_DIR / 'test.pt', weights_only=False)
print(f"Loaded {len(test_graphs)} test graphs")

# Load model (seed 42)
checkpoint = torch.load(MODELS_DIR / 'seed_42' / 'best_model.pt', weights_only=False)
config = checkpoint['config']

model = AffinityModel(
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Loaded model (seed 42)")

# Hook to capture attention weights
attention_weights = {}

def attention_hook(module, input, output):
    """Capture attention weights"""
    # Output is (attn_output, attn_weights)
    if len(output) == 2:
        attention_weights['weights'] = output[1].detach().cpu()

# Register hooks
model.cross_attn_lig.register_forward_hook(attention_hook)

# Select interesting examples (high and low affinity)
test_sorted = sorted(test_graphs, key=lambda x: x.y.item(), reverse=True)
high_affinity = test_sorted[:5]  # Top 5
low_affinity = test_sorted[-5:]  # Bottom 5

print("\nAnalyzing attention patterns...")

def analyze_batch(graphs, label):
    """Analyze attention for a batch"""
    batch = manual_batch(graphs, device)
    
    with torch.no_grad():
        pred = model(batch)
    
    # Extract attention weights (may be None if not captured)
    attn = attention_weights.get('weights', None)
    
    results = {
        'predictions': pred.squeeze().cpu().numpy(),
        'targets': batch.y.squeeze().cpu().numpy(),
        'attention_captured': attn is not None
    }
    
    if attn is not None:
        # Attention shape varies - just store raw
        results['attention_raw'] = attn.cpu().numpy()
        # Get statistics
        results['attention_mean'] = float(attn.mean())
        results['attention_std'] = float(attn.std())
        results['attention_max'] = float(attn.max())
    
    return results

# Analyze high affinity
print("  High affinity complexes...")
high_results = analyze_batch(high_affinity, "high")

print("  Low affinity complexes...")
low_results = analyze_batch(low_affinity, "low")

# Visualize attention patterns
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: Prediction accuracy
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(high_results['targets'], high_results['predictions'], 
               alpha=0.6, s=100, c='green', edgecolor='black')
axes[0].plot([4, 13], [4, 13], 'r--', label='Perfect prediction')
axes[0].set_xlabel('True pKd', fontsize=12)
axes[0].set_ylabel('Predicted pKd', fontsize=12)
axes[0].set_title('High Affinity Complexes', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(low_results['targets'], low_results['predictions'], 
               alpha=0.6, s=100, c='orange', edgecolor='black')
axes[1].plot([4, 13], [4, 13], 'r--', label='Perfect prediction')
axes[1].set_xlabel('True pKd', fontsize=12)
axes[1].set_ylabel('Predicted pKd', fontsize=12)
axes[1].set_title('Low Affinity Complexes', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'prediction_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: prediction_comparison.png")

# Plot 2: Attention statistics comparison
if high_results['attention_captured']:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Mean', 'Std', 'Max']
    high_vals = [high_results['attention_mean'], 
                 high_results['attention_std'],
                 high_results['attention_max']]
    low_vals = [low_results['attention_mean'], 
                low_results['attention_std'],
                low_results['attention_max']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, high_vals, width, label='High Affinity', color='green', alpha=0.7)
    ax.bar(x + width/2, low_vals, width, label='Low Affinity', color='orange', alpha=0.7)
    
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Cross-Attention Statistics: High vs Low Affinity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'attention_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: attention_statistics.png")

# Plot 3: Error analysis
errors_high = np.abs(high_results['predictions'] - high_results['targets'])
errors_low = np.abs(low_results['predictions'] - low_results['targets'])

fig, ax = plt.subplots(figsize=(8, 6))

bp = ax.boxplot([errors_high, errors_low], 
                 labels=['High Affinity', 'Low Affinity'],
                 widths=0.6, patch_artist=True)

bp['boxes'][0].set_facecolor('green')
bp['boxes'][1].set_facecolor('orange')

ax.set_ylabel('Absolute Error (pKd)', fontsize=12)
ax.set_title('Model Error: High vs Low Affinity Complexes', fontsize=14)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'error_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: error_analysis.png")

print("\n" + "="*70)
print("ATTENTION ANALYSIS COMPLETE")
print("="*70)

if high_results['attention_captured']:
    print(f"\nAttention Statistics:")
    print(f"  High affinity - Mean: {high_results['attention_mean']:.4f}, Std: {high_results['attention_std']:.4f}")
    print(f"  Low affinity - Mean: {low_results['attention_mean']:.4f}, Std: {low_results['attention_std']:.4f}")

print(f"\nPrediction Error:")
print(f"  High affinity MAE: {errors_high.mean():.3f} ± {errors_high.std():.3f}")
print(f"  Low affinity MAE: {errors_low.mean():.3f} ± {errors_low.std():.3f}")

print(f"\n✓ Visualizations saved to: {OUTPUT_DIR}")