"""
Calibration analysis for ensemble uncertainty
Shows that predicted uncertainty correlates with actual error
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.append('src/models')
sys.path.append('src/visualization')
from affinity_model import AffinityModel
from train import manual_batch
import matplotlib.pyplot as plt
from scipy import stats
from plot_theme import set_theme, COLORS, format_axis

# Apply theme
set_theme()

# Config
GRAPHS_DIR = Path("data/graphs/protein_cluster")
MODELS_DIR = Path("results/models/affinity")
OUTPUT_DIR = Path("results/analysis/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ENSEMBLE UNCERTAINTY CALIBRATION")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test set
test_graphs = torch.load(GRAPHS_DIR / 'test.pt', weights_only=False)
print(f"Test set: {len(test_graphs)} graphs")

# Load all 4 models
seeds = [42, 43, 44, 45]
models = []

for seed in seeds:
    checkpoint = torch.load(MODELS_DIR / f'seed_{seed}' / 'best_model.pt', weights_only=False)
    config = checkpoint['config']
    
    model = AffinityModel(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)

print(f"✓ Loaded {len(models)} models")

# Get ensemble predictions
print("\nComputing ensemble predictions...")

batch_size = 32
all_predictions = []
all_targets = []

for i in range(0, len(test_graphs), batch_size):
    batch_graphs = test_graphs[i:i+batch_size]
    batch = manual_batch(batch_graphs, device)
    
    batch_preds = []
    with torch.no_grad():
        for model in models:
            pred = model(batch).squeeze().cpu()
            batch_preds.append(pred)
    
    all_predictions.append(torch.stack(batch_preds))
    all_targets.append(batch.y.squeeze().cpu())

# Concatenate
predictions = torch.cat(all_predictions, dim=1)  # [4 models, n_samples]
targets = torch.cat(all_targets)

print(f"✓ Predictions shape: {predictions.shape}")

# Compute ensemble statistics
ensemble_mean = predictions.mean(dim=0)
ensemble_std = predictions.std(dim=0)

# Compute errors
errors = torch.abs(ensemble_mean - targets)

print("\n" + "="*70)
print("CALIBRATION ANALYSIS")
print("="*70)

# Uncertainty vs Error correlation
correlation, p_value = stats.pearsonr(ensemble_std.numpy(), errors.numpy())

print(f"\nUncertainty-Error Correlation:")
print(f"  Pearson r: {correlation:.4f}")
print(f"  p-value: {p_value:.6f}")

if correlation > 0 and p_value < 0.05:
    print(f"  ✓ Significant positive correlation: Well-calibrated!")
else:
    print(f"  ⚠ Model uncertainty needs better calibration")

# Calibration bins
n_bins = 5
uncertainty_sorted = torch.argsort(ensemble_std)
bin_size = len(uncertainty_sorted) // n_bins

print("\nCalibration by Uncertainty Bins:")
for i in range(n_bins):
    start = i * bin_size
    end = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainty_sorted)
    bin_indices = uncertainty_sorted[start:end]
    
    bin_uncertainty = ensemble_std[bin_indices].mean()
    bin_error = errors[bin_indices].mean()
    
    print(f"  Bin {i+1}: Uncertainty={bin_uncertainty:.3f}, Error={bin_error:.3f}")

# Visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: Uncertainty vs Error scatter
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(ensemble_std.numpy(), errors.numpy(), 
                    alpha=0.6, s=50, c=targets.numpy(), 
                    cmap='viridis', edgecolor='white', linewidth=0.8)

# Trend line
z = np.polyfit(ensemble_std.numpy(), errors.numpy(), 1)
p = np.poly1d(z)
x_line = np.linspace(ensemble_std.min(), ensemble_std.max(), 100)
ax.plot(x_line, p(x_line), color=COLORS['danger'], linewidth=3, 
        linestyle='--', label=f'Trend (r={correlation:.3f})')

format_axis(ax, 
           xlabel='Predicted Uncertainty (Ensemble Std Dev, pKd)',
           ylabel='Absolute Prediction Error (pKd)',
           title='Ensemble Calibration: Model Uncertainty vs Actual Error')

ax.legend(fontsize=12, loc='upper left')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('True Binding Affinity (pKd)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'calibration_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved: calibration_scatter.png")

# Plot 2: Calibration curve
bin_uncertainties = []
bin_errors = []
bin_stds = []

for i in range(n_bins):
    start = i * bin_size
    end = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainty_sorted)
    bin_indices = uncertainty_sorted[start:end]
    
    bin_uncertainties.append(ensemble_std[bin_indices].mean().item())
    bin_errors.append(errors[bin_indices].mean().item())
    bin_stds.append(errors[bin_indices].std().item())

fig, ax = plt.subplots(figsize=(10, 7))

ax.errorbar(bin_uncertainties, bin_errors, yerr=bin_stds, 
            fmt='o-', markersize=12, linewidth=3, capsize=6,
            color=COLORS['primary'], ecolor=COLORS['primary'],
            label='Observed Error (±1 SD)', markeredgecolor='white', markeredgewidth=1.5)

max_val = max(max(bin_uncertainties), max(bin_errors))
ax.plot([0, max_val], [0, max_val], 
        color=COLORS['danger'], linewidth=2.5, linestyle='--',
        label='Perfect Calibration', alpha=0.8)

format_axis(ax,
           xlabel='Predicted Uncertainty (pKd)',
           ylabel='Observed Mean Absolute Error (pKd)',
           title='Calibration Curve: Predicted vs Observed Uncertainty')

ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax.set_xlim(0, max_val * 1.05)
ax.set_ylim(0, max_val * 1.05)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'calibration_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: calibration_curve.png")

# Plot 3: Uncertainty distribution
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(ensemble_std.numpy(), bins=40, alpha=0.7, 
        color=COLORS['primary'], edgecolor='white', linewidth=1.2)

ax.axvline(ensemble_std.mean(), color=COLORS['danger'], 
          linestyle='--', linewidth=2.5, 
          label=f'Mean: {ensemble_std.mean():.3f} pKd')
ax.axvline(ensemble_std.median(), color=COLORS['accent'], 
          linestyle='--', linewidth=2.5,
          label=f'Median: {ensemble_std.median():.3f} pKd')

format_axis(ax,
           xlabel='Ensemble Uncertainty (Std Dev, pKd)',
           ylabel='Frequency',
           title='Distribution of Prediction Uncertainty')

ax.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'uncertainty_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: uncertainty_distribution.png")

# Summary statistics
print("\n" + "="*70)
print("CALIBRATION SUMMARY")
print("="*70)

print(f"\nEnsemble Statistics:")
print(f"  Mean uncertainty: {ensemble_std.mean():.3f} ± {ensemble_std.std():.3f} pKd")
print(f"  Mean error: {errors.mean():.3f} ± {errors.std():.3f} pKd")
print(f"  Correlation (uncertainty vs error): {correlation:.4f} (p={p_value:.6f})")

print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")
print("="*70)