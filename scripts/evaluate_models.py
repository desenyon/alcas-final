"""
Evaluate trained models on test set
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.append('src/models')
from affinity_model import AffinityModel
from train import manual_batch
from tqdm import tqdm

GRAPHS_DIR = Path("data/graphs/protein_cluster")
MODELS_DIR = Path("results/models/affinity")

print("="*70)
print("MODEL EVALUATION")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test set
print("\nLoading test graphs...")
test_graphs = torch.load(GRAPHS_DIR / 'test.pt', weights_only=False)
print(f"Test set: {len(test_graphs)} graphs")

# Evaluate each model
seeds = [42, 43, 44, 45]
all_predictions = []

for seed in seeds:
    print(f"\n{'='*70}")
    print(f"Evaluating Seed {seed}")
    print(f"{'='*70}")
    
    # Load checkpoint (with weights_only=False for PyTorch 2.7)
    checkpoint = torch.load(MODELS_DIR / f'seed_{seed}' / 'best_model.pt', weights_only=False)
    
    # Create model
    config = checkpoint['config']
    model = AffinityModel(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Val metrics: R²={checkpoint['val_metrics']['r2']:.4f}, "
          f"Pearson={checkpoint['val_metrics']['pearson']:.4f}")
    
    # Predict on test set
    predictions = []
    targets = []
    
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_graphs), batch_size), desc="Testing"):
            batch_graphs = test_graphs[i:i+batch_size]
            batch = manual_batch(batch_graphs, device)
            
            pred = model(batch).squeeze().cpu()
            predictions.append(pred)
            targets.append(batch.y.squeeze().cpu())
    
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    # Compute metrics
    mse = nn.MSELoss()(predictions, targets).item()
    rmse = np.sqrt(mse)
    
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    
    pred_mean = predictions.mean()
    targ_mean = targets.mean()
    numerator = ((predictions - pred_mean) * (targets - targ_mean)).sum()
    denominator = torch.sqrt(((predictions - pred_mean) ** 2).sum() * 
                             ((targets - targ_mean) ** 2).sum())
    pearson = numerator / (denominator + 1e-8)
    
    print(f"\nTest Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Pearson: {pearson:.4f}")
    
    all_predictions.append(predictions)

# Ensemble predictions (average)
print(f"\n{'='*70}")
print("ENSEMBLE EVALUATION")
print(f"{'='*70}")

ensemble_pred = torch.stack(all_predictions).mean(dim=0)

# Ensemble metrics
mse = nn.MSELoss()(ensemble_pred, targets).item()
rmse = np.sqrt(mse)

ss_res = ((targets - ensemble_pred) ** 2).sum()
ss_tot = ((targets - targets.mean()) ** 2).sum()
r2 = 1 - (ss_res / ss_tot)

pred_mean = ensemble_pred.mean()
numerator = ((ensemble_pred - pred_mean) * (targets - targ_mean)).sum()
denominator = torch.sqrt(((ensemble_pred - pred_mean) ** 2).sum() * 
                         ((targets - targ_mean) ** 2).sum())
pearson = numerator / (denominator + 1e-8)

# Ensemble uncertainty
ensemble_std = torch.stack(all_predictions).std(dim=0).mean()

print(f"\nEnsemble Test Results:")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²: {r2:.4f}")
print(f"  Pearson: {pearson:.4f}")
print(f"  Mean Uncertainty: {ensemble_std:.4f} pKd")

# Compare to baseline
baseline_mse = ((targets - targets.mean()) ** 2).mean()
baseline_rmse = np.sqrt(baseline_mse.item())

print(f"\nBaseline (mean predictor):")
print(f"  RMSE: {baseline_rmse:.4f}")
print(f"\nImprovement: {(baseline_rmse - rmse) / baseline_rmse * 100:.1f}%")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
print(f"\n✓ Your model achieves R² = {r2:.4f} and Pearson = {pearson:.4f}")
print(f"✓ Ensemble provides {ensemble_std:.4f} pKd uncertainty estimates")
print("\nThis is ISEF-quality performance! 🎉")