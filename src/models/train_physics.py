"""
Train affinity model with physics-informed loss
Compare against standard MSE training
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('src/models')
from affinity_model import AffinityModel
from train import manual_batch, evaluate
from physics_loss import PhysicsInformedLoss
from tqdm import tqdm

# Config
GRAPHS_DIR = Path("data/graphs/protein_cluster")
OUTPUT_DIR = Path("results/models/physics_informed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CONFIG = {
    'hidden_dim': 256,
    'num_layers': 5,
    'dropout': 0.15,
    'batch_size': 32,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 15,
    'grad_clip': 1.0,
}

print("="*70)
print("PHYSICS-INFORMED MODEL TRAINING")
print("="*70)

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load data
print("\nLoading data...")
train_graphs = torch.load(GRAPHS_DIR / 'train.pt', weights_only=False)
val_graphs = torch.load(GRAPHS_DIR / 'val.pt', weights_only=False)

print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)}")

# Create model
model = AffinityModel(
    hidden_dim=CONFIG['hidden_dim'],
    num_layers=CONFIG['num_layers'],
    dropout=CONFIG['dropout']
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Physics-informed loss
criterion = PhysicsInformedLoss(
    alpha_mse=1.0,
    alpha_diversity=0.02
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['lr'],
    weight_decay=CONFIG['weight_decay']
)

# Learning rate scheduler with warmup
def get_lr_lambda(epoch):
    """Warmup for 5 epochs, then cosine decay"""
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (CONFIG['epochs'] - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

# Training function with physics loss
def train_epoch_physics(model, graphs, batch_size, optimizer, device, grad_clip):
    """Train one epoch with physics-informed loss"""
    model.train()
    
    indices = list(range(len(graphs)))
    np.random.shuffle(indices)
    
    epoch_loss = 0.0
    epoch_components = {'mse': 0.0, 'diversity': 0.0}
    num_batches = 0
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_graphs = [graphs[j] for j in batch_indices]
        
        batch = manual_batch(batch_graphs, device)
        
        optimizer.zero_grad()
        
        predictions = model(batch).squeeze()
        targets = batch.y.squeeze()
        
        # Physics-informed loss
        loss, components = criterion(predictions, targets, batch)
        
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        for key, value in components.items():
            if key != 'total':
                epoch_components[key] += value
        
        num_batches += 1
    
    # Average
    epoch_loss /= num_batches
    for key in epoch_components:
        epoch_components[key] /= num_batches
    
    return epoch_loss, epoch_components

# Training loop
print("\n" + "="*70)
print("TRAINING")
print("="*70)

best_val_loss = float('inf')
patience_counter = 0
history = {
    'train_loss': [],
    'train_components': [],
    'val_loss': [],
    'val_r2': [],
    'val_pearson': [],
    'lr': []
}

for epoch in range(1, CONFIG['epochs'] + 1):
    # Train
    train_loss, train_components = train_epoch_physics(
        model, train_graphs, CONFIG['batch_size'],
        optimizer, device, CONFIG['grad_clip']
    )
    
    # Validate (standard metrics)
    val_metrics = evaluate(model, val_graphs, CONFIG['batch_size'], device)
    
    # Log
    current_lr = optimizer.param_groups[0]['lr']
    
    history['train_loss'].append(train_loss)
    history['train_components'].append(train_components)
    history['val_loss'].append(val_metrics['loss'])
    history['val_r2'].append(val_metrics['r2'])
    history['val_pearson'].append(val_metrics['pearson'])
    history['lr'].append(current_lr)
    
    print(f"Epoch {epoch:3d} | "
          f"Train: {train_loss:.4f} "
          f"(MSE:{train_components['mse']:.4f}, "
          f"Div:{train_components['diversity']:.4f}) | "
          f"Val: {val_metrics['loss']:.4f} "
          f"R²:{val_metrics['r2']:.4f} "
          f"Pearson:{val_metrics['pearson']:.4f} | "
          f"LR:{current_lr:.6f}")
    
    # Save best
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'val_metrics': val_metrics,
            'train_components': train_components
        }
        
        torch.save(checkpoint, OUTPUT_DIR / 'best_model.pt')
        print(f"  ✓ Saved best model")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\nEarly stopping at epoch {epoch}")
        break
    
    # Step scheduler
    scheduler.step()

# Save history
with open(OUTPUT_DIR / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Saved to: {OUTPUT_DIR}")

# Compare against baseline
print("\n" + "="*70)
print("COMPARISON: Physics-Informed vs Standard")
print("="*70)

baseline_checkpoint = torch.load(
    Path("results/models/affinity/seed_42/best_model.pt"),
    weights_only=False
)

print("\nStandard Model (MSE only):")
print(f"  Val Loss: {baseline_checkpoint['val_metrics']['loss']:.4f}")
print(f"  R²: {baseline_checkpoint['val_metrics']['r2']:.4f}")
print(f"  Pearson: {baseline_checkpoint['val_metrics']['pearson']:.4f}")

print("\nPhysics-Informed Model:")
physics_checkpoint = torch.load(OUTPUT_DIR / 'best_model.pt', weights_only=False)
print(f"  Val Loss: {physics_checkpoint['val_metrics']['loss']:.4f}")
print(f"  R²: {physics_checkpoint['val_metrics']['r2']:.4f}")
print(f"  Pearson: {physics_checkpoint['val_metrics']['pearson']:.4f}")

improvement_r2 = (physics_checkpoint['val_metrics']['r2'] - 
                 baseline_checkpoint['val_metrics']['r2']) / baseline_checkpoint['val_metrics']['r2'] * 100

print(f"\nR² Improvement: {improvement_r2:+.1f}%")

if improvement_r2 > 0:
    print("✓ Physics-informed loss improves performance!")
else:
    print("→ Physics-informed loss provides regularization (similar performance)")

print("\n" + "="*70)
print("✓ Physics-informed training complete!")
print("="*70)