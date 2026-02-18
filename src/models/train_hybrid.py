"""
Train hybrid model with REAL ESM-2 embeddings and compare against structure-only baseline
Shows value of adding sequence information
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('src/models')
from hybrid_model import HybridAffinityModel
from sklearn.metrics import r2_score
import scipy.stats

# Config
GRAPHS_DIR = Path("data/esm_embeddings")  # Updated path with real ESM
BASELINE_DIR = Path("results/models/affinity")
OUTPUT_DIR = Path("results/models/hybrid")
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
print("HYBRID MODEL TRAINING (Structure + Real ESM-2 Sequence)")
print("="*70)

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load data with real ESM embeddings
print("\nLoading data with ESM-2 embeddings...")
train_graphs = torch.load(GRAPHS_DIR / 'train_with_esm.pt', weights_only=False)
val_graphs = torch.load(GRAPHS_DIR / 'val_with_esm.pt', weights_only=False)
test_graphs = torch.load(GRAPHS_DIR / 'test_with_esm.pt', weights_only=False)

print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

# Verify ESM embeddings
sample_graph = train_graphs[0]
if hasattr(sample_graph, 'esm_embedding'):
    print(f"✓ ESM embeddings present: {sample_graph.esm_embedding.shape}")
    print(f"  Sample norm: {sample_graph.esm_embedding.norm():.4f}")
    
    # Check diversity
    emb1 = train_graphs[0].esm_embedding
    emb2 = train_graphs[100].esm_embedding
    cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    print(f"  Diversity check (cos sim): {cos_sim.item():.4f}")
    if cos_sim.item() > 0.99:
        print("  ⚠ WARNING: Embeddings might not be diverse!")
else:
    print("✗ ERROR: No ESM embeddings found!")
    exit(1)

# Create model
model = HybridAffinityModel(
    hidden_dim=CONFIG['hidden_dim'],
    num_layers=CONFIG['num_layers'],
    dropout=CONFIG['dropout'],
    esm_dim=1280
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['lr'],
    weight_decay=CONFIG['weight_decay']
)

# Learning rate scheduler
def get_lr_lambda(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (CONFIG['epochs'] - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

# Manual batching function with ESM
def manual_batch_hybrid(graphs, device):
    """Create batch with ESM embeddings"""
    from types import SimpleNamespace
    
    batch = SimpleNamespace()
    
    # Ligand
    ligand_x_list = []
    ligand_edge_index_list = []
    ligand_batch_list = []
    ligand_offset = 0
    
    # Protein
    protein_x_list = []
    protein_edge_index_list = []
    protein_batch_list = []
    protein_offset = 0
    
    # ESM embeddings
    esm_list = []
    
    # Targets
    y_list = []
    
    for i, g in enumerate(graphs):
        # Ligand
        ligand_x_list.append(g.ligand_x)
        ligand_edge_index_list.append(g.ligand_edge_index + ligand_offset)
        ligand_batch_list.append(torch.full((g.ligand_x.size(0),), i, dtype=torch.long))
        ligand_offset += g.ligand_x.size(0)
        
        # Protein
        protein_x_list.append(g.protein_x)
        protein_edge_index_list.append(g.protein_edge_index + protein_offset)
        protein_batch_list.append(torch.full((g.protein_x.size(0),), i, dtype=torch.long))
        protein_offset += g.protein_x.size(0)
        
        # ESM
        esm_list.append(g.esm_embedding)
        
        # Target
        y_list.append(g.y)
    
    # Concatenate
    batch.ligand_x = torch.cat(ligand_x_list, dim=0).to(device)
    batch.ligand_edge_index = torch.cat(ligand_edge_index_list, dim=1).to(device)
    batch.ligand_batch = torch.cat(ligand_batch_list, dim=0).to(device)
    
    batch.protein_x = torch.cat(protein_x_list, dim=0).to(device)
    batch.protein_edge_index = torch.cat(protein_edge_index_list, dim=1).to(device)
    batch.protein_batch = torch.cat(protein_batch_list, dim=0).to(device)
    
    batch.esm_embedding = torch.stack(esm_list, dim=0).to(device)
    
    batch.y = torch.stack(y_list, dim=0).to(device)
    
    return batch

# Training function
def train_epoch(model, graphs, batch_size, optimizer, criterion, device, grad_clip):
    model.train()
    
    indices = list(range(len(graphs)))
    np.random.shuffle(indices)
    
    epoch_loss = 0.0
    num_batches = 0
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_graphs = [graphs[j] for j in batch_indices]
        
        batch = manual_batch_hybrid(batch_graphs, device)
        
        optimizer.zero_grad()
        
        predictions = model(batch).squeeze()
        targets = batch.y.squeeze()
        
        loss = criterion(predictions, targets)
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    return epoch_loss / num_batches

# Evaluation function
def evaluate_hybrid(model, graphs, batch_size, device):
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i:i+batch_size]
            batch = manual_batch_hybrid(batch_graphs, device)
            
            predictions = model(batch).squeeze()
            targets = batch.y.squeeze()
            
            loss = nn.MSELoss()(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.extend(predictions.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    r2 = r2_score(all_targets, all_preds)
    pearson = scipy.stats.pearsonr(all_targets, all_preds)[0]
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return {
        'loss': total_loss / num_batches,
        'r2': r2,
        'pearson': pearson,
        'rmse': rmse
    }

# Training loop
print("\n" + "="*70)
print("TRAINING")
print("="*70)

best_val_loss = float('inf')
patience_counter = 0
history = {
    'train_loss': [],
    'val_loss': [],
    'val_r2': [],
    'val_pearson': [],
    'lr': []
}

for epoch in range(1, CONFIG['epochs'] + 1):
    # Train
    train_loss = train_epoch(
        model, train_graphs, CONFIG['batch_size'],
        optimizer, criterion, device, CONFIG['grad_clip']
    )
    
    # Validate
    val_metrics = evaluate_hybrid(model, val_graphs, CONFIG['batch_size'], device)
    
    # Log
    current_lr = optimizer.param_groups[0]['lr']
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_metrics['loss'])
    history['val_r2'].append(val_metrics['r2'])
    history['val_pearson'].append(val_metrics['pearson'])
    history['lr'].append(current_lr)
    
    print(f"Epoch {epoch:3d} | "
          f"Train: {train_loss:.4f} | "
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
            'val_metrics': val_metrics
        }
        
        torch.save(checkpoint, OUTPUT_DIR / 'best_model.pt')
        print(f"  ✓ Saved best model")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\nEarly stopping at epoch {epoch}")
        break
    
    scheduler.step()

# Save history
with open(OUTPUT_DIR / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)

# Test evaluation
print("\n" + "="*70)
print("TEST EVALUATION")
print("="*70)

test_metrics = evaluate_hybrid(model, test_graphs, CONFIG['batch_size'], device)
print(f"\nHybrid Model Test Performance:")
print(f"  Loss: {test_metrics['loss']:.4f}")
print(f"  R²: {test_metrics['r2']:.4f}")
print(f"  Pearson: {test_metrics['pearson']:.4f}")
print(f"  RMSE: {test_metrics['rmse']:.4f}")

# Comparison against baseline
print("\n" + "="*70)
print("COMPARISON: Structure-Only vs Hybrid (Structure + ESM-2)")
print("="*70)

baseline_checkpoint = torch.load(
    BASELINE_DIR / 'seed_42' / 'best_model.pt',
    weights_only=False
)

print("\nStructure-Only Model:")
print(f"  Val Loss: {baseline_checkpoint['val_metrics']['loss']:.4f}")
print(f"  R²: {baseline_checkpoint['val_metrics']['r2']:.4f}")
print(f"  Pearson: {baseline_checkpoint['val_metrics']['pearson']:.4f}")

print("\nHybrid Model (Structure + ESM-2 Sequence):")
hybrid_checkpoint = torch.load(OUTPUT_DIR / 'best_model.pt', weights_only=False)
print(f"  Val Loss: {hybrid_checkpoint['val_metrics']['loss']:.4f}")
print(f"  R²: {hybrid_checkpoint['val_metrics']['r2']:.4f}")
print(f"  Pearson: {hybrid_checkpoint['val_metrics']['pearson']:.4f}")

improvement_r2 = (hybrid_checkpoint['val_metrics']['r2'] - 
                 baseline_checkpoint['val_metrics']['r2']) / baseline_checkpoint['val_metrics']['r2'] * 100
improvement_abs = hybrid_checkpoint['val_metrics']['r2'] - baseline_checkpoint['val_metrics']['r2']

print(f"\nR² Change: {improvement_r2:+.1f}% ({improvement_abs:+.4f} absolute)")

if improvement_r2 > 3:
    print("✓ Hybrid model significantly improves performance!")
    print("  → ESM-2 sequence features complement structural information")
elif improvement_r2 > -3:
    print("→ Similar performance: Structure alone is sufficient")
    print("  → Interesting finding: Sequence doesn't add value for binding prediction")
else:
    print("→ Structure-only performs better")
    print("  → ESM features may add noise for this specific task")

print("\n" + "="*70)
print("✓ Hybrid training complete with REAL ESM-2 embeddings!")
print("="*70)