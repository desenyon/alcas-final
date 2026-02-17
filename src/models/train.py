"""
Training script for ALCAS affinity model
Trains ensemble of 4 models with different seeds
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent))
from affinity_model import AffinityModel

# Configuration
GRAPHS_DIR = Path("data/graphs/protein_cluster")
RESULTS_DIR = Path("results/models/affinity")

CONFIG = {
    'hidden_dim': 256,
    'num_layers': 5,
    'dropout': 0.15,
    'batch_size': 32,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 20,
    'grad_clip': 1.0,
    'warmup_epochs': 5,
}

def manual_batch(graphs, device):
    """Manually batch graphs"""
    from types import SimpleNamespace
    
    batch_data = {
        'ligand_x': [],
        'ligand_edge_index': [],
        'ligand_edge_attr': [],
        'protein_x': [],
        'protein_edge_index': [],
        'protein_edge_attr': [],
        'ligand_batch': [],
        'protein_batch': [],
        'y': []
    }
    
    lig_offset = 0
    prot_offset = 0
    
    for i, g in enumerate(graphs):
        batch_data['ligand_x'].append(g.ligand_x)
        batch_data['ligand_edge_index'].append(g.ligand_edge_index + lig_offset)
        batch_data['ligand_edge_attr'].append(g.ligand_edge_attr)
        batch_data['ligand_batch'].append(torch.full((g.ligand_x.size(0),), i, dtype=torch.long))
        
        batch_data['protein_x'].append(g.protein_x)
        batch_data['protein_edge_index'].append(g.protein_edge_index + prot_offset)
        batch_data['protein_edge_attr'].append(g.protein_edge_attr)
        batch_data['protein_batch'].append(torch.full((g.protein_x.size(0),), i, dtype=torch.long))
        
        batch_data['y'].append(g.y)
        
        lig_offset += g.ligand_x.size(0)
        prot_offset += g.protein_x.size(0)
    
    batch = SimpleNamespace()
    batch.ligand_x = torch.cat(batch_data['ligand_x']).to(device)
    batch.ligand_edge_index = torch.cat(batch_data['ligand_edge_index'], dim=1).to(device)
    batch.ligand_edge_attr = torch.cat(batch_data['ligand_edge_attr']).to(device)
    batch.ligand_batch = torch.cat(batch_data['ligand_batch']).to(device)
    
    batch.protein_x = torch.cat(batch_data['protein_x']).to(device)
    batch.protein_edge_index = torch.cat(batch_data['protein_edge_index'], dim=1).to(device)
    batch.protein_edge_attr = torch.cat(batch_data['protein_edge_attr']).to(device)
    batch.protein_batch = torch.cat(batch_data['protein_batch']).to(device)
    
    batch.y = torch.cat(batch_data['y']).to(device)
    batch.num_graphs = len(graphs)
    
    return batch

def train_epoch(model, graphs, batch_size, optimizer, device, grad_clip=None):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    indices = torch.randperm(len(graphs)).tolist()
    
    for i in tqdm(range(0, len(graphs), batch_size), desc="Training", leave=False):
        batch_indices = indices[i:i+batch_size]
        batch_graphs = [graphs[idx] for idx in batch_indices]
        
        batch = manual_batch(batch_graphs, device)
        
        optimizer.zero_grad()
        pred = model(batch).squeeze()
        loss = nn.MSELoss()(pred, batch.y.squeeze())
        loss.backward()
        
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * len(batch_graphs)
    
    return total_loss / len(graphs)

@torch.no_grad()
def evaluate(model, graphs, batch_size, device):
    """Evaluate"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for i in tqdm(range(0, len(graphs), batch_size), desc="Evaluating", leave=False):
        batch_graphs = graphs[i:i+batch_size]
        batch = manual_batch(batch_graphs, device)
        
        pred = model(batch).squeeze()
        loss = nn.MSELoss()(pred, batch.y.squeeze())
        
        total_loss += loss.item() * len(batch_graphs)
        all_preds.append(pred.cpu())
        all_targets.append(batch.y.squeeze().cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    mse = total_loss / len(graphs)
    rmse = np.sqrt(mse)
    
    ss_res = ((all_targets - all_preds) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    
    pred_mean = all_preds.mean()
    targ_mean = all_targets.mean()
    numerator = ((all_preds - pred_mean) * (all_targets - targ_mean)).sum()
    denominator = torch.sqrt(((all_preds - pred_mean) ** 2).sum() * 
                             ((all_targets - targ_mean) ** 2).sum())
    pearson = numerator / (denominator + 1e-8)
    
    return {
        'loss': mse,
        'rmse': rmse,
        'r2': r2.item(),
        'pearson': pearson.item()
    }

def train_single_model(seed, config):
    """Train one model"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL - SEED {seed}")
    print(f"{'='*70}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading graphs...")
    train_graphs = torch.load(GRAPHS_DIR / 'train.pt', weights_only=False)
    val_graphs = torch.load(GRAPHS_DIR / 'val.pt', weights_only=False)
    print(f"  Train: {len(train_graphs)}")
    print(f"  Val: {len(val_graphs)}")
    
    # Create model
    print("\nCreating model...")
    model = AffinityModel(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        return 0.5 * (1 + np.cos(np.pi * (epoch - config['warmup_epochs']) / 
                                 (config['epochs'] - config['warmup_epochs'])))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    output_dir = RESULTS_DIR / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_pearson': [], 'lr': []}
    
    print("\nTraining...\n")
    
    for epoch in range(1, config['epochs'] + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{config['epochs']} (LR: {current_lr:.2e})")
        
        train_loss = train_epoch(
            model, train_graphs, config['batch_size'], 
            optimizer, device, config['grad_clip']
        )
        
        val_metrics = evaluate(model, val_graphs, config['batch_size'], device)
        scheduler.step()
        
        print(f"  Train: {train_loss:.4f}")
        print(f"  Val: {val_metrics['loss']:.4f} | RMSE: {val_metrics['rmse']:.4f} | "
              f"R²: {val_metrics['r2']:.4f} | Pearson: {val_metrics['pearson']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_r2'].append(val_metrics['r2'])
        history['val_pearson'].append(val_metrics['pearson'])
        history['lr'].append(current_lr)
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved (best: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['patience']}")
        
        if patience_counter >= config['patience']:
            print(f"\n  Early stopping at epoch {epoch}")
            break
        
        print()
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training Complete - Seed {seed}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")
    
    return best_val_loss

def main():
    """Main training function"""
    
    print("\n" + "="*70)
    print("ALCAS AFFINITY MODEL TRAINING")
    print("="*70)
    
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Train ensemble (4 seeds)
    seeds = [42, 43, 44, 45]
    results = {}
    
    for seed in seeds:
        best_loss = train_single_model(seed, CONFIG)
        results[f'seed_{seed}'] = best_loss
    
    # Summary
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE")
    print("="*70)
    print("\nResults:")
    for seed_name, loss in results.items():
        print(f"  {seed_name}: {loss:.4f}")
    
    mean_loss = np.mean(list(results.values()))
    std_loss = np.std(list(results.values()))
    print(f"\nEnsemble: {mean_loss:.4f} ± {std_loss:.4f}")
    print("\n✓ Ready for evaluation!")

if __name__ == '__main__':
    main()