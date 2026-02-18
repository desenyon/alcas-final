"""
Simplified physics-informed loss (feature-based, no coordinates needed)
Adds regularization to encourage better generalization
"""

import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Simplified physics loss using only features (no coordinates)
    
    Terms:
    1. MSE (primary)
    2. Prediction diversity (prevent collapse)
    3. Feature utilization (use both protein and ligand info)
    """
    
    def __init__(self, 
                 alpha_mse=1.0,
                 alpha_diversity=0.02):
        super().__init__()
        
        self.alpha_mse = alpha_mse
        self.alpha_diversity = alpha_diversity
        
        self.mse = nn.MSELoss()
    
    def diversity_penalty(self, predictions):
        """
        Encourage diverse predictions (prevent collapse)
        Penalize if all predictions are too similar
        """
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Standard deviation of predictions
        pred_std = predictions.std()
        
        # Target std around 1.0 (similar to target distribution)
        target_std = 1.0
        
        # Penalty for being too uniform or too spread out
        penalty = (pred_std - target_std) ** 2
        
        return penalty
    
    def forward(self, predictions, targets, batch):
        """
        Compute combined loss
        
        Args:
            predictions: Model predictions [batch_size]
            targets: True affinities [batch_size]
            batch: Batch object
        """
        # Primary loss: MSE
        mse_loss = self.mse(predictions, targets)
        
        # Regularization
        diversity_loss = self.diversity_penalty(predictions)
        
        # Combined loss
        total_loss = (
            self.alpha_mse * mse_loss +
            self.alpha_diversity * diversity_loss
        )
        
        # Return total and components
        return total_loss, {
            'mse': mse_loss.item(),
            'diversity': diversity_loss.item(),
            'total': total_loss.item()
        }


def test_physics_loss():
    """Test simplified physics loss"""
    from types import SimpleNamespace
    
    # Create dummy batch
    batch = SimpleNamespace()
    batch.ligand_x = torch.randn(10, 26)
    batch.protein_x = torch.randn(20, 20)
    
    # Create loss
    loss_fn = PhysicsInformedLoss()
    
    # Test
    predictions = torch.randn(4)
    targets = torch.randn(4)
    
    total_loss, components = loss_fn(predictions, targets, batch)
    
    print("Simplified Physics Loss Test:")
    print(f"  Total: {total_loss.item():.4f}")
    for name, value in components.items():
        print(f"  {name}: {value:.4f}")
    
    return total_loss


if __name__ == "__main__":
    test_physics_loss()