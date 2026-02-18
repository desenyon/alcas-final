"""
Hybrid Model: Structure (GNN) + Sequence (ESM-2)
Combines geometric graph features with protein language model embeddings
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d

class HybridAffinityModel(nn.Module):
    """
    Hybrid architecture combining:
    - Structure: GIN on protein-ligand graphs
    - Sequence: ESM-2 protein embeddings
    """
    
    def __init__(self, hidden_dim=256, num_layers=5, dropout=0.15, esm_dim=1280):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.esm_dim = esm_dim
        
        # Structure pathway (same as original model)
        # Ligand GIN
        self.ligand_gin_convs = nn.ModuleList()
        self.ligand_gin_bns = nn.ModuleList()
        
        # First layer
        nn_lig = Sequential(
            Linear(26, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.ligand_gin_convs.append(GINConv(nn_lig))
        self.ligand_gin_bns.append(BatchNorm1d(hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            nn_lig = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            )
            self.ligand_gin_convs.append(GINConv(nn_lig))
            self.ligand_gin_bns.append(BatchNorm1d(hidden_dim))
        
        # Protein GIN
        self.protein_gin_convs = nn.ModuleList()
        self.protein_gin_bns = nn.ModuleList()
        
        # First layer
        nn_prot = Sequential(
            Linear(20, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.protein_gin_convs.append(GINConv(nn_prot))
        self.protein_gin_bns.append(BatchNorm1d(hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            nn_prot = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            )
            self.protein_gin_convs.append(GINConv(nn_prot))
            self.protein_gin_bns.append(BatchNorm1d(hidden_dim))
        
        # Virtual nodes
        self.virtual_lig = nn.Parameter(torch.randn(hidden_dim))
        self.virtual_prot = nn.Parameter(torch.randn(hidden_dim))
        
        # Cross-attention
        self.cross_attn_lig = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.cross_attn_prot = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Sequence pathway (ESM-2 processing)
        self.esm_projector = Sequential(
            Linear(esm_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        
        # Fusion layer
        # Structure features: ligand (hidden_dim) + protein (hidden_dim) = 2*hidden_dim
        # Sequence features: hidden_dim
        # Total: 3*hidden_dim
        fusion_input_dim = 3 * hidden_dim
        
        self.fusion = Sequential(
            Linear(fusion_input_dim, hidden_dim * 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            Dropout(dropout)
        )
        
        # Final predictor
        self.predictor = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch):
        """
        Forward pass combining structure and sequence
        
        Args:
            batch: PyG batch with:
                - ligand_x, protein_x: Node features
                - ligand_edge_index, protein_edge_index: Edges
                - ligand_batch, protein_batch: Batch assignments
                - esm_embedding: ESM-2 sequence features [batch_size, 1280]
        """
        # === Structure Pathway ===
        
        # Ligand GIN
        lig_x = batch.ligand_x
        lig_edge_index = batch.ligand_edge_index
        lig_batch = batch.ligand_batch
        
        for conv, bn in zip(self.ligand_gin_convs, self.ligand_gin_bns):
            lig_x = conv(lig_x, lig_edge_index)
            lig_x = bn(lig_x)
            lig_x = torch.relu(lig_x)
            lig_x = self.dropout(lig_x)
        
        # Protein GIN
        prot_x = batch.protein_x
        prot_edge_index = batch.protein_edge_index
        prot_batch = batch.protein_batch
        
        for conv, bn in zip(self.protein_gin_convs, self.protein_gin_bns):
            prot_x = conv(prot_x, prot_edge_index)
            prot_x = bn(prot_x)
            prot_x = torch.relu(prot_x)
            prot_x = self.dropout(prot_x)
        
        # Virtual nodes
        batch_size = batch.y.size(0)
        
        virtual_lig_batch = self.virtual_lig.unsqueeze(0).expand(batch_size, -1)
        virtual_prot_batch = self.virtual_prot.unsqueeze(0).expand(batch_size, -1)
        
        # Cross-attention
        lig_pooled = global_mean_pool(lig_x, lig_batch).unsqueeze(1)
        prot_pooled = global_mean_pool(prot_x, prot_batch).unsqueeze(1)
        
        lig_attended, _ = self.cross_attn_lig(
            virtual_lig_batch.unsqueeze(1),
            prot_pooled,
            prot_pooled
        )
        
        prot_attended, _ = self.cross_attn_prot(
            virtual_prot_batch.unsqueeze(1),
            lig_pooled,
            lig_pooled
        )
        
        lig_features = lig_attended.squeeze(1)
        prot_features = prot_attended.squeeze(1)
        
        # === Sequence Pathway ===
        
        # Process ESM embeddings
        esm_emb = batch.esm_embedding  # [batch_size, 1280]
        esm_features = self.esm_projector(esm_emb)  # [batch_size, hidden_dim]
        
        # === Fusion ===
        
        # Concatenate structure + sequence features
        combined = torch.cat([lig_features, prot_features, esm_features], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Final prediction
        output = self.predictor(fused)
        
        return output


def test_hybrid_model():
    """Test hybrid model"""
    from types import SimpleNamespace
    
    # Create dummy batch
    batch = SimpleNamespace()
    batch.ligand_x = torch.randn(100, 26)
    batch.protein_x = torch.randn(200, 20)
    batch.ligand_edge_index = torch.randint(0, 100, (2, 300))
    batch.protein_edge_index = torch.randint(0, 200, (2, 600))
    batch.ligand_batch = torch.zeros(100, dtype=torch.long)
    batch.protein_batch = torch.zeros(200, dtype=torch.long)
    batch.esm_embedding = torch.randn(1, 1280)  # ESM features
    batch.y = torch.randn(1)
    
    # Create model
    model = HybridAffinityModel(hidden_dim=256, num_layers=5, dropout=0.15)
    
    # Test forward
    output = model(batch)
    
    print("Hybrid Model Test:")
    print(f"  Input: {batch.ligand_x.shape[0]} ligand atoms, {batch.protein_x.shape[0]} protein atoms")
    print(f"  ESM embedding: {batch.esm_embedding.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return output


if __name__ == "__main__":
    test_hybrid_model()