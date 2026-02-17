"""
ISEF-Quality Geometric GNN for Protein-Ligand Binding Affinity
Uses GIN layers, virtual nodes, and cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, LayerNorm

class GINLayer(nn.Module):
    """Graph Isomorphism Network layer - most expressive GNN"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.mlp = Sequential(
            Linear(in_dim, 2 * out_dim),
            BatchNorm1d(2 * out_dim),
            ReLU(),
            Linear(2 * out_dim, out_dim)
        )
        
        self.conv = GINConv(self.mlp, train_eps=True)
        self.bn = BatchNorm1d(out_dim)
        
    def forward(self, x, edge_index):
        return F.relu(self.bn(self.conv(x, edge_index)))

class AttentivePooling(nn.Module):
    """Learnable attention-weighted pooling"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_net = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, batch):
        # Compute attention scores
        attn_scores = self.attn_net(x)
        
        # Softmax per graph
        batch_size = batch.max().item() + 1
        pooled = []
        
        for i in range(batch_size):
            mask = batch == i
            graph_x = x[mask]
            graph_attn = attn_scores[mask]
            
            # Softmax attention
            attn_weights = F.softmax(graph_attn, dim=0)
            
            # Weighted sum
            pooled_graph = (graph_x * attn_weights).sum(dim=0)
            pooled.append(pooled_graph)
        
        return torch.stack(pooled)

class AffinityModel(nn.Module):
    """
    Ultimate Binding Affinity Prediction Model
    
    Features:
    - GIN layers (most expressive GNN)
    - Virtual nodes (global graph state)
    - Attentive pooling (learnable aggregation)
    - Multiple pooling strategies (mean + max + attention)
    - Deep residual connections
    - Extensive regularization
    """
    
    def __init__(self,
                 ligand_node_dim=26,
                 protein_node_dim=20,
                 hidden_dim=256,
                 num_layers=5,
                 dropout=0.15):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embeddings
        self.ligand_embed = Sequential(
            Linear(ligand_node_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        
        self.protein_embed = Sequential(
            Linear(protein_node_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        
        # Virtual nodes
        self.ligand_virtual = nn.Parameter(torch.randn(1, hidden_dim))
        self.protein_virtual = nn.Parameter(torch.randn(1, hidden_dim))
        
        # GIN layers (Ligand)
        self.ligand_convs = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.ligand_virtual_mlps = nn.ModuleList([
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU()
            ) for _ in range(num_layers)
        ])
        
        # GIN layers (Protein)
        self.protein_convs = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.protein_virtual_mlps = nn.ModuleList([
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU()
            ) for _ in range(num_layers)
        ])
        
        # Cross-graph interaction
        self.cross_attn_lig = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.cross_attn_prot = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.cross_norm_lig = LayerNorm(hidden_dim)
        self.cross_norm_prot = LayerNorm(hidden_dim)
        
        # Pooling
        self.attentive_pool_lig = AttentivePooling(hidden_dim)
        self.attentive_pool_prot = AttentivePooling(hidden_dim)
        
        # Prediction head (6 * hidden_dim input)
        self.predictor = Sequential(
            Linear(6 * hidden_dim, 4 * hidden_dim),
            LayerNorm(4 * hidden_dim),
            ReLU(),
            nn.Dropout(dropout),
            
            Linear(4 * hidden_dim, 2 * hidden_dim),
            LayerNorm(2 * hidden_dim),
            ReLU(),
            nn.Dropout(dropout),
            
            Linear(2 * hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            nn.Dropout(dropout),
            
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            nn.Dropout(dropout),
            
            Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
        
    def forward(self, data):
        """Forward pass"""
        
        batch_size = data.ligand_batch.max().item() + 1
        
        # Embed
        lig_h = self.ligand_embed(data.ligand_x)
        prot_h = self.protein_embed(data.protein_x)
        
        # Initialize virtual nodes
        lig_virtual = self.ligand_virtual.expand(batch_size, -1)
        prot_virtual = self.protein_virtual.expand(batch_size, -1)
        
        # Message passing with virtual nodes
        for i in range(self.num_layers):
            # Ligand: aggregate to virtual node
            lig_virtual_agg = global_add_pool(lig_h, data.ligand_batch)
            lig_virtual = lig_virtual + lig_virtual_agg
            lig_virtual = self.ligand_virtual_mlps[i](lig_virtual)
            
            # Ligand: broadcast from virtual node
            lig_h = lig_h + lig_virtual[data.ligand_batch]
            
            # Ligand: message passing
            lig_h = lig_h + self.ligand_convs[i](lig_h, data.ligand_edge_index)
            lig_h = F.dropout(lig_h, p=self.dropout, training=self.training)
            
            # Protein: same process
            prot_virtual_agg = global_add_pool(prot_h, data.protein_batch)
            prot_virtual = prot_virtual + prot_virtual_agg
            prot_virtual = self.protein_virtual_mlps[i](prot_virtual)
            
            prot_h = prot_h + prot_virtual[data.protein_batch]
            prot_h = prot_h + self.protein_convs[i](prot_h, data.protein_edge_index)
            prot_h = F.dropout(prot_h, p=self.dropout, training=self.training)
        
        # Cross-graph attention
        lig_pooled_pre = global_mean_pool(lig_h, data.ligand_batch)
        prot_pooled_pre = global_mean_pool(prot_h, data.protein_batch)
        
        # Reshape for attention
        lig_q = lig_pooled_pre.unsqueeze(1)
        prot_q = prot_pooled_pre.unsqueeze(1)
        
        # Cross-attention
        lig_attn_out, _ = self.cross_attn_lig(lig_q, prot_q, prot_q)
        prot_attn_out, _ = self.cross_attn_prot(prot_q, lig_q, lig_q)
        
        # Add & norm
        lig_pooled_pre = self.cross_norm_lig(lig_pooled_pre + lig_attn_out.squeeze(1))
        prot_pooled_pre = self.cross_norm_prot(prot_pooled_pre + prot_attn_out.squeeze(1))
        
        # Broadcast back
        lig_h = lig_h + lig_pooled_pre[data.ligand_batch]
        prot_h = prot_h + prot_pooled_pre[data.protein_batch]
        
        # Multi-strategy pooling
        lig_mean = global_mean_pool(lig_h, data.ligand_batch)
        lig_max = global_max_pool(lig_h, data.ligand_batch)
        lig_attn = self.attentive_pool_lig(lig_h, data.ligand_batch)
        
        prot_mean = global_mean_pool(prot_h, data.protein_batch)
        prot_max = global_max_pool(prot_h, data.protein_batch)
        prot_attn = self.attentive_pool_prot(prot_h, data.protein_batch)
        
        # Concatenate
        graph_repr = torch.cat([
            lig_mean, lig_max, lig_attn,
            prot_mean, prot_max, prot_attn
        ], dim=-1)
        
        # Predict
        out = self.predictor(graph_repr)
        
        return out