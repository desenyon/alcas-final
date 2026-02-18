"""
Assign ESM-2 embeddings to graph dataset
Maps PDB codes to their protein's ESM embedding
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Config
GRAPHS_DIR = Path("data/graphs/protein_cluster")
SEQUENCES_FILE = Path("data/processed/protein_sequences.json")
EMBEDDINGS_FILE = Path("data/processed/esm2_embeddings/embeddings.json")
OUTPUT_DIR = Path("data/esm_embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ASSIGNING ESM-2 EMBEDDINGS TO GRAPHS")
print("="*70)

# Load sequences mapping
print("\nLoading sequence mapping...")
with open(SEQUENCES_FILE) as f:
    seq_data = json.load(f)

pdb_to_seq = seq_data['pdb_to_sequence']
print(f"PDB to sequence mappings: {len(pdb_to_seq)}")

# Load ESM embeddings
print("Loading ESM-2 embeddings...")
with open(EMBEDDINGS_FILE) as f:
    esm_data = json.load(f)

print(f"Unique embeddings: {len(esm_data)}")

# Create lookup: pdb_code -> embedding
pdb_to_embedding = {}

for pdb_code, seq_info in pdb_to_seq.items():
    seq_id = seq_info['seq_id']
    if seq_id in esm_data:
        embedding = np.array(esm_data[seq_id]['embedding'], dtype=np.float32)
        pdb_to_embedding[pdb_code] = torch.tensor(embedding)

print(f"PDB codes with embeddings: {len(pdb_to_embedding)}")

# Process each split
for split in ['train', 'val', 'test']:
    print(f"\nProcessing {split}...")
    
    graphs = torch.load(GRAPHS_DIR / f'{split}.pt', weights_only=False)
    
    assigned = 0
    missing = 0
    
    for g in tqdm(graphs, desc=f"  {split}"):
        pdb_code = g.pdb_code if hasattr(g, 'pdb_code') else None
        
        if pdb_code and pdb_code in pdb_to_embedding:
            g.esm_embedding = pdb_to_embedding[pdb_code]
            assigned += 1
        else:
            # Use mean embedding as fallback
            mean_emb = torch.zeros(1280, dtype=torch.float32)
            g.esm_embedding = mean_emb
            missing += 1
    
    # Save
    torch.save(graphs, OUTPUT_DIR / f'{split}_with_esm.pt')
    
    print(f"  ✓ {split}: {assigned} assigned, {missing} missing")

print(f"\n✓ Saved to: {OUTPUT_DIR}")

print("\n" + "="*70)
print("COMPLETE - Ready for hybrid training!")
print("="*70)