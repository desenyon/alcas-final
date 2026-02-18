"""
Extract ESM-2 embeddings using fair-esm (Facebook's implementation)
More reliable than transformers library
"""

import torch
import esm
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Config
SEQUENCES_FILE = Path("data/processed/protein_sequences.json")
OUTPUT_DIR = Path("data/processed/esm2_embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ESM-2 EMBEDDING EXTRACTION")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load sequences
print("\nLoading sequences...")
with open(SEQUENCES_FILE) as f:
    seq_data = json.load(f)

unique_seqs = seq_data['unique_sequences']
print(f"Unique sequences: {len(unique_seqs)}")

# Load ESM-2 model (650M parameters)
print("\nLoading ESM-2 model (esm2_t33_650M_UR50D)")
print("(Downloading ~2.5GB on first run...)")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()

batch_converter = alphabet.get_batch_converter()

print("✓ Model loaded")

# Extract embeddings
embeddings = {}
batch_size = 8  # Process 8 sequences at a time

# Prepare data
seq_ids = list(unique_seqs.keys())
data = [(sid, unique_seqs[sid]['sequence']) for sid in seq_ids]

print(f"\nExtracting embeddings (batch_size={batch_size})...")
print(f"Total batches: {len(data) // batch_size + 1}")

with torch.no_grad():
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i+batch_size]
        
        # Convert batch
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        
        # Get representations
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        
        # Mean pooling over sequence length (ignore special tokens)
        for j, (seq_id, seq) in enumerate(batch_data):
            # Get sequence length (excluding BOS/EOS tokens)
            seq_len = len(seq)
            
            # Mean pool over sequence tokens (1 to seq_len+1, skip BOS at 0)
            embedding = token_representations[j, 1:seq_len+1].mean(0)
            embedding = embedding.cpu().numpy()
            
            embeddings[seq_id] = {
                'embedding': embedding.tolist(),
                'dim': len(embedding),
                'sequence_length': len(unique_seqs[seq_id]['sequence']),
                'pdb_codes': unique_seqs[seq_id]['pdb_codes']
            }

print(f"\n✓ Extracted {len(embeddings)} embeddings")

# Save
print("\nSaving embeddings...")

with open(OUTPUT_DIR / 'embeddings.json', 'w') as f:
    json.dump(embeddings, f)

metadata = {
    'model': 'esm2_t33_650M_UR50D',
    'embedding_dim': 1280,
    'num_sequences': len(embeddings),
    'device': str(device)
}

with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved: {OUTPUT_DIR / 'embeddings.json'}")

# Verify
sample_id = list(embeddings.keys())[0]
sample_emb = np.array(embeddings[sample_id]['embedding'])
print(f"\nVerification:")
print(f"  Embedding dim: {sample_emb.shape}")
print(f"  Norm: {np.linalg.norm(sample_emb):.4f}")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)