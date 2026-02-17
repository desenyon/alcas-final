"""
Create protein cluster-based train/val/test splits
Uses sequence similarity to avoid data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from collections import defaultdict
from tqdm import tqdm

# Configuration
META_FILE = Path("data/processed/meta.csv")
RAW_DIR = Path("data/raw/pdbbind")
SPLITS_DIR = Path("data/splits/protein_cluster")
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("="*70)
print("STEP 2: CREATING TRAIN/VAL/TEST SPLITS")
print("="*70)

# Set seed
np.random.seed(SEED)

# Load metadata
df = pd.read_csv(META_FILE)
print(f"\nLoaded: {len(df)} complexes")

# Helper function
def get_complex_path(pdb_code, year):
    """Get path to complex folder"""
    if year <= 2000:
        year_folder = "1981-2000"
    elif year <= 2010:
        year_folder = "2001-2010"
    else:
        year_folder = "2011-2019"
    return RAW_DIR / year_folder / pdb_code

# Extract protein sequences
print("\nExtracting protein sequences...")

AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

sequences = {}
parser = PDBParser(QUIET=True)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting sequences"):
    pdb = row['pdb_code']
    year = row['year']
    
    complex_path = get_complex_path(pdb, year)
    protein_file = complex_path / f"{pdb}_protein.pdb"
    
    try:
        structure = parser.get_structure(pdb, protein_file)
        
        # Get first chain sequence
        for chain in structure.get_chains():
            seq = ""
            for residue in chain:
                if residue.id[0] == ' ':  # Standard residue
                    resname = residue.resname
                    if resname in AA_MAP:
                        seq += AA_MAP[resname]
            
            if seq:
                sequences[pdb] = seq
                break
    except:
        continue

print(f"Extracted: {len(sequences)} sequences")

# Simple clustering by sequence similarity
print("\nClustering sequences by similarity (30% identity threshold)...")

def simple_sequence_identity(seq1, seq2):
    """Calculate sequence identity"""
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    min_len = min(len(seq1), len(seq2))
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    return matches / min_len

# Fast clustering using random sampling
pdb_codes = list(sequences.keys())
np.random.shuffle(pdb_codes)

clusters = {}
cluster_id = 0
assigned = set()

for i, pdb1 in enumerate(tqdm(pdb_codes, desc="Clustering")):
    if pdb1 in assigned:
        continue
    
    # Start new cluster
    cluster = [pdb1]
    assigned.add(pdb1)
    seq1 = sequences[pdb1]
    
    # Find similar sequences (sample 100 random for speed)
    candidates = [p for p in pdb_codes if p not in assigned]
    if len(candidates) > 100:
        candidates = np.random.choice(candidates, 100, replace=False).tolist()
    
    for pdb2 in candidates:
        seq2 = sequences[pdb2]
        identity = simple_sequence_identity(seq1, seq2)
        
        if identity >= 0.3:  # 30% identity threshold
            cluster.append(pdb2)
            assigned.add(pdb2)
    
    clusters[cluster_id] = cluster
    cluster_id += 1

# Assign remaining singletons
for pdb in pdb_codes:
    if pdb not in assigned:
        clusters[cluster_id] = [pdb]
        cluster_id += 1

print(f"Created: {len(clusters)} clusters")

# Split clusters into train/val/test
cluster_ids = list(clusters.keys())
np.random.shuffle(cluster_ids)

n_train = int(len(cluster_ids) * TRAIN_RATIO)
n_val = int(len(cluster_ids) * VAL_RATIO)

train_clusters = cluster_ids[:n_train]
val_clusters = cluster_ids[n_train:n_train+n_val]
test_clusters = cluster_ids[n_train+n_val:]

# Map PDB codes to splits
pdb_to_split = {}
for cluster_id in train_clusters:
    for pdb in clusters[cluster_id]:
        pdb_to_split[pdb] = 'train'
for cluster_id in val_clusters:
    for pdb in clusters[cluster_id]:
        pdb_to_split[pdb] = 'val'
for cluster_id in test_clusters:
    for pdb in clusters[cluster_id]:
        pdb_to_split[pdb] = 'test'

# Add split column to dataframe
df['split'] = df['pdb_code'].map(pdb_to_split)

# Remove entries without sequences
df_split = df.dropna(subset=['split'])

# Create split dataframes
train_df = df_split[df_split['split'] == 'train'].copy()
val_df = df_split[df_split['split'] == 'val'].copy()
test_df = df_split[df_split['split'] == 'test'].copy()

print(f"\nSplit results:")
print(f"  Train: {len(train_df)} ({len(train_df)/len(df_split)*100:.1f}%)")
print(f"  Val:   {len(val_df)} ({len(val_df)/len(df_split)*100:.1f}%)")
print(f"  Test:  {len(test_df)} ({len(test_df)/len(df_split)*100:.1f}%)")

# Save splits
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(SPLITS_DIR / 'train.csv', index=False)
val_df.to_csv(SPLITS_DIR / 'val.csv', index=False)
test_df.to_csv(SPLITS_DIR / 'test.csv', index=False)

print(f"\n✓ Saved to: {SPLITS_DIR}/")

print("\n" + "="*70)
print("STEP 2 COMPLETE")
print("="*70)