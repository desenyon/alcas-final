"""
Verify the train/val/test splits
"""

import pandas as pd
from pathlib import Path

SPLITS_DIR = Path("data/splits/protein_cluster")

print("="*70)
print("VERIFYING SPLITS")
print("="*70)

# Load splits
train_df = pd.read_csv(SPLITS_DIR / 'train.csv')
val_df = pd.read_csv(SPLITS_DIR / 'val.csv')
test_df = pd.read_csv(SPLITS_DIR / 'test.csv')

total = len(train_df) + len(val_df) + len(test_df)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
print(f"  Val:   {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
print(f"  Test:  {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
print(f"  Total: {total:,}")

# Check for overlap
print("\n" + "="*70)
print("OVERLAP CHECKS")
print("="*70)

train_pdbs = set(train_df['pdb_code'])
val_pdbs = set(val_df['pdb_code'])
test_pdbs = set(test_df['pdb_code'])

train_val_overlap = train_pdbs & val_pdbs
train_test_overlap = train_pdbs & test_pdbs
val_test_overlap = val_pdbs & test_pdbs

issues = 0

if len(train_val_overlap) > 0:
    print(f"✗ Train/Val overlap: {len(train_val_overlap)} complexes")
    issues += 1
else:
    print("✓ No Train/Val overlap")

if len(train_test_overlap) > 0:
    print(f"✗ Train/Test overlap: {len(train_test_overlap)} complexes")
    issues += 1
else:
    print("✓ No Train/Test overlap")

if len(val_test_overlap) > 0:
    print(f"✗ Val/Test overlap: {len(val_test_overlap)} complexes")
    issues += 1
else:
    print("✓ No Val/Test overlap")

# Distribution checks
print("\n" + "="*70)
print("DISTRIBUTION CHECKS")
print("="*70)

for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    print(f"\n{split_name}:")
    print(f"  Resolution: {split_df['resolution'].mean():.2f} ± {split_df['resolution'].std():.2f} Å")
    print(f"  Affinity:   {split_df['paffinity'].mean():.2f} ± {split_df['paffinity'].std():.2f} pKd")
    print(f"  Year range: {split_df['year'].min()}-{split_df['year'].max()}")

# Summary
print("\n" + "="*70)
if issues == 0:
    print("✓ ALL CHECKS PASSED - SPLITS ARE VALID")
else:
    print(f"✗ {issues} ISSUES FOUND")
print("="*70)