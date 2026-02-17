"""
Verify the parsed metadata
"""

import pandas as pd
from pathlib import Path

META_FILE = Path("data/processed/meta.csv")

print("="*70)
print("VERIFYING PARSED METADATA")
print("="*70)

# Load
df = pd.read_csv(META_FILE)

print(f"\n✓ Loaded: {META_FILE}")
print(f"  Total complexes: {len(df)}")

# Check columns
print(f"\nColumns: {list(df.columns)}")

# Statistics
print("\n" + "="*70)
print("STATISTICS")
print("="*70)

print(f"\nResolution:")
print(f"  Range: {df['resolution'].min():.2f} - {df['resolution'].max():.2f} Å")
print(f"  Mean: {df['resolution'].mean():.2f} Å")
print(f"  Median: {df['resolution'].median():.2f} Å")

print(f"\nAffinity (pKd):")
print(f"  Range: {df['paffinity'].min():.2f} - {df['paffinity'].max():.2f}")
print(f"  Mean: {df['paffinity'].mean():.2f}")
print(f"  Median: {df['paffinity'].median():.2f}")

print(f"\nYears:")
print(f"  Range: {df['year'].min()} - {df['year'].max()}")

print(f"\nMeasure types:")
for measure, count in df['measure_type'].value_counts().items():
    print(f"  {measure}: {count} ({count/len(df)*100:.1f}%)")

# Show first few rows
print("\n" + "="*70)
print("SAMPLE DATA")
print("="*70)
print(df.head(10).to_string(index=False))

# Check for issues
print("\n" + "="*70)
print("DATA QUALITY CHECKS")
print("="*70)

issues = 0

# Check for NaN
nan_cols = df.columns[df.isna().any()].tolist()
if nan_cols:
    print(f"✗ NaN values found in: {nan_cols}")
    issues += 1
else:
    print("✓ No NaN values")

# Check duplicates
dupes = df['pdb_code'].duplicated().sum()
if dupes > 0:
    print(f"✗ {dupes} duplicate PDB codes")
    issues += 1
else:
    print("✓ No duplicate PDB codes")

# Check value ranges
if (df['paffinity'] < 4.0).any() or (df['paffinity'] > 13.0).any():
    print("✗ pAffinity values outside expected range (4-13)")
    issues += 1
else:
    print("✓ All pAffinity values in range (4-13)")

if (df['resolution'] > 3.0).any():
    print("✗ Resolution values > 3.0 Å")
    issues += 1
else:
    print("✓ All resolution values ≤ 3.0 Å")

# Summary
print("\n" + "="*70)
if issues == 0:
    print("✓ ALL CHECKS PASSED - DATA IS READY")
else:
    print(f"✗ {issues} ISSUES FOUND - REVIEW ABOVE")
print("="*70)