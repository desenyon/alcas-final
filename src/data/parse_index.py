"""
Parse PDBbind index file and filter to high-quality complexes
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RAW_DIR = Path("data/raw/pdbbind")
INDEX_FILE = RAW_DIR / "IndexFiles/index-p-l.lst"
OUTPUT_FILE = Path("data/processed/meta.csv")

# Quality filters
MAX_RESOLUTION = 3.0
MIN_PAFFINITY = 4.0
MAX_PAFFINITY = 13.0

print("="*70)
print("STEP 1: PARSING PDBBIND INDEX")
print("="*70)

print(f"\nReading: {INDEX_FILE}")

# Parse index file
data = []

with open(INDEX_FILE) as f:
    for line in f:
        # Skip comments and empty lines
        if line.startswith('#') or not line.strip():
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        pdb_code = parts[0]
        resolution = parts[1]
        year = int(parts[2])
        binding_data = parts[3]
        
        # Parse measurement type (Kd, Ki, IC50)
        if binding_data.startswith('Kd'):
            measure_type = 'Kd'
            rest = binding_data[3:]
        elif binding_data.startswith('Ki'):
            measure_type = 'Ki'
            rest = binding_data[3:]
        elif binding_data.startswith('IC50'):
            measure_type = 'IC50'
            rest = binding_data[5:]
        else:
            continue
        
        # Parse operator (=, <, >, ~)
        if rest[0] in ['=', '<', '>', '~']:
            operator = rest[0]
            value_str = rest[1:]
        else:
            operator = '='
            value_str = rest
        
        # Parse value and convert to Molar
        try:
            value_str = value_str.upper()
            
            # Extract number and unit
            if 'MM' in value_str:
                value = float(value_str.replace('MM', '')) * 1e-3
            elif 'UM' in value_str:
                value = float(value_str.replace('UM', '')) * 1e-6
            elif 'NM' in value_str:
                value = float(value_str.replace('NM', '')) * 1e-9
            elif 'PM' in value_str:
                value = float(value_str.replace('PM', '')) * 1e-12
            elif 'FM' in value_str:
                value = float(value_str.replace('FM', '')) * 1e-15
            elif 'M' in value_str:
                value = float(value_str.replace('M', ''))
            else:
                continue
            
            # Convert to pAffinity (-log10(M))
            paffinity = -np.log10(value)
            
            data.append({
                'pdb_code': pdb_code,
                'resolution': resolution,
                'year': year,
                'measure_type': measure_type,
                'operator': operator,
                'affinity_m': value,
                'paffinity': paffinity
            })
            
        except (ValueError, ZeroDivisionError):
            continue

# Create DataFrame
df = pd.DataFrame(data)
print(f"\nParsed: {len(df)} complexes")

# Apply filters
print("\nApplying filters...")

# Filter 1: Resolution
df_res = df[df['resolution'] != 'NMR'].copy()
df_res['resolution'] = df_res['resolution'].astype(float)
df_filtered = df_res[df_res['resolution'] <= MAX_RESOLUTION]
print(f"  Resolution ≤{MAX_RESOLUTION}Å: {len(df_filtered)}")

# Filter 2: Affinity range
df_filtered = df_filtered[
    (df_filtered['paffinity'] >= MIN_PAFFINITY) & 
    (df_filtered['paffinity'] <= MAX_PAFFINITY)
]
print(f"  Affinity {MIN_PAFFINITY}-{MAX_PAFFINITY} pKd: {len(df_filtered)}")

# Filter 3: Exact measurements only
df_filtered = df_filtered[df_filtered['operator'] == '=']
print(f"  Exact measurements (=): {len(df_filtered)}")

# Verify files exist
print("\nVerifying files exist...")

def get_complex_path(pdb_code, year):
    """Get path to complex folder"""
    if year <= 2000:
        year_folder = "1981-2000"
    elif year <= 2010:
        year_folder = "2001-2010"
    else:
        year_folder = "2011-2019"
    
    return RAW_DIR / year_folder / pdb_code

valid_indices = []
for idx, row in df_filtered.iterrows():
    complex_path = get_complex_path(row['pdb_code'], row['year'])
    protein_file = complex_path / f"{row['pdb_code']}_protein.pdb"
    ligand_file = complex_path / f"{row['pdb_code']}_ligand.mol2"
    
    if protein_file.exists() and ligand_file.exists():
        valid_indices.append(idx)

df_final = df_filtered.loc[valid_indices].copy()
print(f"  Files verified: {len(df_final)}")

# Save
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"\n✓ Saved: {OUTPUT_FILE}")
print(f"\nFinal dataset: {len(df_final)} complexes")
print(f"  Resolution: {df_final['resolution'].min():.2f}-{df_final['resolution'].max():.2f}Å (mean: {df_final['resolution'].mean():.2f}Å)")
print(f"  Affinity: {df_final['paffinity'].min():.2f}-{df_final['paffinity'].max():.2f} pKd (mean: {df_final['paffinity'].mean():.2f})")
print(f"  Years: {df_final['year'].min()}-{df_final['year'].max()}")

print("\n" + "="*70)
print("STEP 1 COMPLETE")
print("="*70)