"""
Identify active site and allosteric residues in PETase
"""

import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
import json

PETASE_HOLO = Path("data/raw/petase/6EQE.pdb")
OUTPUT_DIR = Path("data/processed/petase")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PETASE RESIDUE IDENTIFICATION")
print("="*70)

# Known catalytic triad
CATALYTIC_RESIDUES = [160, 206, 237]  # Ser160, Asp206, His237

print(f"\nCatalytic triad: {CATALYTIC_RESIDUES}")

# Parse structure
parser = PDBParser(QUIET=True)
structure = parser.get_structure('petase', PETASE_HOLO)

# Get all residues
all_residues = []
residue_coords = {}

for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] == ' ' and 'CA' in residue:
                res_num = residue.id[1]
                all_residues.append(res_num)
                residue_coords[res_num] = residue['CA'].coord

print(f"Total residues with CA: {len(all_residues)}")

# Active site center (from catalytic residues)
cat_coords = np.array([residue_coords[r] for r in CATALYTIC_RESIDUES])
active_center = cat_coords.mean(axis=0)

print(f"Active site center: {active_center}")

# Identify active site (≤8Å from center)
print("\n" + "="*70)
print("ACTIVE SITE RESIDUES")
print("="*70)

active_site = []
for res_num in all_residues:
    coord = residue_coords[res_num]
    dist = np.linalg.norm(coord - active_center)
    
    if dist <= 8.0 or res_num in CATALYTIC_RESIDUES:
        active_site.append(res_num)

active_site = sorted(active_site)
print(f"Active site: {len(active_site)} residues (≤8Å from catalytic center)")
print(f"Residues: {active_site}")

# Identify allosteric candidates (improved criteria)
print("\n" + "="*70)
print("ALLOSTERIC RESIDUES")
print("="*70)

# Get active site coordinates
active_coords = np.array([residue_coords[r] for r in active_site])

allosteric = []
allosteric_distances = {}

for res_num in all_residues:
    # Skip catalytic and active site
    if res_num in CATALYTIC_RESIDUES or res_num in active_site:
        continue
    
    coord = residue_coords[res_num]
    
    # Distance to nearest active site residue
    min_dist = np.min(np.linalg.norm(active_coords - coord, axis=1))
    
    # Criteria: >12Å from active site
    if min_dist > 12.0:
        allosteric.append(res_num)
        allosteric_distances[res_num] = min_dist

allosteric = sorted(allosteric)

print(f"Allosteric candidates: {len(allosteric)} residues (>12Å from active site)")
print(f"\nDistance distribution:")
print(f"  Min: {min(allosteric_distances.values()):.1f} Å")
print(f"  Max: {max(allosteric_distances.values()):.1f} Å")
print(f"  Mean: {np.mean(list(allosteric_distances.values())):.1f} Å")

# Select diverse allosteric residues (top 50 by distance)
if len(allosteric) > 50:
    # Sort by distance, take top 50
    sorted_by_dist = sorted(allosteric_distances.items(), key=lambda x: x[1], reverse=True)
    allosteric_selected = sorted([r for r, d in sorted_by_dist[:50]])
    print(f"\nSelected top 50 most distant allosteric residues")
else:
    allosteric_selected = allosteric

print(f"Final allosteric set: {len(allosteric_selected)} residues")

# Create residue masks
masks = {
    'catalytic': CATALYTIC_RESIDUES,
    'active_site': active_site,
    'allosteric': allosteric_selected,
    'allosteric_all': allosteric,  # Keep all for reference
    'total_residues': len(all_residues),
    'active_site_center': active_center.tolist(),
    'summary': {
        'total': len(all_residues),
        'catalytic': len(CATALYTIC_RESIDUES),
        'active_site': len(active_site),
        'allosteric': len(allosteric_selected)
    }
}

# Save
output_file = OUTPUT_DIR / 'residue_masks.json'
with open(output_file, 'w') as f:
    json.dump(masks, f, indent=2)

print(f"\n✓ Saved to: {output_file}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total residues: {len(all_residues)}")
print(f"Catalytic triad: {len(CATALYTIC_RESIDUES)}")
print(f"Active site (≤8Å): {len(active_site)}")
print(f"Allosteric (>12Å): {len(allosteric_selected)}")

print("\nFor ALCAS search:")
print(f"  Active-site constrained: {len(active_site)} mutable positions")
print(f"  Allosteric-constrained: {len(allosteric_selected)} mutable positions")

print("\n" + "="*70)
print("RESIDUE IDENTIFICATION COMPLETE")
print("="*70)