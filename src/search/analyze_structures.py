"""
Analyze folded structures and calculate stability metrics
"""

import json
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, NeighborSearch
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

# Config
STRUCTURES_DIR = Path("results/search/structures_esm")
METADATA_FILE = STRUCTURES_DIR / "folding_metadata.json"
OUTPUT_FILE = Path("results/search/structure_analysis.json")

print("="*70)
print("STRUCTURE ANALYSIS + STABILITY SCORING")
print("="*70)

# Load metadata
with open(METADATA_FILE) as f:
    metadata = json.load(f)

variants = [v for v in metadata['variants'] if v.get('folded', False)]
print(f"\nAnalyzing {len(variants)} folded structures")

parser = PDBParser(QUIET=True)

def calculate_structure_metrics(pdb_file):
    """Calculate structural quality metrics"""
    structure = parser.get_structure('protein', pdb_file)
    
    # Get all atoms
    atoms = [atom for atom in structure.get_atoms()]
    
    if len(atoms) == 0:
        return None
    
    # Get CA atoms
    ca_atoms = [atom for atom in structure.get_atoms() if atom.name == 'CA']
    
    if len(ca_atoms) < 10:
        return None
    
    # Calculate metrics
    metrics = {}
    
    # 1. Radius of gyration (compactness)
    coords = np.array([atom.coord for atom in ca_atoms])
    center = coords.mean(axis=0)
    rg = np.sqrt(((coords - center)**2).sum(axis=1).mean())
    metrics['radius_of_gyration'] = float(rg)
    
    # 2. Contact density (more contacts = more stable)
    ns = NeighborSearch(list(structure.get_atoms()))
    contacts = 0
    for atom in ca_atoms:
        neighbors = ns.search(atom.coord, 8.0, level='A')
        contacts += len([n for n in neighbors if n.parent != atom.parent])
    
    contact_density = contacts / len(ca_atoms)
    metrics['contact_density'] = float(contact_density)
    
    # 3. Secondary structure content proxy (CA-CA distances)
    # Shorter average distances = more structured
    distances = distance.pdist(coords)
    metrics['mean_ca_distance'] = float(distances.mean())
    metrics['std_ca_distance'] = float(distances.std())
    
    # 4. B-factor statistics (flexibility - lower is better)
    b_factors = [atom.bfactor for atom in ca_atoms]
    metrics['mean_bfactor'] = float(np.mean(b_factors))
    metrics['std_bfactor'] = float(np.std(b_factors))
    
    # 5. Clash score (count close contacts)
    clash_threshold = 2.0  # Angstroms
    clashes = 0
    for i, atom1 in enumerate(atoms):
        neighbors = ns.search(atom1.coord, clash_threshold, level='A')
        for atom2 in neighbors:
            if atom2.parent != atom1.parent:
                dist = atom1 - atom2
                if dist < clash_threshold:
                    clashes += 1
    
    metrics['clash_count'] = clashes // 2  # Divide by 2 (counted twice)
    
    return metrics

# Analyze all structures
print("\nAnalyzing structures...")

from tqdm import tqdm

for variant in tqdm(variants, desc="Analyzing"):
    pdb_file = variant['pdb_file']
    
    metrics = calculate_structure_metrics(pdb_file)
    
    if metrics:
        variant['structure_metrics'] = metrics
        
        # Calculate stability score (composite)
        # Lower rg, higher contact_density, fewer clashes = better
        stability_score = (
            -0.1 * metrics['radius_of_gyration'] +
            2.0 * metrics['contact_density'] +
            -0.5 * metrics['clash_count'] +
            -0.05 * metrics['mean_bfactor']
        )
        
        variant['stability_score'] = float(stability_score)
    else:
        variant['structure_metrics'] = None
        variant['stability_score'] = None

# Filter successful
analyzed = [v for v in variants if v.get('stability_score') is not None]

print(f"\n✓ Successfully analyzed: {len(analyzed)}/{len(variants)}")

# Statistics by group
active_variants = [v for v in analyzed if v['group'] == 'active']
allosteric_variants = [v for v in analyzed if v['group'] == 'allosteric']

print("\n" + "="*70)
print("STABILITY COMPARISON")
print("="*70)

active_stability = [v['stability_score'] for v in active_variants]
allosteric_stability = [v['stability_score'] for v in allosteric_variants]

print(f"\nActive-site group (n={len(active_stability)}):")
print(f"  Mean stability: {np.mean(active_stability):.3f} ± {np.std(active_stability):.3f}")
print(f"  Best: {max(active_stability):.3f}")

print(f"\nAllosteric group (n={len(allosteric_stability)}):")
print(f"  Mean stability: {np.mean(allosteric_stability):.3f} ± {np.std(allosteric_stability):.3f}")
print(f"  Best: {max(allosteric_stability):.3f}")

# Combined score (affinity + stability)
print("\n" + "="*70)
print("COMBINED SCORE (Affinity + Stability)")
print("="*70)

for v in analyzed:
    # Normalize both scores to 0-1 range
    affinity_norm = (v['score'] - 4.0) / 5.0  # Assume 4-9 range
    stability_norm = (v['stability_score'] + 10) / 20  # Rough normalization
    
    # Combined score (weighted)
    v['combined_score'] = 0.7 * affinity_norm + 0.3 * stability_norm

# Re-sort by combined score
active_variants = sorted(active_variants, key=lambda x: x['combined_score'], reverse=True)
allosteric_variants = sorted(allosteric_variants, key=lambda x: x['combined_score'], reverse=True)

print("\nTop 5 Active-site variants (by combined score):")
for i, v in enumerate(active_variants[:5]):
    print(f"  {i+1}. {v['id']}: Affinity={v['score']:.2f}, Stability={v['stability_score']:.2f}, Combined={v['combined_score']:.3f}")

print("\nTop 5 Allosteric variants (by combined score):")
for i, v in enumerate(allosteric_variants[:5]):
    print(f"  {i+1}. {v['id']}: Affinity={v['score']:.2f}, Stability={v['stability_score']:.2f}, Combined={v['combined_score']:.3f}")

# Save results
results = {
    'variants': analyzed,
    'summary': {
        'active_site': {
            'n': len(active_variants),
            'mean_stability': float(np.mean(active_stability)),
            'mean_affinity': float(np.mean([v['score'] for v in active_variants])),
            'top_variant': active_variants[0]['id']
        },
        'allosteric': {
            'n': len(allosteric_variants),
            'mean_stability': float(np.mean(allosteric_stability)),
            'mean_affinity': float(np.mean([v['score'] for v in allosteric_variants])),
            'top_variant': allosteric_variants[0]['id']
        }
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved to: {OUTPUT_FILE}")

# Select top 5 for MD
print("\n" + "="*70)
print("TOP CANDIDATES FOR MD SIMULATION")
print("="*70)

top_5 = (
    active_variants[:3] +  # Top 3 active
    allosteric_variants[:3]  # Top 3 allosteric (total 6, we'll run 5-6)
)

print("\nSelected for MD validation:")
for v in top_5[:5]:
    print(f"  {v['group']}_{v['id']}: Combined={v['combined_score']:.3f}")

md_selection = {
    'variants': top_5[:5],
    'selection_criteria': 'Top by combined affinity + stability score'
}

md_file = Path("results/search/md_selection.json")
with open(md_file, 'w') as f:
    json.dump(md_selection, f, indent=2)

print(f"\n✓ MD selection saved to: {md_file}")

print("\n" + "="*70)
print("STRUCTURE ANALYSIS COMPLETE")
print("="*70)
print("\n✓ Ready for MD simulations!")