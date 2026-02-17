"""
Fold top variants using ESM Atlas API
"""

import json
import requests
import time
from pathlib import Path
from Bio.PDB import PDBParser

# Config
RESULTS_FILE = Path("results/search/mutation_results.json")
PETASE_WT = Path("data/raw/petase/6EQE.pdb")
OUTPUT_DIR = Path("results/search/structures_esm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ESM_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

print("="*70)
print("FOLDING TOP VARIANTS WITH ESMFOLD")
print("="*70)

# Load results
with open(RESULTS_FILE) as f:
    results = json.load(f)

# Get top 10 from each group
active_top = sorted(results['active_site']['variants'], 
                   key=lambda x: x['score'], reverse=True)[:10]
allosteric_top = sorted(results['allosteric']['variants'], 
                       key=lambda x: x['score'], reverse=True)[:10]

print(f"\nSelected top candidates:")
print(f"  Active-site: {len(active_top)} variants")
print(f"  Allosteric: {len(allosteric_top)} variants")

# Get WT sequence
print("\nExtracting WT sequence...")
parser = PDBParser(QUIET=True)
structure = parser.get_structure('wt', PETASE_WT)

AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

wt_sequence_list = []
residue_map = {}

for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] == ' ':
                pos = residue.id[1]
                resname = residue.resname
                if resname in AA_MAP:
                    aa = AA_MAP[resname]
                    wt_sequence_list.append((pos, aa))
                    residue_map[pos] = aa

wt_sequence = ''.join([aa for _, aa in wt_sequence_list])
print(f"✓ WT sequence: {len(wt_sequence)} residues")

# Apply mutations
def apply_mutations(wt_seq, residue_map, mutations):
    """Apply mutations to get mutant sequence"""
    seq_list = list(wt_seq)
    positions = [p for p, _ in wt_sequence_list]
    
    for mut_pos, mut_aa in mutations.items():
        if mut_pos in positions:
            idx = positions.index(mut_pos)
            seq_list[idx] = mut_aa
    
    return ''.join(seq_list)

# Prepare all variants
all_variants = []

for v in active_top:
    v['sequence'] = apply_mutations(wt_sequence, residue_map, v['mutations'])
    v['group'] = 'active'
    all_variants.append(v)

for v in allosteric_top:
    v['sequence'] = apply_mutations(wt_sequence, residue_map, v['mutations'])
    v['group'] = 'allosteric'
    all_variants.append(v)

print(f"\nTotal variants to fold: {len(all_variants)}")

# Fold with ESM API
print("\n" + "="*70)
print("SUBMITTING TO ESM ATLAS API")
print("="*70)
print("NOTE: This uses Meta's free API - rate limited to ~1 per minute")
print("Estimated time: ~20 minutes for 20 structures")

folded_variants = []

for i, variant in enumerate(all_variants):
    print(f"\n[{i+1}/{len(all_variants)}] Folding {variant['group']}_{variant['id']}...")
    print(f"  Mutations: {variant['mutations']}")
    
    # Call API
    try:
        response = requests.post(
            ESM_API_URL,
            data=variant['sequence'],
            headers={'Content-Type': 'text/plain'},
            timeout=300  # 5 min timeout
        )
        
        if response.status_code == 200:
            # Save PDB
            pdb_content = response.text
            pdb_file = OUTPUT_DIR / f"{variant['group']}_{variant['id']}.pdb"
            
            with open(pdb_file, 'w') as f:
                f.write(pdb_content)
            
            variant['pdb_file'] = str(pdb_file)
            variant['folded'] = True
            folded_variants.append(variant)
            
            print(f"  ✓ Structure predicted and saved")
            
        elif response.status_code == 429:
            print(f"  ⚠ Rate limited - waiting 60s...")
            time.sleep(60)
            # Retry
            response = requests.post(ESM_API_URL, data=variant['sequence'], 
                                   headers={'Content-Type': 'text/plain'}, timeout=300)
            if response.status_code == 200:
                pdb_content = response.text
                pdb_file = OUTPUT_DIR / f"{variant['group']}_{variant['id']}.pdb"
                with open(pdb_file, 'w') as f:
                    f.write(pdb_content)
                variant['pdb_file'] = str(pdb_file)
                variant['folded'] = True
                folded_variants.append(variant)
                print(f"  ✓ Retry successful")
            else:
                print(f"  ✗ Failed after retry: {response.status_code}")
                variant['folded'] = False
        else:
            print(f"  ✗ Failed: HTTP {response.status_code}")
            variant['folded'] = False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        variant['folded'] = False
    
    # Rate limiting - wait between requests
    if i < len(all_variants) - 1:
        print(f"  Waiting 65s before next request...")
        time.sleep(65)

# Summary
print("\n" + "="*70)
print("FOLDING COMPLETE")
print("="*70)

successful = [v for v in all_variants if v.get('folded', False)]
print(f"\nSuccessfully folded: {len(successful)}/{len(all_variants)}")
print(f"  Active-site: {len([v for v in successful if v['group'] == 'active'])}")
print(f"  Allosteric: {len([v for v in successful if v['group'] == 'allosteric'])}")

# Save metadata
metadata = {
    'wild_type': {
        'sequence': wt_sequence,
        'length': len(wt_sequence)
    },
    'variants': all_variants,
    'summary': {
        'total': len(all_variants),
        'successful': len(successful),
        'failed': len(all_variants) - len(successful)
    }
}

metadata_file = OUTPUT_DIR / 'folding_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Saved metadata to: {metadata_file}")
print(f"✓ Structures saved to: {OUTPUT_DIR}")

print("\n" + "="*70)
print("READY FOR RESCORING WITH REAL STRUCTURES")
print("="*70)