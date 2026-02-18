"""
Extract actual protein sequences from PDB files in PDBbind
Creates mapping: PDB_ID → protein sequence
"""

from pathlib import Path
from Bio.PDB import PDBParser
import pandas as pd
import json
from tqdm import tqdm
import hashlib

# Config
PDBIND_BASE = Path("data/raw/pdbbind")
METADATA_FILE = Path("data/processed/meta.csv")
OUTPUT_FILE = Path("data/processed/protein_sequences.json")

print("="*70)
print("EXTRACTING PROTEIN SEQUENCES FROM PDBIND")
print("="*70)

# Load metadata
metadata = pd.read_csv(METADATA_FILE)
print(f"\nTotal complexes in metadata: {len(metadata)}")

# Year ranges in PDBbind
year_ranges = {
    range(1981, 2001): "1981-2000",
    range(2001, 2011): "2001-2010", 
    range(2011, 2020): "2011-2019"
}

def get_pdb_path(pdb_code, year):
    """Find PDB file path based on year"""
    for year_range, folder_name in year_ranges.items():
        if year in year_range:
            pdb_dir = PDBIND_BASE / folder_name / pdb_code
            protein_file = pdb_dir / f"{pdb_code}_protein.pdb"
            if protein_file.exists():
                return protein_file
    return None

# Build file mapping
print("\nMapping PDB codes to files...")
pdb_to_file = {}
missing = []

for _, row in metadata.iterrows():
    pdb_code = row['pdb_code']
    year = int(row['year'])
    
    pdb_file = get_pdb_path(pdb_code, year)
    if pdb_file:
        pdb_to_file[pdb_code] = pdb_file
    else:
        missing.append(pdb_code)

print(f"✓ Found: {len(pdb_to_file)} protein files")
print(f"✗ Missing: {len(missing)} protein files")

# Extract sequences
sequences = {}
failed = []
parser = PDBParser(QUIET=True)

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

print("\nExtracting sequences...")

for pdb_code, pdb_file in tqdm(pdb_to_file.items()):
    try:
        structure = parser.get_structure(pdb_code, pdb_file)
        
        # Get first model, first chain
        model = structure[0]
        chains = list(model.get_chains())
        
        if len(chains) == 0:
            failed.append(pdb_code)
            continue
        
        # Extract sequence from first chain
        chain = chains[0]
        sequence = []
        
        for residue in chain:
            if residue.id[0] == ' ':  # Standard residue
                res_name = residue.resname
                if res_name in three_to_one:
                    sequence.append(three_to_one[res_name])
        
        if len(sequence) > 0:
            seq_str = ''.join(sequence)
            seq_hash = hashlib.md5(seq_str.encode()).hexdigest()[:8]
            
            sequences[pdb_code] = {
                'sequence': seq_str,
                'length': len(seq_str),
                'seq_id': seq_hash,
                'num_chains': len(chains)
            }
        else:
            failed.append(pdb_code)
    
    except Exception as e:
        failed.append(pdb_code)

print(f"\n✓ Extracted: {len(sequences)} sequences")
print(f"✗ Failed: {len(failed)} sequences")

# Find unique sequences
unique_seqs = {}
for pdb_code, data in sequences.items():
    seq_id = data['seq_id']
    if seq_id not in unique_seqs:
        unique_seqs[seq_id] = {
            'sequence': data['sequence'],
            'length': data['length'],
            'pdb_codes': []
        }
    unique_seqs[seq_id]['pdb_codes'].append(pdb_code)

print(f"\n✓ Unique sequences: {len(unique_seqs)}")

# Statistics
if unique_seqs:
    seq_lengths = [data['length'] for data in unique_seqs.values()]
    print(f"\nSequence length stats:")
    print(f"  Min: {min(seq_lengths)}")
    print(f"  Max: {max(seq_lengths)}")
    print(f"  Mean: {sum(seq_lengths)/len(seq_lengths):.1f}")

# Save
output = {
    'pdb_to_sequence': sequences,
    'unique_sequences': unique_seqs,
    'missing_files': missing,
    'failed_parsing': failed,
    'stats': {
        'total_metadata': len(metadata),
        'found_files': len(pdb_to_file),
        'extracted': len(sequences),
        'unique': len(unique_seqs)
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved: {OUTPUT_FILE}")
print("\n" + "="*70)
print("COMPLETE - Next: Extract ESM embeddings")
print("="*70)