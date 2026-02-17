"""
Download and setup PETase structures
"""

from pathlib import Path
import urllib.request

RAW_PETASE_DIR = Path("data/raw/petase")
RAW_PETASE_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PETASE STRUCTURE SETUP")
print("="*70)

# PETase structures
structures = {
    '6EQE': 'Holo (with MHET ligand)',
    '6EQD': 'Apo (no ligand)'
}

print("\nDownloading PETase structures from RCSB PDB...")

for pdb_id, description in structures.items():
    output_file = RAW_PETASE_DIR / f"{pdb_id}.pdb"
    
    if output_file.exists():
        print(f"  ✓ {pdb_id} ({description}) - already exists")
    else:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"  Downloading {pdb_id} ({description})...")
        
        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"    ✓ Saved to {output_file}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

# Verify files
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

for pdb_id, description in structures.items():
    output_file = RAW_PETASE_DIR / f"{pdb_id}.pdb"
    
    if output_file.exists():
        file_size = output_file.stat().st_size / 1024
        print(f"✓ {pdb_id}: {file_size:.1f} KB - {description}")
    else:
        print(f"✗ {pdb_id}: Missing")

print("\n" + "="*70)
print("PETASE SETUP COMPLETE")
print("="*70)