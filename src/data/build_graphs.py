"""
Build PyTorch Geometric graphs from protein-ligand complexes
Creates graphs for train/val/test splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from rdkit import Chem
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Configuration
SPLITS_DIR = Path("data/splits/protein_cluster")
RAW_DIR = Path("data/raw/pdbbind")
GRAPHS_DIR = Path("data/graphs/protein_cluster")
POCKET_RADIUS = 6.0  # Angstroms

print("="*70)
print("STEP 3: BUILDING MOLECULAR GRAPHS")
print("="*70)

def get_complex_path(pdb_code, year):
    """Get path to complex folder"""
    if year <= 2000:
        year_folder = "1981-2000"
    elif year <= 2010:
        year_folder = "2001-2010"
    else:
        year_folder = "2011-2019"
    return RAW_DIR / year_folder / pdb_code

def get_atom_features(atom):
    """Extract atom features (26-dim)"""
    # Atom type (10-dim one-hot)
    atom_type = [0] * 10
    atom_symbols = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Other']
    symbol = atom.GetSymbol()
    if symbol in atom_symbols:
        atom_type[atom_symbols.index(symbol)] = 1
    else:
        atom_type[-1] = 1
    
    # Degree (5-dim one-hot, 0-4+)
    degree = [0] * 5
    d = min(atom.GetDegree(), 4)
    degree[d] = 1
    
    # Formal charge (5-dim one-hot, -2 to +2)
    charge = [0] * 5
    fc = atom.GetFormalCharge()
    fc_idx = min(max(fc + 2, 0), 4)
    charge[fc_idx] = 1
    
    # Hybridization (4-dim one-hot)
    hybrid = [0] * 4
    hybrid_types = [Chem.rdchem.HybridizationType.SP,
                   Chem.rdchem.HybridizationType.SP2,
                   Chem.rdchem.HybridizationType.SP3,
                   Chem.rdchem.HybridizationType.SP3D]
    h = atom.GetHybridization()
    if h in hybrid_types:
        hybrid[hybrid_types.index(h)] = 1
    else:
        hybrid[-1] = 1
    
    # Additional (2-dim)
    aromatic = [int(atom.GetIsAromatic())]
    in_ring = [int(atom.IsInRing())]
    
    return atom_type + degree + charge + hybrid + aromatic + in_ring

def get_bond_features(bond):
    """Extract bond features (6-dim)"""
    # Bond type (4-dim one-hot)
    bond_type = [0] * 4
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_type[0] = 1
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_type[1] = 1
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_type[2] = 1
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_type[3] = 1
    
    # Additional (2-dim)
    in_ring = [int(bond.IsInRing())]
    conjugated = [int(bond.GetIsConjugated())]
    
    return bond_type + in_ring + conjugated

def get_ligand_graph(ligand_file):
    """Create ligand graph"""
    # Load ligand
    if ligand_file.suffix == '.mol2':
        mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False)
    else:
        mol = Chem.MolFromMolFile(str(ligand_file), removeHs=False)
    
    if mol is None:
        return None
    
    # Remove hydrogens
    mol = Chem.RemoveHs(mol)
    
    if mol.GetNumAtoms() == 0:
        return None
    
    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        features = get_atom_features(atom)
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge indices and features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)
    
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # 3D coordinates
    conf = mol.GetConformer()
    pos = torch.tensor([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z]
                       for i in range(mol.GetNumAtoms())], dtype=torch.float)
    
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'pos': pos}

def radial_basis(dist, num_basis=10, cutoff=10.0):
    """RBF encoding of distances"""
    centers = np.linspace(0, cutoff, num_basis)
    width = cutoff / num_basis
    rbf = np.exp(-((dist - centers) ** 2) / (2 * width ** 2))
    return rbf.tolist()

def get_protein_pocket(protein_file, ligand_pos, pocket_radius):
    """Extract protein pocket around ligand"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_file)
    
    # Get ligand center
    ligand_center = ligand_pos.mean(dim=0).numpy()
    
    # Find pocket residues
    pocket_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                
                # Check if any atom within radius
                for atom in residue:
                    coord = atom.get_coord()
                    dist = np.linalg.norm(coord - ligand_center)
                    if dist <= pocket_radius:
                        pocket_residues.append(residue)
                        break
    
    if len(pocket_residues) == 0:
        return None
    
    # Convert to graph
    AA_MAP = {
        'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
        'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
        'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
        'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
    }
    
    residue_features = []
    residue_positions = []
    
    for res in pocket_residues:
        # One-hot residue type (20-dim)
        res_feat = [0] * 20
        resname = res.get_resname()
        if resname in AA_MAP:
            res_feat[AA_MAP[resname]] = 1
        
        residue_features.append(res_feat)
        
        # CA position
        if 'CA' in res:
            ca_pos = res['CA'].get_coord()
        else:
            ca_pos = np.mean([atom.get_coord() for atom in res], axis=0)
        residue_positions.append(ca_pos)
    
    x = torch.tensor(residue_features, dtype=torch.float)
    pos = torch.tensor(residue_positions, dtype=torch.float)
    
    # Build edges (distance-based, 10Å cutoff)
    edge_indices = []
    edge_features = []
    
    for i in range(len(pocket_residues)):
        for j in range(i+1, len(pocket_residues)):
            dist = np.linalg.norm(pos[i].numpy() - pos[j].numpy())
            if dist <= 10.0:
                rbf = radial_basis(dist, num_basis=10, cutoff=10.0)
                
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                edge_features.append(rbf)
                edge_features.append(rbf)
    
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'pos': pos}

def build_graph(row):
    """Build complete graph for one complex"""
    from types import SimpleNamespace
    
    pdb = row['pdb_code']
    year = row['year']
    paffinity = row['paffinity']
    
    complex_path = get_complex_path(pdb, year)
    protein_file = complex_path / f"{pdb}_protein.pdb"
    ligand_file = complex_path / f"{pdb}_ligand.mol2"
    
    if not ligand_file.exists():
        ligand_file = complex_path / f"{pdb}_ligand.sdf"
    
    try:
        # Build ligand graph
        ligand_graph = get_ligand_graph(ligand_file)
        if ligand_graph is None:
            return None
        
        # Build protein pocket graph
        protein_graph = get_protein_pocket(protein_file, ligand_graph['pos'], POCKET_RADIUS)
        if protein_graph is None:
            return None
        
        # Create data as SimpleNamespace (picklable)
        data = SimpleNamespace(
            ligand_x=ligand_graph['x'],
            ligand_edge_index=ligand_graph['edge_index'],
            ligand_edge_attr=ligand_graph['edge_attr'],
            ligand_pos=ligand_graph['pos'],
            
            protein_x=protein_graph['x'],
            protein_edge_index=protein_graph['edge_index'],
            protein_edge_attr=protein_graph['edge_attr'],
            protein_pos=protein_graph['pos'],
            
            y=torch.tensor([paffinity], dtype=torch.float),
            pdb_code=pdb
        )
        
        return data
        
    except Exception:
        return None

def build_split(split_name):
    """Build graphs for one split"""
    print(f"\n{'='*70}")
    print(f"Building {split_name.upper()} split")
    print(f"{'='*70}")
    
    # Load split
    split_df = pd.read_csv(SPLITS_DIR / f'{split_name}.csv')
    print(f"Complexes: {len(split_df)}")
    
    # Build graphs
    graphs = []
    failed = []
    
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Building {split_name}"):
        graph = build_graph(row)
        if graph is not None:
            graphs.append(graph)
        else:
            failed.append(row['pdb_code'])
    
    print(f"\n✓ Built: {len(graphs)} graphs")
    print(f"✗ Failed: {len(failed)} ({len(failed)/len(split_df)*100:.1f}%)")
    
    # Save
    output_file = GRAPHS_DIR / f"{split_name}.pt"
    torch.save(graphs, output_file)
    
    file_size_mb = output_file.stat().st_size / 1e6
    print(f"✓ Saved: {output_file} ({file_size_mb:.1f} MB)")
    
    return len(graphs), len(failed)

# Build all splits
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

results = {}
for split_name in ['train', 'val', 'test']:
    success, failed = build_split(split_name)
    results[split_name] = {'success': success, 'failed': failed}

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

total_success = sum(r['success'] for r in results.values())
total_failed = sum(r['failed'] for r in results.values())

for split_name, stats in results.items():
    print(f"{split_name.capitalize()}: {stats['success']} graphs ({stats['failed']} failed)")

print(f"\nTotal: {total_success} graphs ({total_failed} failed)")
print(f"Success rate: {total_success/(total_success+total_failed)*100:.1f}%")

print("\n" + "="*70)
print("STEP 3 COMPLETE")
print("="*70)