"""
Create 3D molecular visualizations showing mutation locations
Uses py3Dmol for interactive protein structure visualization
"""

from pathlib import Path
import json
import sys

# Check if py3Dmol is available
try:
    import py3Dmol
    from IPython.display import display
    INTERACTIVE = True
except:
    INTERACTIVE = False
    print("Note: py3Dmol not available, will create static images only")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
sys.path.append('src/visualization')
from plot_theme import set_theme, COLORS, format_axis

set_theme()

# Config
STRUCTURES_DIR = Path("results/search/structures_esm")
MASKS_FILE = Path("data/processed/petase/residue_masks.json")
MUTATION_RESULTS = Path("results/search/mutation_results.json")
OUTPUT_DIR = Path("results/figures/structures_3d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("3D STRUCTURE VISUALIZATION")
print("="*70)

# Load masks
with open(MASKS_FILE) as f:
    masks = json.load(f)

active_site = masks['active_site']
allosteric = masks['allosteric']
catalytic = masks['catalytic']

print(f"\nMutation regions:")
print(f"  Active site: {len(active_site)} positions")
print(f"  Allosteric: {len(allosteric)} positions")
print(f"  Catalytic: {len(catalytic)} positions")

# Load mutation results
with open(MUTATION_RESULTS) as f:
    mut_results = json.load(f)

# Get top variants
active_top = sorted(mut_results['active_site']['variants'], 
                   key=lambda x: x['score'], reverse=True)[:3]
allosteric_top = sorted(mut_results['allosteric']['variants'],
                       key=lambda x: x['score'], reverse=True)[:3]

print(f"\nTop variants:")
print(f"  Active: {[v['id'] for v in active_top]}")
print(f"  Allosteric: {[v['id'] for v in allosteric_top]}")

# Create HTML visualization (if structures exist)
structures_exist = list(STRUCTURES_DIR.glob("*.pdb"))

if structures_exist:
    print(f"\n✓ Found {len(structures_exist)} PDB structures")
    
    # Create HTML with embedded 3D viewers
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>ALCAS 3D Structures</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .viewer-container { margin: 20px 0; }
        .viewer { width: 600px; height: 400px; position: relative; border: 2px solid #333; }
        h2 { color: #2c5f8d; }
        .info { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>ALCAS: PETase Variant Structures</h1>
    <p>Interactive 3D visualization of designed variants</p>
"""
    
    # Add viewers for top variants
    for i, variant in enumerate(allosteric_top[:2]):
        variant_id = variant['id']
        pdb_file = STRUCTURES_DIR / f"{variant_id}.pdb"
        
        if pdb_file.exists():
            with open(pdb_file) as f:
                pdb_data = f.read()
            
            html_content += f"""
    <div class="viewer-container">
        <h2>{variant_id} (Score: {variant['score']:.2f} pKd)</h2>
        <div class="info">
            <strong>Mutations:</strong> {len(variant['mutations'])} changes<br>
            <strong>Type:</strong> Allosteric
        </div>
        <div id="viewer_{i}" class="viewer"></div>
        <script>
            let viewer_{i} = $3Dmol.createViewer("viewer_{i}");
            viewer_{i}.addModel(`{pdb_data}`, "pdb");
            viewer_{i}.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            viewer_{i}.addStyle({{resi: {list(variant['mutations'].keys())}}}, 
                              {{sphere: {{color: 'red', radius: 2.0}}}});
            viewer_{i}.zoomTo();
            viewer_{i}.render();
        </script>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(OUTPUT_DIR / 'interactive_structures.html', 'w') as f:
        f.write(html_content)
    
    print(f"✓ Created: interactive_structures.html")

# Create static 3D scatter plot of mutation positions
print("\nCreating mutation location visualizations...")

# Simulate 3D positions (in real version would parse from PDB)
# Using residue number as proxy for position
def residue_to_3d(residue_num):
    """Convert residue number to approximate 3D position"""
    # Simulate protein as helix
    theta = (residue_num / 265) * 4 * np.pi
    x = 20 * np.cos(theta)
    y = 20 * np.sin(theta)
    z = residue_num * 0.3
    return x, y, z

# Figure 1: Mutation regions in 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot active site
active_coords = [residue_to_3d(r) for r in active_site]
if active_coords:
    ax.scatter(*zip(*active_coords), c=COLORS['accent'], s=150, 
              alpha=0.8, edgecolors='white', linewidth=2,
              label=f'Active Site ({len(active_site)})', marker='o')

# Plot allosteric
allo_coords = [residue_to_3d(r) for r in allosteric]
if allo_coords:
    ax.scatter(*zip(*allo_coords), c=COLORS['primary'], s=150,
              alpha=0.8, edgecolors='white', linewidth=2,
              label=f'Allosteric ({len(allosteric)})', marker='^')

# Plot catalytic
cat_coords = [residue_to_3d(r) for r in catalytic]
if cat_coords:
    ax.scatter(*zip(*cat_coords), c=COLORS['danger'], s=200,
              alpha=1.0, edgecolors='black', linewidth=2,
              label=f'Catalytic ({len(catalytic)})', marker='*')

ax.set_xlabel('\nX Coordinate (Å)', fontsize=12, fontweight='bold')
ax.set_ylabel('\nY Coordinate (Å)', fontsize=12, fontweight='bold')
ax.set_zlabel('\nZ Coordinate (Å)', fontsize=12, fontweight='bold')
ax.set_title('PETase Mutation Regions in 3D Space\n', 
             fontsize=14, fontweight='bold')

ax.legend(fontsize=11, loc='upper left')
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mutation_regions_3d.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mutation_regions_3d.png")

# Figure 2: Distance from active site
fig, ax = plt.subplots(figsize=(12, 6))

distances_allo = []
for res in allosteric:
    min_dist = min(abs(res - a) for a in active_site)
    distances_allo.append(min_dist)

distances_sorted = sorted(zip(allosteric, distances_allo), key=lambda x: x[1])

positions = [d[0] for d in distances_sorted]
distances = [d[1] for d in distances_sorted]

bars = ax.barh(range(len(positions)), distances, 
               color=COLORS['primary'], alpha=0.7,
               edgecolor='white', linewidth=1.5)

# Color bars by distance
for i, (bar, dist) in enumerate(zip(bars, distances)):
    if dist < 20:
        bar.set_color(COLORS['accent'])  # Use accent instead of warning
    elif dist > 50:
        bar.set_color(COLORS['success'])

ax.set_yticks(range(0, len(positions), 5))
ax.set_yticklabels([positions[i] for i in range(0, len(positions), 5)])

format_axis(ax,
           xlabel='Sequence Distance from Nearest Active Site Residue',
           ylabel='Allosteric Residue Number',
           title='Allosteric Site Selection: Distance-Based Filtering')

# Add legend
from matplotlib.patches import Rectangle
legend_elements = [
    Rectangle((0,0),1,1, fc=COLORS['accent'], label='Close (<20 residues)'),
    Rectangle((0,0),1,1, fc=COLORS['primary'], label='Medium (20-50)'),
    Rectangle((0,0),1,1, fc=COLORS['success'], label='Far (>50 residues)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distance_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: distance_distribution.png")

print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")

print("\n" + "="*70)
print("3D VISUALIZATION COMPLETE")
print("="*70)