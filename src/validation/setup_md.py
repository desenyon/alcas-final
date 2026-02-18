"""
Setup GPU-accelerated MD simulations for top variants
OpenCL-compatible (no bonded GPU flag)
"""

import json
import subprocess
from pathlib import Path
import shutil

# Config
MD_SELECTION = Path("results/search/md_selection.json")
OUTPUT_DIR = Path("results/validation/md")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("MD SIMULATION SETUP (GPU-ACCELERATED)")
print("="*70)

# Load selection
with open(MD_SELECTION) as f:
    selection = json.load(f)

variants = selection['variants']
print(f"\nPreparing MD for {len(variants)} variants")

# GROMACS parameters
MD_CONFIG = {
    'force_field': 'amber99sb-ildn',
    'water_model': 'tip3p',
    'box_distance': 1.0,  # nm
    'ion_concentration': 0.15,  # M
    'temperature': 300,  # K
    'pressure': 1.0,  # bar
    'production_time': 50.0,  # ns
}

print("\nMD Parameters:")
for key, value in MD_CONFIG.items():
    print(f"  {key}: {value}")

# Create MDP files
def create_mdp_files(variant_dir):
    """Create GROMACS parameter files"""
    
    # Energy minimization
    em_mdp = variant_dir / "em.mdp"
    with open(em_mdp, 'w') as f:
        f.write("""
; Energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = cutoff
rvdw        = 1.2
pbc         = xyz
""")
    
    # NVT equilibration
    nvt_mdp = variant_dir / "nvt.mdp"
    with open(nvt_mdp, 'w') as f:
        f.write(f"""
; NVT equilibration
define      = -DPOSRES
integrator  = md
nsteps      = 500000
dt          = 0.002
nstlog      = 5000
nstenergy   = 5000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = cutoff
rvdw        = 1.2
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = {MD_CONFIG['temperature']} {MD_CONFIG['temperature']}
pcoupl      = no
pbc         = xyz
gen_vel     = yes
gen_temp    = {MD_CONFIG['temperature']}
""")
    
    # NPT equilibration
    npt_mdp = variant_dir / "npt.mdp"
    with open(npt_mdp, 'w') as f:
        f.write(f"""
; NPT equilibration
define      = -DPOSRES
integrator  = md
nsteps      = 500000
dt          = 0.002
nstlog      = 5000
nstenergy   = 5000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = cutoff
rvdw        = 1.2
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = {MD_CONFIG['temperature']} {MD_CONFIG['temperature']}
pcoupl      = Parrinello-Rahman
tau_p       = 2.0
ref_p       = {MD_CONFIG['pressure']}
compressibility = 4.5e-5
pbc         = xyz
""")
    
    # Production MD
    md_mdp = variant_dir / "md.mdp"
    with open(md_mdp, 'w') as f:
        f.write(f"""
; Production MD
integrator  = md
nsteps      = 25000000
dt          = 0.002
nstxout-compressed = 5000
nstlog      = 5000
nstenergy   = 5000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = cutoff
rvdw        = 1.2
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = {MD_CONFIG['temperature']} {MD_CONFIG['temperature']}
pcoupl      = Parrinello-Rahman
tau_p       = 2.0
ref_p       = {MD_CONFIG['pressure']}
compressibility = 4.5e-5
pbc         = xyz
""")

# Create GPU-enabled run script (OpenCL compatible - no bonded GPU)
def create_run_script(variant_dir, variant_id):
    """Create bash script with GPU flags compatible with OpenCL build"""
    
    script = variant_dir / "run_md_gpu.sh"
    with open(script, 'w') as f:
        f.write(f"""#!/bin/bash
# GPU-Accelerated MD for {variant_id}
# OpenCL build: nb, pme, update on GPU; bonded on CPU

set -e

echo "=========================================="
echo "Starting MD for {variant_id}"
echo "Time: $(date)"
echo "=========================================="

# 1. Process structure
echo "[1/8] Processing structure..."
gmx pdb2gmx -f input.pdb -o processed.gro -water tip3p -ff amber99sb-ildn -ignh

# 2. Create box
echo "[2/8] Creating box..."
gmx editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt cubic

# 3. Solvate
echo "[3/8] Solvating..."
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top

# 4. Add ions
echo "[4/8] Adding ions..."
gmx grompp -f em.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 1
echo "SOL" | gmx genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15

# 5. Energy minimization (CPU - steep integrator)
echo "[5/8] Energy minimization..."
gmx grompp -f em.mdp -c ionized.gro -p topol.top -o em.tpr -maxwarn 1
gmx mdrun -v -deffnm em

# 6. NVT equilibration (GPU: nb, pme, update)
echo "[6/8] NVT equilibration (GPU)..."
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 1
gmx mdrun -v -deffnm nvt -nb gpu -pme gpu -update gpu

# 7. NPT equilibration (GPU: nb, pme, update)
echo "[7/8] NPT equilibration (GPU)..."
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt -nb gpu -pme gpu -update gpu

# 8. Production MD (GPU: nb, pme, update)
echo "[8/8] Production MD - 50ns (GPU)..."
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 1
gmx mdrun -v -deffnm md -nb gpu -pme gpu -update gpu

echo "=========================================="
echo "✓ MD COMPLETE for {variant_id}"
echo "Time: $(date)"
echo "=========================================="
""")
    
    script.chmod(0o755)

# Setup each variant
print("\nSetting up MD directories...")

for variant in variants:
    variant_id = f"{variant['group']}_{variant['id']}"
    variant_dir = OUTPUT_DIR / variant_id
    variant_dir.mkdir(exist_ok=True)
    
    print(f"\n{variant_id}:")
    
    # Copy structure
    src_pdb = Path(variant['pdb_file'])
    dst_pdb = variant_dir / "input.pdb"
    shutil.copy(src_pdb, dst_pdb)
    print(f"  ✓ Copied structure")
    
    # Create MDP files
    create_mdp_files(variant_dir)
    print(f"  ✓ Created MDP files")
    
    # Create run script
    create_run_script(variant_dir, variant_id)
    print(f"  ✓ Created GPU run script")

# Create launch script
print("\nCreating launch script...")

launch_script = OUTPUT_DIR / "launch_all.sh"
with open(launch_script, 'w') as f:
    f.write("""#!/bin/bash
# Launch all GPU-accelerated MD simulations

cd $(dirname $0)

echo "=========================================="
echo "Launching MD Simulations"
echo "Time: $(date)"
echo "=========================================="
echo ""

""")
    
    for variant in variants:
        variant_id = f"{variant['group']}_{variant['id']}"
        f.write(f"""
# Launch {variant_id}
cd {variant_id}
nohup bash run_md_gpu.sh > md.log 2>&1 &
PID=$!
echo "Started {variant_id} (PID: $PID)"
cd ..
""")
    
    f.write("""
echo ""
echo "=========================================="
echo "All simulations launched!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  tail -f */md.log"
echo "  watch -n 10 nvidia-smi"
echo ""
echo "Check completion:"
echo "  bash monitor.sh"
""")

launch_script.chmod(0o755)

# Create monitoring script
monitor_script = OUTPUT_DIR / "monitor.sh"
with open(monitor_script, 'w') as f:
    f.write("""#!/bin/bash
# Monitor MD simulation progress

cd $(dirname $0)

echo "=========================================="
echo "MD Simulation Status"
echo "Time: $(date)"
echo "=========================================="
echo ""

for dir in active_* allosteric_*; do
    if [ -d "$dir" ]; then
        echo "[$dir]"
        if [ -f "$dir/md.log" ]; then
            # Get last step info
            tail -10 "$dir/md.log" | grep -E "Step|Performance|will finish" | tail -3 || echo "  Running..."
        else
            echo "  Not started"
        fi
        echo ""
    fi
done

echo "=========================================="
echo "GPU Status:"
nvidia-smi | grep -A 2 "GPU.*Util"
""")

monitor_script.chmod(0o755)

print(f"\n✓ Created: {launch_script}")
print(f"✓ Created: {monitor_script}")

print("\n" + "="*70)
print("MD SETUP COMPLETE")
print("="*70)
print(f"\nDirectory: {OUTPUT_DIR}")
print(f"\nTo launch all simulations:")
print(f"  cd {OUTPUT_DIR}")
print(f"  bash launch_all.sh")
print(f"\nTo monitor:")
print(f"  bash monitor.sh")
print(f"\nEstimated time: 6-8 hours per simulation (with GPU)")
print("="*70)