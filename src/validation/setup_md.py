"""
Setup MD simulations for top variants
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
print("MD SIMULATION SETUP")
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
    'box_type': 'cubic',
    'box_distance': 1.0,  # nm from protein
    'ion_concentration': 0.15,  # M (physiological)
    'temperature': 300,  # K
    'pressure': 1.0,  # bar
    'equilibration_time': 1.0,  # ns
    'production_time': 50.0,  # ns
    'timestep': 0.002,  # ps (2 fs)
}

print("\nMD Parameters:")
for key, value in MD_CONFIG.items():
    print(f"  {key}: {value}")

# Create MDP files (GROMACS parameter files)
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
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2
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
nsteps      = 500000  ; 1 ns
dt          = 0.002
nstxout     = 5000
nstvout     = 5000
nstlog      = 5000
nstcalcenergy = 100
nstenergy   = 1000
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rlist       = 1.2
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
gen_seed    = -1
""")
    
    # NPT equilibration
    npt_mdp = variant_dir / "npt.mdp"
    with open(npt_mdp, 'w') as f:
        f.write(f"""
; NPT equilibration
define      = -DPOSRES
integrator  = md
nsteps      = 500000  ; 1 ns
dt          = 0.002
nstxout     = 5000
nstvout     = 5000
nstlog      = 5000
nstcalcenergy = 100
nstenergy   = 1000
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = cutoff
rvdw        = 1.2
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = {MD_CONFIG['temperature']} {MD_CONFIG['temperature']}
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
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
nsteps      = 25000000  ; 50 ns
dt          = 0.002
nstxout     = 0
nstvout     = 0
nstfout     = 0
nstxout-compressed = 5000  ; Save every 10 ps
compressed-x-grps = Protein
nstlog      = 5000
nstcalcenergy = 100
nstenergy   = 5000
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = cutoff
rvdw        = 1.2
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = {MD_CONFIG['temperature']} {MD_CONFIG['temperature']}
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = {MD_CONFIG['pressure']}
compressibility = 4.5e-5
pbc         = xyz
""")

# Create run script
def create_run_script(variant_dir, variant_id):
    """Create bash script to run MD"""
    
    script = variant_dir / "run_md.sh"
    with open(script, 'w') as f:
        f.write(f"""#!/bin/bash
# MD simulation for {variant_id}

set -e

echo "Starting MD for {variant_id}"

# 1. Prepare system
echo "Step 1: Processing structure..."
gmx pdb2gmx -f input.pdb -o processed.gro -water tip3p -ff amber99sb-ildn -ignh

# 2. Define box
echo "Step 2: Defining box..."
gmx editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt cubic

# 3. Solvate
echo "Step 3: Solvating..."
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top

# 4. Add ions
echo "Step 4: Adding ions..."
gmx grompp -f em.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 1
echo "SOL" | gmx genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15

# 5. Energy minimization
echo "Step 5: Energy minimization..."
gmx grompp -f em.mdp -c ionized.gro -p topol.top -o em.tpr -maxwarn 1
gmx mdrun -v -deffnm em

# 6. NVT equilibration
echo "Step 6: NVT equilibration..."
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 1
gmx mdrun -v -deffnm nvt

# 7. NPT equilibration
echo "Step 7: NPT equilibration..."
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt

# 8. Production MD
echo "Step 8: Production MD (50 ns)..."
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 1
gmx mdrun -v -deffnm md -nb gpu

echo "MD complete for {variant_id}!"
""")
    
    # Make executable
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
    print(f"  ✓ Created run script")

# Create master launch script
print("\nCreating master launch script...")

launch_script = OUTPUT_DIR / "launch_all.sh"
with open(launch_script, 'w') as f:
    f.write("""#!/bin/bash
# Launch all MD simulations in parallel

echo "Launching MD simulations..."

""")
    
    for variant in variants:
        variant_id = f"{variant['group']}_{variant['id']}"
        f.write(f"""
cd {variant_id}
nohup bash run_md.sh > md.log 2>&1 &
cd ..
""")
    
    f.write("""
echo "All simulations launched!"
echo "Monitor progress: tail -f */md.log"
echo "Check status: ps aux | grep mdrun"
""")

launch_script.chmod(0o755)

print(f"✓ Created: {launch_script}")

# Create monitoring script
monitor_script = OUTPUT_DIR / "monitor.sh"
with open(monitor_script, 'w') as f:
    f.write("""#!/bin/bash
# Monitor MD progress

echo "MD Simulation Status"
echo "===================="
for dir in active_* allosteric_*; do
    if [ -f "$dir/md.log" ]; then
        echo "$dir:"
        tail -n 3 "$dir/md.log" | grep -E "Step|Time|Performance" || echo "  Starting..."
    fi
done
""")

monitor_script.chmod(0o755)

print(f"✓ Created: {monitor_script}")

print("\n" + "="*70)
print("MD SETUP COMPLETE")
print("="*70)
print(f"\nTo launch all simulations:")
print(f"  cd {OUTPUT_DIR}")
print(f"  bash launch_all.sh")
print(f"\nTo monitor progress:")
print(f"  bash monitor.sh")
print(f"\nEstimated time: 12-18 hours (running in parallel)")
print("="*70)