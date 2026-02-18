"""
Multi-Objective Pareto Optimization
Optimizes affinity + stability + solubility simultaneously
Shows trade-offs between competing objectives
"""

import numpy as np
from pathlib import Path
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('src/visualization')
from plot_theme import set_theme, COLORS, format_axis

# Apply theme
set_theme()

# Config
MASKS_FILE = Path("data/processed/petase/residue_masks.json")
MUTATION_RESULTS = Path("results/search/mutation_results.json")
OUTPUT_DIR = Path("results/analysis/pareto")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_GENERATIONS = 50
POPULATION_SIZE = 100

random.seed(SEED)
np.random.seed(SEED)

print("="*70)
print("MULTI-OBJECTIVE PARETO OPTIMIZATION")
print("="*70)

# Load masks
with open(MASKS_FILE) as f:
    masks = json.load(f)

allosteric = masks['allosteric']

# Amino acids
AMINO_ACIDS = ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

print(f"\nAllosteric positions: {len(allosteric)}")

# Objective functions
def score_affinity(mutations):
    """Simulate affinity score (higher = better binding)"""
    score = 6.0
    for pos, aa in mutations.items():
        if aa in ['F', 'W', 'Y', 'L', 'I', 'V']:  # Hydrophobic
            score += random.uniform(0.2, 0.5)
        elif aa in ['D', 'E']:  # Acidic
            score += random.uniform(-0.1, 0.3)
        elif aa in ['K', 'R']:  # Basic
            score += random.uniform(-0.2, 0.2)
        else:
            score += random.uniform(-0.1, 0.2)
    
    score += random.gauss(0, 0.3)
    return max(0, score)

def score_stability(mutations):
    """Simulate stability score (higher = more stable)"""
    score = 50.0
    for pos, aa in mutations.items():
        if aa in ['P', 'G']:  # Proline/Glycine affect stability
            score += random.uniform(-2, 1)
        elif aa in ['C']:  # Cysteine (disulfide potential)
            score += random.uniform(1, 3)
        elif aa in ['A', 'V', 'L', 'I']:  # Small/hydrophobic
            score += random.uniform(0, 2)
        else:
            score += random.uniform(-1, 1)
    
    score += random.gauss(0, 2)
    return max(0, score)

def score_solubility(mutations):
    """Simulate solubility score (higher = more soluble)"""
    score = 40.0
    for pos, aa in mutations.items():
        if aa in ['K', 'R', 'D', 'E', 'N', 'Q']:  # Charged/polar
            score += random.uniform(1, 3)
        elif aa in ['F', 'W', 'Y', 'I', 'L', 'V']:  # Hydrophobic
            score += random.uniform(-2, 0)
        else:
            score += random.uniform(-0.5, 1)
    
    score += random.gauss(0, 2)
    return max(0, score)

# NSGA-II inspired Pareto optimization
class Individual:
    def __init__(self, mutations):
        self.mutations = mutations
        self.objectives = None
        self.rank = None
        self.crowding_distance = 0
    
    def evaluate(self):
        self.objectives = {
            'affinity': score_affinity(self.mutations),
            'stability': score_stability(self.mutations),
            'solubility': score_solubility(self.mutations)
        }
    
    def dominates(self, other):
        """Check if this individual dominates another"""
        better_in_one = False
        for obj in ['affinity', 'stability', 'solubility']:
            if self.objectives[obj] < other.objectives[obj]:
                return False
            if self.objectives[obj] > other.objectives[obj]:
                better_in_one = True
        return better_in_one

def create_individual(max_mutations=4):
    """Create random individual"""
    n_mutations = random.randint(1, max_mutations)
    positions = random.sample(allosteric, n_mutations)
    
    mutations = {}
    for pos in positions:
        mutations[pos] = random.choice(AMINO_ACIDS)
    
    ind = Individual(mutations)
    ind.evaluate()
    return ind

def fast_non_dominated_sort(population):
    """Assign Pareto ranks"""
    fronts = [[]]
    
    for p in population:
        p.domination_count = 0
        p.dominated_set = []
        
        for q in population:
            if p.dominates(q):
                p.dominated_set.append(q)
            elif q.dominates(p):
                p.domination_count += 1
        
        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)
    
    i = 0
    while i < len(fronts) and fronts[i]:  # FIX: Check bounds
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_set:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    
    return [f for f in fronts if f]  # Return only non-empty fronts

def calculate_crowding_distance(front):
    """Calculate crowding distance for diversity"""
    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return
    
    for ind in front:
        ind.crowding_distance = 0
    
    for obj in ['affinity', 'stability', 'solubility']:
        front_sorted = sorted(front, key=lambda x: x.objectives[obj])
        
        front_sorted[0].crowding_distance = float('inf')
        front_sorted[-1].crowding_distance = float('inf')
        
        obj_range = front_sorted[-1].objectives[obj] - front_sorted[0].objectives[obj]
        if obj_range == 0:
            continue
        
        for i in range(1, len(front_sorted) - 1):
            distance = (front_sorted[i+1].objectives[obj] - 
                       front_sorted[i-1].objectives[obj]) / obj_range
            front_sorted[i].crowding_distance += distance

def select_parents(population, tournament_size=2):
    """Tournament selection"""
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
    return tournament[0]

def crossover(parent1, parent2):
    """Create offspring by combining parents"""
    child_mutations = {}
    
    all_positions = set(parent1.mutations.keys()) | set(parent2.mutations.keys())
    
    for pos in all_positions:
        if random.random() < 0.5:
            if pos in parent1.mutations:
                child_mutations[pos] = parent1.mutations[pos]
        else:
            if pos in parent2.mutations:
                child_mutations[pos] = parent2.mutations[pos]
    
    return Individual(child_mutations)

def mutate(individual, mutation_rate=0.2):
    """Mutate individual"""
    if random.random() < mutation_rate:
        if individual.mutations and random.random() < 0.5:
            # Change existing mutation
            pos = random.choice(list(individual.mutations.keys()))
            individual.mutations[pos] = random.choice(AMINO_ACIDS)
        else:
            # Add new mutation
            new_pos = random.choice(allosteric)
            individual.mutations[new_pos] = random.choice(AMINO_ACIDS)
    
    individual.evaluate()

# Run optimization
print("\n" + "="*70)
print("RUNNING MULTI-OBJECTIVE OPTIMIZATION")
print("="*70)

# Initialize population
print(f"\nInitializing population (size={POPULATION_SIZE})...")
population = [create_individual() for _ in range(POPULATION_SIZE)]

# Track evolution
history = {
    'pareto_fronts': [],
    'hypervolume': []
}

print(f"\nEvolving for {NUM_GENERATIONS} generations...")

for gen in tqdm(range(NUM_GENERATIONS)):
    # Non-dominated sorting
    fronts = fast_non_dominated_sort(population)
    
    # Calculate crowding distance
    for front in fronts:
        calculate_crowding_distance(front)
    
    # Store Pareto front
    if gen % 10 == 0:
        pareto_front = [{
            'affinity': ind.objectives['affinity'],
            'stability': ind.objectives['stability'],
            'solubility': ind.objectives['solubility'],
            'mutations': len(ind.mutations)
        } for ind in fronts[0]]
        history['pareto_fronts'].append({
            'generation': gen,
            'individuals': pareto_front
        })
    
    # Create offspring
    offspring = []
    for _ in range(POPULATION_SIZE):
        parent1 = select_parents(population)
        parent2 = select_parents(population)
        
        child = crossover(parent1, parent2)
        mutate(child)
        
        offspring.append(child)
    
    # Combine and select
    population = population + offspring
    fronts = fast_non_dominated_sort(population)
    
    new_population = []
    for front in fronts:
        calculate_crowding_distance(front)
        if len(new_population) + len(front) <= POPULATION_SIZE:
            new_population.extend(front)
        else:
            front_sorted = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
            new_population.extend(front_sorted[:POPULATION_SIZE - len(new_population)])
            break
    
    population = new_population

# Final Pareto front
final_fronts = fast_non_dominated_sort(population)
pareto_front = final_fronts[0]

print(f"\n✓ Final Pareto front: {len(pareto_front)} solutions")

# Visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Figure 1: 3D Pareto front
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

affinity = [ind.objectives['affinity'] for ind in pareto_front]
stability = [ind.objectives['stability'] for ind in pareto_front]
solubility = [ind.objectives['solubility'] for ind in pareto_front]
n_mutations = [len(ind.mutations) for ind in pareto_front]

scatter = ax.scatter(affinity, stability, solubility, 
                    c=n_mutations, cmap='viridis', s=100, 
                    alpha=0.7, edgecolors='white', linewidth=1.5)

ax.set_xlabel('\nAffinity (pKd)', fontsize=12, fontweight='bold')
ax.set_ylabel('\nStability Score', fontsize=12, fontweight='bold')
ax.set_zlabel('\nSolubility Score', fontsize=12, fontweight='bold')
ax.set_title('3D Pareto Frontier: Multi-Objective Optimization\n', 
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Number of Mutations', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pareto_3d.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pareto_3d.png")

# Figure 2: 2D projections
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Affinity vs Stability
ax = axes[0]
scatter = ax.scatter(affinity, stability, c=n_mutations, cmap='viridis',
                    s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
format_axis(ax, xlabel='Affinity (pKd)', ylabel='Stability Score',
           title='Affinity vs Stability Trade-off')

# Affinity vs Solubility
ax = axes[1]
scatter = ax.scatter(affinity, solubility, c=n_mutations, cmap='viridis',
                    s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
format_axis(ax, xlabel='Affinity (pKd)', ylabel='Solubility Score',
           title='Affinity vs Solubility Trade-off')

# Stability vs Solubility
ax = axes[2]
scatter = ax.scatter(stability, solubility, c=n_mutations, cmap='viridis',
                    s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
format_axis(ax, xlabel='Stability Score', ylabel='Solubility Score',
           title='Stability vs Solubility Trade-off')

cbar = plt.colorbar(scatter, ax=axes, pad=0.02)
cbar.set_label('Mutations', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pareto_2d_projections.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pareto_2d_projections.png")

# Figure 3: Evolution over generations
fig, ax = plt.subplots(figsize=(12, 6))

for i, snapshot in enumerate(history['pareto_fronts']):
    gen = snapshot['generation']
    front_affinity = [ind['affinity'] for ind in snapshot['individuals']]
    front_stability = [ind['stability'] for ind in snapshot['individuals']]
    
    color = plt.cm.plasma(i / len(history['pareto_fronts']))
    ax.scatter(front_affinity, front_stability, 
              color=color, s=50, alpha=0.6,
              label=f'Gen {gen}')

format_axis(ax, xlabel='Affinity (pKd)', ylabel='Stability Score',
           title='Pareto Front Evolution Over Generations')
ax.legend(loc='upper right', ncol=2, fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pareto_evolution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pareto_evolution.png")

# Save results
results = {
    'final_pareto_front': [{
        'affinity': ind.objectives['affinity'],
        'stability': ind.objectives['stability'],
        'solubility': ind.objectives['solubility'],
        'mutations': dict(ind.mutations),
        'num_mutations': len(ind.mutations)
    } for ind in pareto_front],
    'optimization_params': {
        'generations': NUM_GENERATIONS,
        'population_size': POPULATION_SIZE
    },
    'statistics': {
        'pareto_size': len(pareto_front),
        'avg_affinity': float(np.mean(affinity)),
        'avg_stability': float(np.mean(stability)),
        'avg_solubility': float(np.mean(solubility))
    }
}

with open(OUTPUT_DIR / 'pareto_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {OUTPUT_DIR}")

print("\n" + "="*70)
print("PARETO OPTIMIZATION COMPLETE")
print("="*70)
print(f"\nKey Finding: Identified {len(pareto_front)} Pareto-optimal solutions")
print("showing trade-offs between affinity, stability, and solubility.")
print("No single solution dominates all others - each represents")
print("a different balance of competing objectives!")