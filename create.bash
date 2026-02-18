#!/bin/bash
# Create perfect ALCAS directory structure

echo "Creating ALCAS directory structure..."

# Data directories (processing pipeline)
mkdir -p data/processed
mkdir -p data/splits/protein_cluster
mkdir -p data/graphs/protein_cluster

# Source code (Python modules)
mkdir -p src/data
mkdir -p src/models
mkdir -p src/utils

# Results (model outputs)
mkdir -p results/models/affinity

# Scripts (bash automation)
mkdir -p scripts

# Logs (training logs)
mkdir -p logs

# Documentation
mkdir -p docs

echo "✓ Directory structure created"

# Show final structure
tree -L 3 -I '__pycache__|*.pyc' .

echo -e "\n✓ Ready for pipeline!"