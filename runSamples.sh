#!/bin/bash

FILES=(
    data/instanciac5.mcgb
    data/instanciac10.mcgb
    data/instanciac15.mcgb
    data/instanciac20.mcgb
    data/instanciac25.mcgb
)

for file in "${FILES[@]}"; do
    echo "Running $file"
    echo "Running mlp with containers"
    pixi run python3 mlp_with_containers.py "$file"

    echo "Running mlp"
    pixi run python3 mlp.py "$file"
done