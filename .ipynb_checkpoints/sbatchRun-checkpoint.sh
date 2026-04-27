#!/bin/bash
#SBATCH --job-name=mlp_batch
#SBATCH --time=04:15:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --output=out_%A_%a.txt
#SBATCH --constraint=chip_type_9654,interconnect_hdr

# Load modules if needed
# module load cuda
module load gurobi

FILES=(
    data/instanciac5.mcgb
    data/instanciac10.mcgb
    data/instanciac15.mcgb
    data/instanciac20.mcgb
    data/instanciac25.mcgb
)

FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "Running file: $FILE on $SLURM_JOB_NODELIST"

cd /home/aniemcz/mlp_optim

echo "Running mlp"
pixi run python3 mlp.py "$FILE"

echo "Running mlp with containers"
pixi run python3 mlp_with_containers.py "$FILE"