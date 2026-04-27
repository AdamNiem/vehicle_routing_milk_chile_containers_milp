#!/bin/bash
#SBATCH --job-name=mlp_batch_max
#SBATCH --time=04:15:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-19
#SBATCH --output=out_max_%A_%a.txt
#SBATCH --constraint=chip_type_9654,interconnect_hdr

# Load modules if needed
# module load cuda
module load gurobi

FILES=(
    data/instanciac1.mcgb
    data/instanciac2.mcgb
    data/instanciac3.mcgb
    data/instanciac4.mcgb
    
    data/instanciac6.mcgb
    data/instanciac7.mcgb
    data/instanciac8.mcgb
    data/instanciac9.mcgb
    
    data/instanciac11.mcgb
    data/instanciac12.mcgb
    data/instanciac13.mcgb
    data/instanciac14.mcgb
    
    data/instanciac16.mcgb
    data/instanciac17.mcgb
    data/instanciac18.mcgb
    data/instanciac19.mcgb
    
    data/instanciac21.mcgb
    data/instanciac22.mcgb
    data/instanciac23.mcgb
    data/instanciac24.mcgb
)

FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "Running file: $FILE on $SLURM_JOB_NODELIST"

cd /home/aniemcz/mlp_optim

echo "Running mlp"
pixi run python3 mlp.py "$FILE"

echo "Running mlp with containers"
pixi run python3 mlp_with_containers.py "$FILE"