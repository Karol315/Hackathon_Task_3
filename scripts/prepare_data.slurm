#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --partition=tutorial
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_%j.log

# Load modules and activate environment
module load Python/3.10.12
source "$SCRATCH/venvs/hackathon_task_3/bin/activate"

mkdir -p logs

echo "Starting data preparation on compute node..."
python scripts/prepare_data.py
echo "Data preparation finished."
