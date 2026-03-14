#!/bin/bash
#SBATCH --job-name=train_forecasting
#SBATCH --partition=tutorial
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log

# Load modules and activate environment
# Load modules and activate environment
module add GCCcore/13.2.0 Python/3.11.5
source "$SCRATCH/venvs/hackathon_task_3/bin/activate"

# Ensure output directory exists for models
mkdir -p models
mkdir -p logs

# Run training
# To change model, edit the --model argument below (lstm or transformer)
python train.py --model transformer --epochs 50 --batch_size 256
