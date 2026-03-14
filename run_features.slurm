#!/bin/bash -l
#SBATCH -J feature_eng
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --time=04:00:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=32GB
#SBATCH --output=feature_eng_%J.out

# Load the newest available Python
module add GCCcore/13.2.0 Python/3.11.5

# Activate virtual environment
source $SCRATCH/venvs/hackathon_env/bin/activate

# Install pandas and numpy if not already installed
# pip --no-cache-dir --require-virtualenv install pandas numpy

# Run the python script
echo "Starting feature engineering..."
python3 feature_engineering.py --input_dir data/split --output_dir data/devices
echo "Done."
