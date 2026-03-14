#!/bin/bash -l
#SBATCH -J split_csv
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --time=02:00:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=16GB
#SBATCH --output=split_csv_%J.out

# Load the newest available Python
module add GCCcore/13.2.0 Python/3.11.5

# Activate virtual environment (assuming it's created as per Athena docs)
source $SCRATCH/venvs/hackathon_env/bin/activate

# Install pandas if not already installed (usually done once)
# pip --no-cache-dir --require-virtualenv install pandas

# Run the python script
# Assuming data is in data/data.csv and script is in current directory
echo "Starting data split..."
python3 split_data.py --input data/data.csv --output_dir data/split --chunksize 500000
echo "Done."
