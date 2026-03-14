#!/bin/bash

# Setup environment for Athena GPU Training
module load Python/3.10.12

# Create venv in scratch directory to avoid space issues in home
VENV_DIR="$SCRATCH/venvs/hackathon_task_3"
mkdir -p "$VENV_DIR"

if [ ! -d "$VENV_DIR/bin" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Run 'source $VENV_DIR/bin/activate' to use."
