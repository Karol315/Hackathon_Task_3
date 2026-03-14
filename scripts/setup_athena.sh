#!/bin/bash

# Setup environment for Athena GPU Training
module add GCCcore/13.2.0 Python/3.11.5

# Ensure we are in $SCRATCH
if [[ "$PWD" != "$SCRATCH"* ]]; then
    echo "WARNING: You are NOT in your \$SCRATCH directory."
    echo "Athena requires working in \$SCRATCH to avoid \$HOME quota limits."
    # Optionally: exit 1
fi

# Create venv in scratch directory
VENV_DIR="$SCRATCH/venvs/hackathon_task_3"
mkdir -p "$VENV_DIR"

if [ ! -d "$VENV_DIR/bin" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing requirements with --no-cache-dir..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir --require-virtualenv -r requirements.txt

echo "Setup complete. Run 'source $VENV_DIR/bin/activate' to use."
