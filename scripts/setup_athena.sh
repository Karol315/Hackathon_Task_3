#!/bin/bash

# Setup environment for Athena GPU Training
module add GCCcore/13.2.0 Python/3.11.5



# Create venv in scratch directory
VENV_DIR="$SCRATCH/venvs/hackathon_task_3"
mkdir -p "$(dirname "$VENV_DIR")"

# Check if venv exists and matches our current Python version
if [ -d "$VENV_DIR" ]; then
    VENV_PY_VER=$("$VENV_DIR/bin/python" --version 2>&1 | cut -d' ' -f2 | cut -d. -f1,2)
    CUR_PY_VER=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d. -f1,2)
    
    if [ "$VENV_PY_VER" != "$CUR_PY_VER" ]; then
        echo "Python version mismatch (Venv: $VENV_PY_VER, System: $CUR_PY_VER). Recreating venv..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR/bin" ]; then
    echo "Creating fresh virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing requirements with --no-cache-dir..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir --require-virtualenv -r requirements.txt

echo "Setup complete. Run 'source $VENV_DIR/bin/activate' to use."
