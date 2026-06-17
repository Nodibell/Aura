#!/bin/bash
# Aura Virtual Environment Setup script
set -e

# Resolve script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=================================================="
echo "      Aura Python Environment Bootstrapper      "
echo "=================================================="

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 command not found. Please install Python 3 and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found python3 (Version $PYTHON_VERSION) at $(command -v python3)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in $(pwd)/.venv..."
    python3 -m venv .venv
else
    echo "Virtual environment (.venv) already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing base packages manually..."
    pip install pandas numpy scikit-learn pyarrow kaggle huggingface_hub
fi

echo "=================================================="
echo "              Verifying Environment               "
echo "=================================================="

# Test imports and output path
python -c "
import sys
import pandas as pd
import numpy as np
import sklearn
print(f'Python path: {sys.executable}')
print(f'Pandas version: {pd.__version__} - OK')
print(f'NumPy version: {np.__version__} - OK')
print(f'Scikit-learn version: {sklearn.__version__} - OK')
"

# Test optional packages
python -c "
try:
    import pyarrow
    print('PyArrow - OK')
except ImportError:
    print('PyArrow - MISSING')
try:
    import kaggle
    print('Kaggle API - OK')
except ImportError:
    print('Kaggle API - MISSING')
try:
    import huggingface_hub
    print('Hugging Face Hub - OK')
except ImportError:
    print('Hugging Face Hub - MISSING')
"

echo "=================================================="
echo "                 Setup Successful!                "
echo "=================================================="
echo "To use this virtual environment in the Aura app:"
echo "1. Go to Settings in the Aura app."
echo "2. Paste the following absolute path into the 'Custom Python Path' field:"
echo "$DIR/.venv/bin/python3"
echo ""
echo "To activate this environment in your terminal:"
echo "source $DIR/.venv/bin/activate"
echo "=================================================="
