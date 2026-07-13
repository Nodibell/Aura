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

# Check if uv is available
if ! command -v uv &> /dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv package manager not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

if ! command -v uv &> /dev/null; then
    echo "Error: Failed to install or locate uv."
    exit 1
fi

echo "Found uv version: $(uv --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in $(pwd)/.venv using uv..."
    uv venv .venv
else
    echo "Virtual environment (.venv) already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies using uv pip
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt using uv..."
    uv pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing base packages manually using uv..."
    uv pip install pandas numpy scikit-learn pyarrow kaggle huggingface_hub fastapi uvicorn pmdarima optuna shap xgboost lightgbm catboost ultralytics prophet sentence-transformers polars
fi

echo "=================================================="
echo "              Verifying Environment               "
echo "=================================================="

# Test imports and output path
python -c "
import sys
import pandas as pd
import numpy as np
import polars as pl
import sklearn
import torch
print(f'Python path: {sys.executable}')
print(f'Pandas version: {pd.__version__} - OK')
print(f'NumPy version: {np.__version__} - OK')
print(f'Polars version: {pl.__version__} - OK')
print(f'Scikit-learn version: {sklearn.__version__} - OK')
print(f'PyTorch version: {torch.__version__} - OK')
print(f'MPS GPU training support available: {torch.backends.mps.is_available()}')
"

# Test optional packages
python -c "
def check_import(name, import_name=None):
    import_name = import_name or name
    try:
        __import__(import_name)
        print(f'{name} - OK')
    except ImportError:
        print(f'{name} - MISSING')

check_import('PyArrow', 'pyarrow')
check_import('Kaggle API', 'kaggle')
check_import('Hugging Face Hub', 'huggingface_hub')
check_import('FastAPI', 'fastapi')
check_import('Uvicorn', 'uvicorn')
check_import('pmdarima')
check_import('Optuna', 'optuna')
check_import('SHAP', 'shap')
check_import('XGBoost', 'xgboost')
check_import('LightGBM', 'lightgbm')
check_import('CatBoost', 'catboost')
check_import('Ultralytics (YOLO)', 'ultralytics')
check_import('Prophet', 'prophet')
check_import('Sentence Transformers', 'sentence_transformers')
check_import('Polars', 'polars')
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
