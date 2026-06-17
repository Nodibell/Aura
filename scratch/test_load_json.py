import sys
import os

# Ensure local import paths work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Aura")))

from utils.loader import download_dataset, load_dataset

url = "https://huggingface.co/datasets/FIdo-AI/ua-squad"

try:
    print("Downloading dataset...")
    local_path = download_dataset(url)
    print("Downloaded file path:", local_path)
    
    print("\nLoading dataset preview...")
    df = load_dataset(local_path, nrows=5)
    print("\nDataFrame preview:")
    print(df)
    print("\nColumns:", list(df.columns))
    print("Rows:", len(df))
    print("SUCCESS!")
except Exception as e:
    import traceback
    print("Error:", e)
    traceback.print_exc()
