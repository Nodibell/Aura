"""
arrow_helper.py — Arrow IPC serialization utility for Aura.

Converts any dataset using the existing `utils.loader.load_dataset` module
to an in-memory or on-disk Apache Arrow binary IPC record batch stream.
This allows massive performance gains (up to 100x) over HTTP by bypassing
expensive JSON row-by-row serialization.
"""

from __future__ import annotations

import os
import sys

# Align import path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)

import types
import importlib.util

# 1. Check if workspace layout: _SCRIPT_DIR = Aura/utils/ -> container basename is "utils"
# 2. Check if flat bundle: _SCRIPT_DIR = Aura.app/Contents/Resources/ -> container basename is "Resources"
if os.path.basename(_SCRIPT_DIR) != "utils" and os.path.exists(os.path.join(_SCRIPT_DIR, "loader.py")):
    # Flat bundle layout: register virtual namespace mapping
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
        
    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = []
    sys.modules["utils"] = utils_mod
    
    # Import sibling scripts directly and map to virtual submodules
    import helpers
    sys.modules["utils.helpers"] = helpers
    utils_mod.helpers = helpers
    
    import loader
    sys.modules["utils.loader"] = loader
    utils_mod.loader = loader
else:
    # Workspace layout: ensure parent dir is in sys.path
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)


import argparse
import pyarrow as pa
import pandas as pd
from utils.loader import load_dataset



def to_arrow(input_path: str, output_path: str) -> None:
    """Read a dataset from any supported format and serialize to an Arrow IPC stream."""
    df = load_dataset(input_path)
    
    # Ensure column names are strictly string typed for Arrow
    df.columns = df.columns.astype(str)
    
    table = pa.Table.from_pandas(df)
    with pa.OSFile(output_path, "wb") as f:
        with pa.ipc.new_stream(f, table.schema) as writer:
            writer.write_table(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aura Apache Arrow In-Memory IPC Helper")
    parser.add_argument("--to-arrow", nargs=2, metavar=("INPUT_PATH", "OUTPUT_PATH"),
                        help="Convert dataset at INPUT_PATH to Arrow IPC stream file at OUTPUT_PATH")
    parser.add_argument("--selftest", action="store_true", help="Run diagnostic checks")
    args = parser.parse_args()

    if args.to_arrow:
        to_arrow(args.to_arrow[0], args.to_arrow[1])
        print("Successfully converted dataset to Arrow IPC format.")
        sys.exit(0)
        
    elif args.selftest:
        import tempfile
        # Create temp dummy csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col_a,col_b\n1,foo\n2,bar\n")
            tmp_csv = f.name
        
        tmp_arrow = tmp_csv + ".arrow"
        try:
            to_arrow(tmp_csv, tmp_arrow)
            # Verify Arrow stream can be parsed back
            with pa.OSFile(tmp_arrow, "rb") as f:
                with pa.ipc.open_stream(f) as reader:
                    table = reader.read_all()
                    df = table.to_pandas()
                    assert len(df) == 2
                    assert list(df.columns) == ["col_a", "col_b"]
            print("✅ Arrow IPC self-test passed successfully.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Arrow IPC self-test failed: {e}")
            sys.exit(1)
        finally:
            if os.path.exists(tmp_csv):
                os.unlink(tmp_csv)
            if os.path.exists(tmp_arrow):
                os.unlink(tmp_arrow)
