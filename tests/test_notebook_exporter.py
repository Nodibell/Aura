import os
import json
import tempfile
from Aura.analyze import analyze

def test_notebook_export_replaces_parquet_path():
    # Set the original file path environment variable
    original_path = "sample_data/iris.csv"
    os.environ["AURA_ORIGINAL_FILE_PATH"] = original_path

    # Create a temporary file path for the notebook
    with tempfile.TemporaryDirectory() as tmp_dir:
        notebook_path = os.path.join(tmp_dir, "test_notebook.ipynb")

        # Run analysis (which will generate the notebook)
        # We pass a temporary parquet path as the input to simulate Arrow cache
        # but the generated notebook should reference 'original_path' instead
        res = analyze(
            file_path="sample_data/iris.csv", # We use iris as input
            target_col="species",
            dataset_type="tabular",
            notebook_export_path=notebook_path
        )

        assert "error" not in res or res["error"] is None
        assert os.path.exists(notebook_path)

        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        # Inspect the code cells to verify the FILE_PATH value
        found_file_path_assignment = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                if "FILE_PATH " in source and "=" in source:
                    found_file_path_assignment = True
                    # The exported string format for the file path should contain original_path
                    assert repr(original_path) in source, f"Expected {repr(original_path)} in cell source: {source}"

        assert found_file_path_assignment, "Could not find FILE_PATH assignment in notebook cells"

    # Clean up environment variable
    if "AURA_ORIGINAL_FILE_PATH" in os.environ:
        del os.environ["AURA_ORIGINAL_FILE_PATH"]
