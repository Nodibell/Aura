import pytest
import os
import tempfile
import numpy as np
import pandas as pd
import subprocess
import json
import sys


def test_run_predict_batch_direct():
    from Aura.pipelines.tabular import analyze_tabular
    from Aura.analyze import run_predict_batch

    # Create dummy dataset
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": np.random.normal(size=100),
        "num2": np.random.normal(size=100),
        "target": np.random.choice(["A", "B"], size=100)
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        input_csv = os.path.join(tmpdir, "input.csv")
        output_csv = os.path.join(tmpdir, "output.csv")

        # Train and export model
        res = analyze_tabular(
            df=df, target_col="target", task_type_override="classification",
            row_count=100, col_count=3, columns=["num1", "num2", "target"],
            full_preview=None, missing={},
            numeric_cols=["num1", "num2"], categorical_cols=[],
            model_export_path=model_path
        )
        assert res["error"] is None
        assert os.path.exists(model_path)

        # Create input features file (without target column)
        input_df = pd.DataFrame({
            "num1": [0.5, -0.2, 1.1],
            "num2": [-0.1, 0.8, -0.5]
        })
        input_df.to_csv(input_csv, index=False)

        # Run batch prediction helper
        result = run_predict_batch(model_path, input_csv, output_csv)
        assert result["success"] is True
        assert result["row_count"] == 3
        assert os.path.exists(output_csv)

        # Verify output CSV has the prediction column
        out_df = pd.read_csv(output_csv)
        assert out_df.shape == (3, 3)
        assert "target" in out_df.columns
        assert out_df["target"].tolist() in [["A", "A", "A"], ["A", "B", "A"], ["B", "B", "B"], ["B", "A", "B"], ["A", "B", "B"], ["B", "A", "A"], ["A", "A", "B"], ["B", "B", "A"]] # check it has labels


def test_predict_batch_cli():
    # Verify CLI execution works end-to-end
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": np.random.normal(size=50),
        "num2": np.random.normal(size=50),
        "target": np.random.choice([0, 1], size=50)
    })

    from Aura.pipelines.tabular import analyze_tabular
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        input_csv = os.path.join(tmpdir, "input.csv")
        output_csv = os.path.join(tmpdir, "output.csv")

        # Export model
        res = analyze_tabular(
            df=df, target_col="target", task_type_override="classification",
            row_count=50, col_count=3, columns=["num1", "num2", "target"],
            full_preview=None, missing={},
            numeric_cols=["num1", "num2"], categorical_cols=[],
            model_export_path=model_path
        )
        assert res["error"] is None

        # Create input CSV
        input_df = pd.DataFrame({
            "num1": [0.1, 0.2],
            "num2": [-0.1, -0.2]
        })
        input_df.to_csv(input_csv, index=False)

        # Invoke analyze.py CLI
        cmd = [
            sys.executable,
            "Aura/analyze.py",
            "--predict",
            "--model-path", model_path,
            "--input-file-path", input_csv,
            "--output-csv-path", output_csv
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        
        # Verify stdout JSON
        out_json = json.loads(result.stdout.strip())
        assert out_json["success"] is True
        assert out_json["row_count"] == 2
        assert os.path.exists(output_csv)
