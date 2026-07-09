import pytest
import os
import tempfile
import numpy as np
import pandas as pd


def _make_classification_df(n=100, seed=42):
    np.random.seed(seed)
    return pd.DataFrame({
        "num1": np.random.normal(size=n),
        "num2": np.random.normal(size=n),
        "target": np.random.choice(["A", "B"], size=n)
    })


def test_tabular_selected_model_overrides_export():
    """
    When selected_model is provided and it exists in trained_models, the export
    function should be called with that model name, NOT best_model_name.
    """
    from Aura.pipelines.tabular import analyze_tabular

    df = _make_classification_df()
    numeric_cols = ["num1", "num2"]
    categorical_cols = []
    columns = list(df.columns)
    row_count = len(df)
    col_count = len(df.columns)
    missing = df.isna().sum().to_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "selected_model.joblib")

        # First run to discover available model names
        result_base = analyze_tabular(
            df=df, target_col="target", task_type_override="classification",
            row_count=row_count, col_count=col_count, columns=columns,
            full_preview=None, missing=missing,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
        )
        assert result_base["error"] is None
        models_compared = result_base.get("models_compared", [])
        assert len(models_compared) > 0

        # Pick any model (winner or non-winner)
        winner = result_base["metrics"]["model"]
        non_winner = next(
            (m["name"] for m in models_compared if m["name"] != winner),
            winner
        )

        # Run again with selected_model override
        result = analyze_tabular(
            df=df, target_col="target", task_type_override="classification",
            row_count=row_count, col_count=col_count, columns=columns,
            full_preview=None, missing=missing,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            model_export_path=model_path,
            selected_model=non_winner
        )

        assert result["error"] is None
        assert os.path.exists(model_path), f"Model not exported to {model_path}"


def test_tabular_selected_model_ignored_if_missing():
    """
    When selected_model names a model that doesn't exist in trained_models,
    the pipeline should fall back to exporting the best model without errors.
    """
    from Aura.pipelines.tabular import analyze_tabular

    df = _make_classification_df()
    numeric_cols = ["num1", "num2"]
    categorical_cols = []
    columns = list(df.columns)
    row_count = len(df)
    col_count = len(df.columns)
    missing = df.isna().sum().to_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "fallback_model.joblib")

        result = analyze_tabular(
            df=df, target_col="target", task_type_override="classification",
            row_count=row_count, col_count=col_count, columns=columns,
            full_preview=None, missing=missing,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            model_export_path=model_path,
            selected_model="NonExistentModel_XYZ_123"
        )

        assert result["error"] is None
        assert os.path.exists(model_path), "Best model fallback should be exported"
