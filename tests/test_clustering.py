import pytest
import numpy as np
import pandas as pd
from Aura.pipelines.clustering import analyze_clustering

def test_analyze_clustering():
    # Make a dummy clustering dataset (no target column)
    np.random.seed(42)
    n_samples = 150
    data = {
        "num1": np.random.normal(size=n_samples),
        "num2": np.random.normal(size=n_samples),
        "cat1": np.random.choice(["X", "Y", "Z"], size=n_samples),
    }
    # Add some nulls to test imputation
    data["num1"][10] = np.nan
    data["cat1"][20] = np.nan
    
    df = pd.DataFrame(data)
    
    numeric_cols = ["num1", "num2"]
    categorical_cols = ["cat1"]
    columns = list(df.columns)
    row_count = len(df)
    col_count = len(df.columns)
    missing = df.isna().sum().to_dict()
    
    result = analyze_clustering(
        df=df,
        row_count=row_count,
        col_count=col_count,
        columns=columns,
        full_preview=None,
        missing=missing,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )
    
    assert result["error"] is None
    assert result["metrics"]["model"] == "K-Means + HDBSCAN Clustering"
    assert result["metrics"]["score_type"] == "Silhouette Score"
    assert len(result["charts"]) > 0
    # K-Means Cluster and HDBSCAN Cluster should be in columns
    assert "K-Means Cluster" in result["columns"]
    assert "HDBSCAN Cluster" in result["columns"]
