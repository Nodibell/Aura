import pytest
import numpy as np
import pandas as pd
from Aura.pipelines.timeseries import analyze_timeseries

def test_timeseries_pipeline():
    # Make a synthetic time series dataset
    np.random.seed(42)
    n_samples = 120
    dates = pd.date_range(start="2026-01-01", periods=n_samples, freq="D")
    
    # Simple linear trend + sine wave seasonality + noise
    trend = np.linspace(10, 50, n_samples)
    seasonality = 5.0 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    noise = np.random.normal(scale=2.0, size=n_samples)
    values = trend + seasonality + noise
    
    df = pd.DataFrame({
        "date": dates,
        "value": values,
        "other_feature": np.random.normal(size=n_samples)
    })
    
    # Define columns
    columns = list(df.columns)
    row_count = len(df)
    col_count = len(df.columns)
    missing = df.isna().sum().to_dict()
    numeric_cols = ["value", "other_feature"]
    categorical_cols = []
    
    # Run the time series analysis
    result = analyze_timeseries(
        df=df,
        target_col="value",
        time_col="date",
        task_type_override="time_series",
        row_count=row_count,
        col_count=col_count,
        columns=columns,
        full_preview=None,
        missing=missing,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )
    
    # Check output
    assert result is not None
    assert result["error"] is None
    assert result["metrics"]["score_type"] == "R² Score"
    assert len(result["charts"]) > 0
    
    # Verify that forecast outputs are present
    forecast_chart = next((c for c in result["charts"] if "Forecast" in c["title"]), None)
    assert forecast_chart is not None
    assert len(forecast_chart["data"]) > 0
