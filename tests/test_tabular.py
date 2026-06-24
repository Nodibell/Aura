import pytest
import numpy as np
import pandas as pd
from Aura.pipelines.tabular import preprocess, train_models, compute_metrics, build_charts, analyze_tabular

def test_preprocess_and_tabular_pipeline():
    # Make a dummy classification dataset
    np.random.seed(42)
    n_samples = 100
    data = {
        "num1": np.random.normal(size=n_samples),
        "num2": np.random.normal(size=n_samples),
        "cat1": np.random.choice(["X", "Y", "Z"], size=n_samples),
        "target": np.random.choice(["A", "B"], size=n_samples)
    }
    # Add some nulls to test imputation
    data["num1"][10] = np.nan
    data["cat1"][20] = np.nan
    
    df = pd.DataFrame(data)
    
    # Define columns
    numeric_cols = ["num1", "num2"]
    categorical_cols = ["cat1"]
    columns = list(df.columns)
    
    # 1. Test preprocess function directly
    X_train = df.drop(columns=["target"])
    y_train = df["target"].values
    
    X_train_proc, X_test_proc, X_val_proc, _ = preprocess(
        X_train=X_train,
        y_train=y_train,
        X_test=X_train.copy(),
        y_test=y_train,
        X_val=None,
        y_val=None,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        text_cols=[],
        target_encode_cols=[],
        one_hot_cols=["cat1"],
        is_classification=True
    )
    
    # Check that processed datasets have no NaNs
    assert X_train_proc.isna().sum().sum() == 0
    assert X_test_proc.isna().sum().sum() == 0
    
    # 2. Test train_models
    best_model_name, best_model_obj, models_compared, best_preds, rf_classical_model, le = train_models(
        X_train_proc, y_train, X_test_proc, y_train, is_classification=True
    )
    
    assert best_model_name is not None
    assert best_model_obj is not None
    assert len(models_compared) > 0
    assert len(best_preds) == len(y_train)
    
    # 3. Test compute_metrics
    cv_scores, cv_mean, cv_std, dummy_score, confusion_matrix_data, val_metrics, val_confusion_matrix_data = compute_metrics(
        best_model_name=best_model_name,
        best_model_obj=best_model_obj,
        X_train=X_train_proc,
        y_train=y_train,
        X_test=X_test_proc,
        y_test=y_train,
        X_val=None,
        y_val=None,
        is_classification=True,
        le=le,
        X_processed=X_train_proc,
        y=y_train,
        X_train_full=X_train,
        y_train_full=y_train,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        text_cols=[],
        target_encode_cols=[],
        one_hot_cols=["cat1"]
    )
    
    assert len(cv_scores) > 0
    assert cv_mean is not None
    assert dummy_score is not None
    
    # 4. Test build_charts
    charts = build_charts(
        best_model_name=best_model_name,
        best_model_obj=best_model_obj,
        X_train=X_train_proc,
        y_train=y_train,
        X_test=X_test_proc,
        y_test=y_train,
        is_classification=True,
        le=le,
        target_series=df["target"],
        target_col="target",
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        X_processed=X_train_proc,
        best_preds=best_preds,
        rf_classical_model=rf_classical_model
    )
    
    assert len(charts) > 0
    
    # 5. Test full analyze_tabular classification entry point
    row_count = len(df)
    col_count = len(df.columns)
    missing = df.isna().sum().to_dict()
    
    result = analyze_tabular(
        df=df,
        target_col="target",
        task_type_override="classification",
        row_count=row_count,
        col_count=col_count,
        columns=columns,
        full_preview=None,
        missing=missing,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        smart_sample=False
    )
    
    assert result["error"] is None
    assert result["metrics"]["score_type"] == "Accuracy"
    assert len(result["charts"]) > 0
    assert result["cv_mean"] is not None

def test_tabular_pipeline_regression():
    # Make a dummy regression dataset
    np.random.seed(42)
    n_samples = 100
    data = {
        "num1": np.random.normal(size=n_samples),
        "num2": np.random.normal(size=n_samples),
        "cat1": np.random.choice(["X", "Y", "Z"], size=n_samples),
        "target": np.random.normal(size=n_samples)
    }
    df = pd.DataFrame(data)
    
    numeric_cols = ["num1", "num2"]
    categorical_cols = ["cat1"]
    columns = list(df.columns)
    row_count = len(df)
    col_count = len(df.columns)
    missing = df.isna().sum().to_dict()
    
    result = analyze_tabular(
        df=df,
        target_col="target",
        task_type_override="regression",
        row_count=row_count,
        col_count=col_count,
        columns=columns,
        full_preview=None,
        missing=missing,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        smart_sample=False
    )
    
    assert result["error"] is None
    assert result["metrics"]["score_type"] == "R² Score"
    assert len(result["charts"]) > 0
    assert result["cv_mean"] is not None

def test_tabular_pipeline_feature_selection():
    # Make a dummy wide regression dataset
    np.random.seed(42)
    n_samples = 100
    data = {f"num{i}": np.random.normal(size=n_samples) for i in range(1, 21)}
    # Add a target that only depends on num1 and num2
    data["target"] = data["num1"] * 2 + data["num2"] * 0.5 + np.random.normal(scale=0.1, size=n_samples)
    df = pd.DataFrame(data)
    
    numeric_cols = [f"num{i}" for i in range(1, 21)]
    categorical_cols = []
    columns = list(df.columns)
    row_count = len(df)
    col_count = len(df.columns)
    missing = df.isna().sum().to_dict()
    
    result = analyze_tabular(
        df=df,
        target_col="target",
        task_type_override="regression",
        row_count=row_count,
        col_count=col_count,
        columns=columns,
        full_preview=None,
        missing=missing,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        smart_sample=False,
        feature_selection=True
    )
    
    assert result["error"] is None
    assert result["metrics"]["score_type"] == "R² Score"
    assert len(result["charts"]) > 0
    assert result["cv_mean"] is not None

