import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from Aura.pipelines.model_engine import (
    calculate_pdp_ice,
    train_tabular_models,
    ArimaForecasting,
    HoltWintersForecasting,
    MLRegressorForecasting
)

def test_calculate_pdp_ice():
    # Simple data and model
    X = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    model = LinearRegression()
    model.fit(X, y)
    
    points = calculate_pdp_ice(model, X, "feat", grid_resolution=5, num_ice_samples=2)
    assert len(points) > 0
    # PDP + ICE series points should exist
    series_names = {p["series"] for p in points}
    assert "PDP" in series_names
    assert "ICE_0" in series_names

def test_train_tabular_models():
    # Classification check
    X_train = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    y_train = np.array(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
    X_test = pd.DataFrame({"feat1": [2.5, 7.5]})
    y_test = np.array(["A", "B"])
    
    best_model, best_clf, models_compared, best_preds, rf, le = train_tabular_models(
        X_train, y_train, X_test, y_test, is_classification=True
    )
    assert best_model in ["Logistic Regression", "Tuned Random Forest", "Tuned XGBoost", "Tabular Deep Learning"]
    assert len(models_compared) >= 3
    assert len(best_preds) == 2

def test_forecasting_strategies():
    y_train = np.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0, 48.0, 57.0])
    
    # Holt-Winters
    hw = HoltWintersForecasting()
    hw.fit(y_train)
    preds = hw.predict_or_forecast(steps=3)
    assert len(preds) == 3
    
    # ML Regressor
    ml = MLRegressorForecasting(LinearRegression)
    ml.fit(y_train)
    preds_ml = ml.predict_or_forecast(steps=3)
    assert len(preds_ml) == 3
