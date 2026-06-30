import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from utils.event_bus import publish_progress
from pipelines.deep_learning import train_and_evaluate_tabular_nn

# -------------------------------------------------------------------------
# 1. Explainability & PDP/ICE
# -------------------------------------------------------------------------

def get_pdp_predictions(model, df):
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df)
            if len(probs.shape) == 2:
                if probs.shape[1] == 2:
                    return probs[:, 1]
                else:
                    return probs[:, 0]
            else:
                return probs[0][:, 1]
        except Exception:
            return model.predict(df)
    else:
        return model.predict(df)

def calculate_pdp_ice(model, X, feature_name, grid_resolution=20, num_ice_samples=10):
    col_vals = X[feature_name].values
    min_val = col_vals.min()
    max_val = col_vals.max()
    if min_val == max_val:
        return []
    
    grid = np.linspace(min_val, max_val, grid_resolution)
    
    pdp_samples = X
    if len(pdp_samples) > 50:
        pdp_samples = pdp_samples.sample(n=50, random_state=42)
        
    ice_samples = pdp_samples
    if len(ice_samples) > num_ice_samples:
        ice_samples = ice_samples.head(num_ice_samples)
        
    pdp_y = []
    ice_y = [[] for _ in range(len(ice_samples))]
    
    pdp_df_temp = pdp_samples.copy()
    ice_df_temp = ice_samples.copy()
    
    for v in grid:
        pdp_df_temp[feature_name] = v
        pdp_preds = get_pdp_predictions(model, pdp_df_temp)
        pdp_y.append(float(np.mean(pdp_preds)))
        
        ice_df_temp[feature_name] = v
        ice_preds = get_pdp_predictions(model, ice_df_temp)
        for idx, pred_val in enumerate(ice_preds):
            ice_y[idx].append(float(pred_val))
            
    points = []
    for idx, v in enumerate(grid):
        points.append({
            "x_val": feature_name,
            "x_num": float(v),
            "y": float(pdp_y[idx]),
            "series": "PDP"
        })
    for s_idx in range(len(ice_samples)):
        for idx, v in enumerate(grid):
            points.append({
                "x_val": feature_name,
                "x_num": float(v),
                "y": float(ice_y[s_idx][idx]),
                "series": f"ICE_{s_idx}"
            })
    return points


# -------------------------------------------------------------------------
# 2. AutoML Tabular Model Ingestion & Training
# -------------------------------------------------------------------------

def train_tabular_models(X_train, y_train, X_test, y_test, is_classification):
    charts = []
    models_compared = []
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
    
    if is_classification:
        publish_progress(0.62, "Training Logistic Regression baseline...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_preds)
        lr_f1 = f1_score(y_test, lr_preds, average='weighted', zero_division=0)
        lr_prec = precision_score(y_test, lr_preds, average='weighted', zero_division=0)
        lr_rec = recall_score(y_test, lr_preds, average='weighted', zero_division=0)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        from collections import Counter
        counts = Counter(y_train_encoded)
        can_stratify = len(counts) > 1 and min(counts.values()) >= 2
        
        tuning_X_tr, tuning_X_val, tuning_y_tr, tuning_y_val = train_test_split(
            X_train, y_train_encoded, test_size=0.2, random_state=42,
            stratify=y_train_encoded if can_stratify else None
        )
        
        publish_progress(0.66, "Tuning hyperparameters with Optuna...")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            model_type = trial.suggest_categorical("model_type", ["rf", "xgb"])
            if model_type == "rf":
                n_estimators = trial.suggest_int("rf_n_estimators", 10, 100)
                max_depth = trial.suggest_int("rf_max_depth", 3, 10)
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            else:
                n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
                max_depth = trial.suggest_int("xgb_max_depth", 3, 8)
                learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
                clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, n_jobs=-1, eval_metric="mlogloss")
            
            clf.fit(tuning_X_tr, tuning_y_tr)
            preds = clf.predict(tuning_X_val)
            return accuracy_score(tuning_y_val, preds)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, timeout=8.0)
        best_params = study.best_params
        
        rf_best_n = best_params.get("rf_n_estimators", 100)
        rf_best_d = best_params.get("rf_max_depth", 5)
        publish_progress(0.72, f"Training Tuned Random Forest (n_estimators={rf_best_n}, max_depth={rf_best_d})...")
        rf = RandomForestClassifier(n_estimators=rf_best_n, max_depth=rf_best_d, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_acc_rf = accuracy_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds, average='weighted', zero_division=0)
        rf_prec = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
        rf_rec = recall_score(y_test, rf_preds, average='weighted', zero_division=0)
        
        xgb_best_n = best_params.get("xgb_n_estimators", 100)
        xgb_best_d = best_params.get("xgb_max_depth", 5)
        xgb_best_lr = best_params.get("xgb_learning_rate", 0.1)
        publish_progress(0.76, f"Training Tuned XGBoost (n_estimators={xgb_best_n}, max_depth={xgb_best_d}, lr={xgb_best_lr:.4f})...")
        xgb = XGBClassifier(n_estimators=xgb_best_n, max_depth=xgb_best_d, learning_rate=xgb_best_lr, random_state=42, n_jobs=-1, eval_metric="mlogloss")
        xgb.fit(X_train, y_train_encoded)
        xgb_preds_encoded = xgb.predict(X_test)
        xgb_preds = le.inverse_transform(xgb_preds_encoded)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_f1 = f1_score(y_test, xgb_preds, average='weighted', zero_division=0)
        xgb_prec = precision_score(y_test, xgb_preds, average='weighted', zero_division=0)
        xgb_rec = recall_score(y_test, xgb_preds, average='weighted', zero_division=0)
        
        # LightGBM integration
        lgb_clf = None
        lgb_acc = -1.0
        lgb_f1, lgb_prec, lgb_rec = -1.0, -1.0, -1.0
        lgb_preds = None
        try:
            from lightgbm import LGBMClassifier
            publish_progress(0.78, "Training LightGBM Classifier...")
            lgb = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            lgb.fit(X_train, y_train_encoded)
            lgb_preds_encoded = lgb.predict(X_test)
            lgb_preds = le.inverse_transform(lgb_preds_encoded)
            lgb_acc = accuracy_score(y_test, lgb_preds)
            lgb_f1 = f1_score(y_test, lgb_preds, average='weighted', zero_division=0)
            lgb_prec = precision_score(y_test, lgb_preds, average='weighted', zero_division=0)
            lgb_rec = recall_score(y_test, lgb_preds, average='weighted', zero_division=0)
            lgb_clf = lgb
        except Exception as lgb_err:
            sys.stderr.write(f"Warning: LightGBM training failed: {str(lgb_err)}\n")
            
        # CatBoost integration
        cat_clf = None
        cat_acc = -1.0
        cat_f1, cat_prec, cat_rec = -1.0, -1.0, -1.0
        cat_preds = None
        try:
            from catboost import CatBoostClassifier
            publish_progress(0.80, "Training CatBoost Classifier...")
            cat = CatBoostClassifier(random_state=42, thread_count=1, verbose=0)
            cat.fit(X_train, y_train_encoded)
            cat_preds_encoded = cat.predict(X_test)
            if len(cat_preds_encoded.shape) > 1 and cat_preds_encoded.shape[1] == 1:
                cat_preds_encoded = cat_preds_encoded.ravel()
            cat_preds = le.inverse_transform(cat_preds_encoded)
            cat_acc = accuracy_score(y_test, cat_preds)
            cat_f1 = f1_score(y_test, cat_preds, average='weighted', zero_division=0)
            cat_prec = precision_score(y_test, cat_preds, average='weighted', zero_division=0)
            cat_rec = recall_score(y_test, cat_preds, average='weighted', zero_division=0)
            cat_clf = cat
        except Exception as cat_err:
            sys.stderr.write(f"Warning: CatBoost training failed: {str(cat_err)}\n")
            
        best_model = "Logistic Regression"
        best_score = lr_acc
        best_preds = lr_preds
        best_clf = lr
        
        if rf_acc_rf >= best_score:
            best_model = "Tuned Random Forest"
            best_score = rf_acc_rf
            best_preds = rf_preds
            best_clf = rf
            
        if xgb_acc >= best_score:
            best_model = "Tuned XGBoost"
            best_score = xgb_acc
            best_preds = xgb_preds
            best_clf = xgb
            
        if lgb_clf is not None and lgb_acc >= best_score:
            best_model = "LightGBM"
            best_score = lgb_acc
            best_preds = lgb_preds
            best_clf = lgb_clf
            
        if cat_clf is not None and cat_acc >= best_score:
            best_model = "CatBoost"
            best_score = cat_acc
            best_preds = cat_preds
            best_clf = cat_clf
            
        models_compared = [
            {"name": "Logistic Regression", "score": float(lr_acc), "metric": "Accuracy", "f1": float(lr_f1), "precision": float(lr_prec), "recall": float(lr_rec)},
            {"name": "Tuned Random Forest", "score": float(rf_acc_rf), "metric": "Accuracy", "f1": float(rf_f1), "precision": float(rf_prec), "recall": float(rf_rec)},
            {"name": "Tuned XGBoost", "score": float(xgb_acc), "metric": "Accuracy", "f1": float(xgb_f1), "precision": float(xgb_prec), "recall": float(xgb_rec)}
        ]
        if lgb_clf is not None:
            models_compared.append({"name": "LightGBM", "score": float(lgb_acc), "metric": "Accuracy", "f1": float(lgb_f1), "precision": float(lgb_prec), "recall": float(lgb_rec)})
        if cat_clf is not None:
            models_compared.append({"name": "CatBoost", "score": float(cat_acc), "metric": "Accuracy", "f1": float(cat_f1), "precision": float(cat_prec), "recall": float(cat_rec)})
        
        try:
            publish_progress(0.82, "Training Tabular Deep Learning (CPU)...")
            dl_score, dl_preds, dl_model = train_and_evaluate_tabular_nn(
                X_train.to_numpy(), y_train, X_test.to_numpy(), y_test,
                is_classification=True, epochs=15, batch_size=128
            )
            dl_f1 = f1_score(y_test, dl_preds, average='weighted', zero_division=0)
            dl_prec = precision_score(y_test, dl_preds, average='weighted', zero_division=0)
            dl_rec = recall_score(y_test, dl_preds, average='weighted', zero_division=0)
            models_compared.append({"name": "Tabular Deep Learning (CPU)", "score": float(dl_score), "metric": "Accuracy", "f1": float(dl_f1), "precision": float(dl_prec), "recall": float(dl_rec)})
            if dl_score >= best_score:
                best_model = "Tabular Deep Learning"
                best_score = dl_score
                best_preds = dl_preds
                best_clf = dl_model
        except Exception as dl_err:
            sys.stderr.write(f"Warning: Tabular Deep Learning training failed: {str(dl_err)}\n")
            
        return best_model, best_clf, models_compared, best_preds, rf, le
        
    else:
        publish_progress(0.62, "Training Linear Regression baseline...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_r2 = r2_score(y_test, lr_preds)
        lr_mse = mean_squared_error(y_test, lr_preds)
        lr_rmse = np.sqrt(lr_mse)
        lr_mae = mean_absolute_error(y_test, lr_preds)
        
        tuning_X_tr, tuning_X_val, tuning_y_tr, tuning_y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        publish_progress(0.66, "Tuning hyperparameters with Optuna...")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            model_type = trial.suggest_categorical("model_type", ["rf", "xgb"])
            if model_type == "rf":
                n_estimators = trial.suggest_int("rf_n_estimators", 10, 100)
                max_depth = trial.suggest_int("rf_max_depth", 3, 10)
                reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            else:
                n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
                max_depth = trial.suggest_int("xgb_max_depth", 3, 8)
                learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
                reg = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, n_jobs=-1)
            
            reg.fit(tuning_X_tr, tuning_y_tr)
            preds = reg.predict(tuning_X_val)
            return r2_score(tuning_y_val, preds)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, timeout=8.0)
        best_params = study.best_params
        
        rf_best_n = best_params.get("rf_n_estimators", 100)
        rf_best_d = best_params.get("rf_max_depth", 5)
        publish_progress(0.72, f"Training Tuned Random Forest (n_estimators={rf_best_n}, max_depth={rf_best_d})...")
        rf = RandomForestRegressor(n_estimators=rf_best_n, max_depth=rf_best_d, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_r2_rf = r2_score(y_test, rf_preds)
        rf_mse = mean_squared_error(y_test, rf_preds)
        rf_rmse = np.sqrt(rf_mse)
        rf_mae = mean_absolute_error(y_test, rf_preds)
        
        xgb_best_n = best_params.get("xgb_n_estimators", 100)
        xgb_best_d = best_params.get("xgb_max_depth", 5)
        xgb_best_lr = best_params.get("xgb_learning_rate", 0.1)
        publish_progress(0.76, f"Training Tuned XGBoost (n_estimators={xgb_best_n}, max_depth={xgb_best_d}, lr={xgb_best_lr:.4f})...")
        xgb = XGBRegressor(n_estimators=xgb_best_n, max_depth=xgb_best_d, learning_rate=xgb_best_lr, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_preds)
        xgb_mse = mean_squared_error(y_test, xgb_preds)
        xgb_rmse = np.sqrt(xgb_mse)
        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        
        # LightGBM Regressor
        lgb_reg = None
        lgb_r2 = -999.0
        lgb_mse, lgb_rmse, lgb_mae = 0.0, 0.0, 0.0
        lgb_preds = None
        try:
            from lightgbm import LGBMRegressor
            publish_progress(0.78, "Training LightGBM Regressor...")
            lgb = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
            lgb.fit(X_train, y_train)
            lgb_preds = lgb.predict(X_test)
            lgb_r2 = r2_score(y_test, lgb_preds)
            lgb_mse = mean_squared_error(y_test, lgb_preds)
            lgb_rmse = np.sqrt(lgb_mse)
            lgb_mae = mean_absolute_error(y_test, lgb_preds)
            lgb_reg = lgb
        except Exception as lgb_err:
            sys.stderr.write(f"Warning: LightGBM training failed: {str(lgb_err)}\n")
            
        # CatBoost Regressor
        cat_reg = None
        cat_r2 = -999.0
        cat_mse, cat_rmse, cat_mae = 0.0, 0.0, 0.0
        cat_preds = None
        try:
            from catboost import CatBoostRegressor
            publish_progress(0.80, "Training CatBoost Regressor...")
            cat = CatBoostRegressor(random_state=42, thread_count=1, verbose=0)
            cat.fit(X_train, y_train)
            cat_preds = cat.predict(X_test)
            cat_r2 = r2_score(y_test, cat_preds)
            cat_mse = mean_squared_error(y_test, cat_preds)
            cat_rmse = np.sqrt(cat_mse)
            cat_mae = mean_absolute_error(y_test, cat_preds)
            cat_reg = cat
        except Exception as cat_err:
            sys.stderr.write(f"Warning: CatBoost training failed: {str(cat_err)}\n")
            
        best_model = "Linear Regression"
        best_score = lr_r2
        best_preds = lr_preds
        best_reg = lr
        
        if rf_r2_rf >= best_score:
            best_model = "Tuned Random Forest"
            best_score = rf_r2_rf
            best_preds = rf_preds
            best_reg = rf
            
        if xgb_r2 >= best_score:
            best_model = "Tuned XGBoost"
            best_score = xgb_r2
            best_preds = xgb_preds
            best_reg = xgb
            
        if lgb_reg is not None and lgb_r2 >= best_score:
            best_model = "LightGBM"
            best_score = lgb_r2
            best_preds = lgb_preds
            best_reg = lgb_reg
            
        if cat_reg is not None and cat_r2 >= best_score:
            best_model = "CatBoost"
            best_score = cat_r2
            best_preds = cat_preds
            best_reg = cat_reg
            
        models_compared = [
            {"name": "Linear Regression", "score": float(lr_r2), "metric": "R² Score", "mse": float(lr_mse), "rmse": float(lr_rmse), "mae": float(lr_mae)},
            {"name": "Tuned Random Forest", "score": float(rf_r2_rf), "metric": "R² Score", "mse": float(rf_mse), "rmse": float(rf_rmse), "mae": float(rf_mae)},
            {"name": "Tuned XGBoost", "score": float(xgb_r2), "metric": "R² Score", "mse": float(xgb_mse), "rmse": float(xgb_rmse), "mae": float(xgb_mae)}
        ]
        if lgb_reg is not None:
            models_compared.append({"name": "LightGBM", "score": float(lgb_r2), "metric": "R² Score", "mse": float(lgb_mse), "rmse": float(lgb_rmse), "mae": float(lgb_mae)})
        if cat_reg is not None:
            models_compared.append({"name": "CatBoost", "score": float(cat_r2), "metric": "R² Score", "mse": float(cat_mse), "rmse": float(cat_rmse), "mae": float(cat_mae)})
        
        try:
            publish_progress(0.82, "Training Tabular Deep Learning (CPU)...")
            dl_score, dl_preds, dl_model = train_and_evaluate_tabular_nn(
                X_train.to_numpy(), y_train, X_test.to_numpy(), y_test,
                is_classification=False, epochs=15, batch_size=128
            )
            dl_mse = mean_squared_error(y_test, dl_preds)
            dl_rmse = np.sqrt(dl_mse)
            dl_mae = mean_absolute_error(y_test, dl_preds)
            models_compared.append({"name": "Tabular Deep Learning (CPU)", "score": float(dl_score), "metric": "R² Score", "mse": float(dl_mse), "rmse": float(dl_rmse), "mae": float(dl_mae)})
            if dl_score >= best_score:
                best_model = "Tabular Deep Learning"
                best_score = dl_score
                best_preds = dl_preds
                best_reg = dl_model
        except Exception as dl_err:
            sys.stderr.write(f"Warning: Tabular Deep Learning training failed: {str(dl_err)}\n")
            
        return best_model, best_reg, models_compared, best_preds, rf, None


# -------------------------------------------------------------------------
# 3. Strategy Pattern for Forecasting
# -------------------------------------------------------------------------

class ForecastingStrategy:
    def __init__(self):
        self.model = None

    def fit(self, y_train, X_train=None):
        raise NotImplementedError

    def predict_or_forecast(self, steps, X_test=None):
        raise NotImplementedError

class ArimaForecasting(ForecastingStrategy):
    def __init__(self, adf_p=None):
        super().__init__()
        self.adf_p = adf_p
        self.best_order = (1, 1, 1)

    def fit(self, y_train, X_train=None):
        try:
            import pmdarima as pm
            sys.stderr.write("Running pmdarima auto_arima to select p, d, q dynamically...\n")
            stepwise_fit = pm.auto_arima(
                y_train,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            self.best_order = stepwise_fit.order
            self.model = stepwise_fit.arima_res_
        except Exception as pm_err:
            sys.stderr.write(f"Warning: pmdarima auto_arima search failed: {str(pm_err)}. Falling back to statsmodels grid search...\n")
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                best_aic = float("inf")
                d_val = 0 if (self.adf_p is not None and self.adf_p < 0.05) else 1
                for p_val in [0, 1, 2]:
                    for q_val in [0, 1, 2]:
                        if p_val == 0 and q_val == 0:
                            continue
                        try:
                            test_model = SARIMAX(y_train, order=(p_val, d_val, q_val), enforce_stationarity=False, enforce_invertibility=False)
                            test_results = test_model.fit(disp=False, maxiter=50)
                            if test_results.aic < best_aic:
                                best_aic = test_results.aic
                                self.best_order = (p_val, d_val, q_val)
                        except Exception:
                            pass
                
                arima_model = SARIMAX(y_train, order=self.best_order, enforce_stationarity=False, enforce_invertibility=False)
                self.model = arima_model.fit(disp=False, maxiter=50)
            except Exception as arima_err:
                sys.stderr.write(f"Warning: Dynamic SARIMAX fitting failed: {str(arima_err)}\n")

    def predict_or_forecast(self, steps, X_test=None):
        if self.model is None:
            return np.zeros(steps)
        forecast_res = self.model.forecast(steps=steps)
        return forecast_res.to_numpy() if hasattr(forecast_res, "to_numpy") else np.asarray(forecast_res)

class HoltWintersForecasting(ForecastingStrategy):
    def fit(self, y_train, X_train=None):
        from statsmodels.tsa.api import ExponentialSmoothing
        try:
            periods = 7 if len(y_train) > 14 else 2
            es = ExponentialSmoothing(y_train, seasonal_periods=periods, trend='add', seasonal='add', initialization_method="estimated")
            self.model = es.fit()
        except Exception as es_err:
            sys.stderr.write(f"Warning: Holt-Winters ES failed: {str(es_err)}. Trying simpler ES...\n")
            try:
                es = ExponentialSmoothing(y_train, trend='add', seasonal=None)
                self.model = es.fit()
            except Exception:
                self.model = None

    def predict_or_forecast(self, steps, X_test=None):
        if self.model is None:
            return np.zeros(steps)
        forecast_res = self.model.forecast(steps)
        return forecast_res.to_numpy() if hasattr(forecast_res, "to_numpy") else np.asarray(forecast_res)

class MLRegressorForecasting(ForecastingStrategy):
    def __init__(self, regressor_cls, **kwargs):
        super().__init__()
        self.regressor_cls = regressor_cls
        self.kwargs = kwargs

    def fit(self, y_train, X_train=None):
        if X_train is None:
            X_train = np.arange(len(y_train)).reshape(-1, 1)
        self.model = self.regressor_cls(**self.kwargs)
        self.model.fit(X_train, y_train)

    def predict_or_forecast(self, steps, X_test=None):
        if self.model is None:
            return np.zeros(steps)
        if X_test is None:
            X_test = np.arange(steps).reshape(-1, 1)
        return self.model.predict(X_test)
