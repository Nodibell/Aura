import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from utils.helpers import print_progress, _export_model_and_code
from utils.charts import generate_boxplots
from utils.profiler import profile_dataset

def analyze_timeseries(df, target_col, time_col, task_type_override,
                       row_count, col_count, columns, full_preview, missing,
                       numeric_cols, categorical_cols,
                       file_path=None, model_export_path=None, code_export_path=None):
    # Initialize stationarity variables early
    adf_stat = None
    adf_p = None
    adf_conclusion = "Unknown"
    try:
        df = df.replace([np.inf, -np.inf], np.nan)
        print_progress(0.30, "Setting up time series dates and sorting...")
        
        # 1. Resolve time column
        if not time_col:
            # Look for date/time-like columns
            cols_lower = [c.lower() for c in columns]
            ts_keywords = ["date", "time", "timestamp", "datetime", "year", "month", "period", "week"]
            for col in columns:
                if any(kw in col.lower() for kw in ts_keywords):
                    time_col = col
                    break
            if not time_col:
                # Fallback: find any column that parses to datetime with few NaNs
                for col in columns:
                    if col != target_col:
                        try:
                            parsed = pd.to_datetime(df[col], errors='coerce')
                            if parsed.isnull().sum() / len(df) < 0.5:
                                time_col = col
                                break
                        except Exception:
                            pass
            if not time_col:
                # Final fallback: first column that is not target
                time_col = [c for c in columns if c != target_col][0] if len(columns) > 1 else columns[0]
        
        # 2. Sort chronologically
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            # Drop rows with null timestamps
            df_sorted = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)
            if df_sorted.empty:
                df_sorted = df.sort_index().reset_index(drop=True)
            else:
                df = df_sorted
        except Exception as sort_err:
            sys.stderr.write(f"Warning: Failed to parse/sort by datetime column: {str(sort_err)}. Analyzing in index order.\n")
            df = df.sort_index().reset_index(drop=True)
            
        print_progress(0.40, "Engineering time series lag features...")
        # Target variable setup
        # If target has missing values, fill them using linear interpolation (standard for TS)
        y_raw = df[target_col]
        # Check if numeric
        is_numeric_target = pd.api.types.is_numeric_dtype(y_raw.dtype)
        
        # We can override classification/regression
        is_classification = False
        if task_type_override == "classification":
            is_classification = True
        elif task_type_override == "regression" or task_type_override == "forecast":
            is_classification = False
        else:
            is_classification = not is_numeric_target
            
        if is_classification:
            # Classification TS
            y = y_raw.ffill().bfill().astype(str).to_numpy()
        else:
            # Regression/Forecasting TS
            y_series = y_raw.interpolate(method='linear').ffill().bfill().fillna(0)
            y = y_series.to_numpy()
            
            # Run stationarity test early
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(y_series)
                adf_stat = float(adf_result[0])
                adf_p = float(adf_result[1])
                adf_conclusion = "Stationary" if adf_p < 0.05 else "Non-Stationary (Unit Root Present)"
            except Exception as adf_err:
                sys.stderr.write(f"Warning: Early ADF test failed: {str(adf_err)}\n")
            
        # 3. Create lag features
        X_df = pd.DataFrame(index=df.index)
        # Lag features from target if target is numeric
        if not is_classification:
            for lag in [1, 2, 3, 7]:
                if len(df) > lag:
                    X_df[f"target_lag_{lag}"] = y_series.shift(lag)
            # Rolling means
            for roll in [3, 7]:
                if len(df) > roll:
                    X_df[f"target_roll_mean_{roll}"] = y_series.shift(1).rolling(roll).mean()
                    
        # Add other numeric columns as lags too
        for col in numeric_cols:
            if col != target_col and col != time_col:
                col_series = df[col].interpolate(method='linear').ffill().bfill().fillna(0)
                for lag in [1, 3]:
                    if len(df) > lag:
                        X_df[f"{col}_lag_{lag}"] = col_series.shift(lag)
                        
        # Drop rows with NaN (introduced by lags)
        X_df = X_df.bfill().ffill().fillna(0)
        if X_df.empty:
            # If no features could be created (e.g. tiny dataset), just use index
            X_df["time_index"] = df.index
            
        X_processed = X_df.to_numpy()
        
        print_progress(0.55, "Splitting time series chronologically & training...")
        
        X_val, y_val = None, None
        val_metrics = None
        val_confusion_matrix_data = None
        
        if "__is_test" in df.columns:
            train_mask = (df["__is_test"] == 0).to_numpy()
            test_mask = (df["__is_test"] == 1).to_numpy()
            val_mask = (df["__is_test"] == 2).to_numpy()
            
            X_train_full = X_processed[train_mask]
            y_train_full = y[train_mask]
            
            if np.any(val_mask):
                X_val, y_val = X_processed[val_mask], y[val_mask]
                
            if np.any(test_mask):
                X_train, X_test = X_train_full, X_processed[test_mask]
                y_train, y_test = y_train_full, y[test_mask]
            else:
                split_idx = int(len(X_train_full) * 0.8)
                if split_idx < 1 or split_idx >= len(X_train_full):
                    split_idx = max(1, len(X_train_full) - 1)
                X_train, X_test = X_train_full[:split_idx], X_train_full[split_idx:]
                y_train, y_test = y_train_full[:split_idx], y_train_full[split_idx:]
        else:
            # 4. Temporal split: 80% train, 20% test
            split_idx = int(len(df) * 0.8)
            if split_idx < 1 or split_idx >= len(df):
                split_idx = max(1, len(df) - 1)
                
            X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
        # Cap dataset size for model training to avoid slow execution/OOM (keep most recent for training)
        if len(X_train) > 10000:
            X_train = X_train[-10000:]
            y_train = y_train[-10000:]
        if len(X_test) > 5000:
            X_test = X_test[:5000]
            y_test = y_test[:5000]
        if X_val is not None and len(X_val) > 5000:
            X_val = X_val[:5000]
            y_val = y_val[:5000]
        
        charts = []
        models_compared = []
        metrics = {}
        
        # Fit models
        if is_classification:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_acc = accuracy_score(y_test, lr_preds)
            
            # 50-trial Optuna hyperparameter tuning for Random Forest & XGBoost (chronological split validation)
            print_progress(0.66, "Tuning Time Series models with Optuna...")
            import optuna
            from xgboost import XGBClassifier
            from sklearn.preprocessing import LabelEncoder
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            tuning_split_idx = int(len(X_train) * 0.8)
            if tuning_split_idx >= 2:
                tuning_X_tr, tuning_X_val = X_train[:tuning_split_idx], X_train[tuning_split_idx:]
                tuning_y_tr_encoded = y_train_encoded[:tuning_split_idx]
                tuning_y_val_encoded = y_train_encoded[tuning_split_idx:]
                
                def objective(trial):
                    model_type = trial.suggest_categorical("model_type", ["rf", "xgb"])
                    if model_type == "rf":
                        n_estimators = trial.suggest_int("rf_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("rf_max_depth", 3, 10)
                        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=2)
                        clf.fit(tuning_X_tr, tuning_y_tr_encoded)
                        preds = clf.predict(tuning_X_val)
                        return accuracy_score(tuning_y_val_encoded, preds)
                    else:
                        n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("xgb_max_depth", 3, 8)
                        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
                        clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, n_jobs=2, eval_metric="mlogloss")
                        clf.fit(tuning_X_tr, tuning_y_tr_encoded)
                        preds = clf.predict(tuning_X_val)
                        return accuracy_score(tuning_y_val_encoded, preds)
                
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50, timeout=8.0)
                best_params = study.best_params
            else:
                best_params = {}
                
            # Train Tuned RF
            rf_best_n = best_params.get("rf_n_estimators", 100)
            rf_best_d = best_params.get("rf_max_depth", 5)
            rf = RandomForestClassifier(n_estimators=rf_best_n, max_depth=rf_best_d, random_state=42, n_jobs=2)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_acc_rf = accuracy_score(y_test, rf_preds)
            
            # Train Tuned XGBoost
            xgb_best_n = best_params.get("xgb_n_estimators", 100)
            xgb_best_d = best_params.get("xgb_max_depth", 5)
            xgb_best_lr = best_params.get("xgb_learning_rate", 0.1)
            xgb = XGBClassifier(n_estimators=xgb_best_n, max_depth=xgb_best_d, learning_rate=xgb_best_lr, random_state=42, n_jobs=2, eval_metric="mlogloss")
            xgb.fit(X_train, y_train_encoded)
            xgb_preds_encoded = xgb.predict(X_test)
            xgb_preds = le.inverse_transform(xgb_preds_encoded)
            xgb_acc_xgb = accuracy_score(y_test, xgb_preds)
            
            best_model = "Logistic Regression"
            best_score = lr_acc
            best_preds = lr_preds
            best_clf = lr
            
            if rf_acc_rf >= best_score:
                best_model = "Random Forest Classifier"
                best_score = rf_acc_rf
                best_preds = rf_preds
                best_clf = rf
                
            if xgb_acc_xgb >= best_score:
                best_model = "Tuned XGBoost Classifier"
                best_score = xgb_acc_xgb
                best_preds = xgb_preds
                best_clf = xgb
                
            models_compared = [
                {"name": "Logistic Regression", "score": float(lr_acc), "metric": "Accuracy"},
                {"name": f"Random Forest Classifier (n={rf_best_n}, d={rf_best_d})", "score": float(rf_acc_rf), "metric": "Accuracy"},
                {"name": f"Tuned XGBoost Classifier (n={xgb_best_n}, d={xgb_best_d}, lr={xgb_best_lr:.4f})", "score": float(xgb_acc_xgb), "metric": "Accuracy"}
            ]
            metrics = {
                "model": best_model,
                "score_type": "Accuracy",
                "score": float(best_score),
                "additional_metrics": {
                    "F1 Score": float(f1_score(y_test, best_preds, average='weighted', zero_division=0))
                }
            }
        else:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_r2 = r2_score(y_test, lr_preds)
            
            from sklearn.linear_model import Ridge, Lasso
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            ridge_preds = ridge.predict(X_test)
            ridge_r2 = r2_score(y_test, ridge_preds)
            
            lasso = Lasso(alpha=0.1, max_iter=2000)
            lasso.fit(X_train, y_train)
            lasso_preds = lasso.predict(X_test)
            lasso_r2 = r2_score(y_test, lasso_preds)
            
            # 50-trial Optuna hyperparameter tuning for Random Forest & XGBoost (chronological split validation)
            print_progress(0.66, "Tuning Time Series models with Optuna...")
            import optuna
            from xgboost import XGBRegressor
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            tuning_split_idx = int(len(X_train) * 0.8)
            if tuning_split_idx >= 2:
                tuning_X_tr, tuning_X_val = X_train[:tuning_split_idx], X_train[tuning_split_idx:]
                tuning_y_tr, tuning_y_val = y_train[:tuning_split_idx], y_train[tuning_split_idx:]
                
                def objective(trial):
                    model_type = trial.suggest_categorical("model_type", ["rf", "xgb"])
                    if model_type == "rf":
                        n_estimators = trial.suggest_int("rf_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("rf_max_depth", 3, 10)
                        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=2)
                        reg.fit(tuning_X_tr, tuning_y_tr)
                        preds = reg.predict(tuning_X_val)
                        return r2_score(tuning_y_val, preds)
                    else:
                        n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("xgb_max_depth", 3, 8)
                        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
                        reg = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, n_jobs=2)
                        reg.fit(tuning_X_tr, tuning_y_tr)
                        preds = reg.predict(tuning_X_val)
                        return r2_score(tuning_y_val, preds)
                
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50, timeout=8.0)
                best_params = study.best_params
            else:
                best_params = {}
                
            # Train Tuned RF
            rf_best_n = best_params.get("rf_n_estimators", 100)
            rf_best_d = best_params.get("rf_max_depth", 5)
            rf = RandomForestRegressor(n_estimators=rf_best_n, max_depth=rf_best_d, random_state=42, n_jobs=2)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_r2_rf = r2_score(y_test, rf_preds)
            
            # Train Tuned XGBoost
            xgb_best_n = best_params.get("xgb_n_estimators", 100)
            xgb_best_d = best_params.get("xgb_max_depth", 5)
            xgb_best_lr = best_params.get("xgb_learning_rate", 0.1)
            xgb = XGBRegressor(n_estimators=xgb_best_n, max_depth=xgb_best_d, learning_rate=xgb_best_lr, random_state=42, n_jobs=2)
            xgb.fit(X_train, y_train)
            xgb_preds = xgb.predict(X_test)
            xgb_r2_xgb = r2_score(y_test, xgb_preds)
            
            # Holt-Winters Exponential Smoothing Strategy
            from pipelines.model_engine import HoltWintersForecasting, ArimaForecasting
            
            hw_strategy = HoltWintersForecasting()
            hw_strategy.fit(y_train)
            es_preds = hw_strategy.predict_or_forecast(len(y_test))
            es_fit = hw_strategy.model
            es_r2 = max(r2_score(y_test, es_preds), -100.0) if es_fit is not None else -100.0
            
            # ARIMA Strategy
            arima_strategy = ArimaForecasting(adf_p=adf_p)
            arima_strategy.fit(y_train)
            arima_preds = arima_strategy.predict_or_forecast(len(y_test), X_test)
            arima_fit = arima_strategy.model
            best_order = arima_strategy.best_order
            arima_r2 = max(r2_score(y_test, arima_preds), -100.0) if arima_fit is not None else -100.0

            # ── Prophet Strategy (Phase 19) ──────────────────────────────────
            train_dates = None
            test_dates = None
            try:
                if hasattr(y_train, 'index'):
                    train_dates = df[time_col].loc[y_train.index]
                else:
                    train_dates = df[time_col].iloc[:len(y_train)]

                if test_file_path and os.path.exists(test_file_path):
                    last_date = pd.to_datetime(train_dates.iloc[-1])
                    test_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(y_test), freq='D')
                else:
                    if hasattr(y_test, 'index'):
                        test_dates = df[time_col].loc[y_test.index]
                    else:
                        test_dates = df[time_col].iloc[len(y_train):len(y_train)+len(y_test)]
            except Exception:
                pass

            prophet_r2 = -100.0
            prophet_fit = None
            prophet_preds = np.zeros(len(y_test))
            try:
                from pipelines.model_engine import ProphetForecasting
                prophet_strategy = ProphetForecasting()
                prophet_strategy.fit(y_train, train_dates)
                prophet_preds = prophet_strategy.predict_or_forecast(len(y_test), test_dates)
                prophet_fit = prophet_strategy.model
                prophet_r2 = max(r2_score(y_test, prophet_preds), -100.0) if prophet_fit is not None else -100.0
            except Exception as pr_err:
                sys.stderr.write(f"Warning: Prophet forecasting failed: {str(pr_err)}\n")

            # ── LSTM Strategy (Phase 19) ─────────────────────────────────────
            lstm_r2 = -100.0
            lstm_fit = None
            lstm_preds = np.zeros(len(y_test))
            try:
                from pipelines.model_engine import LSTMForecasting
                lstm_strategy = LSTMForecasting(lookback=7, epochs=20, hidden_dim=32)
                lstm_strategy.fit(y_train)
                lstm_preds = lstm_strategy.predict_or_forecast(len(y_test))
                lstm_fit = lstm_strategy.model
                lstm_r2 = max(r2_score(y_test, lstm_preds), -100.0) if lstm_fit is not None else -100.0
            except Exception as lstm_err:
                sys.stderr.write(f"Warning: LSTM forecasting failed: {str(lstm_err)}\n")
            
            # Compare models
            best_model = "Linear Regression"
            best_score = lr_r2
            best_preds = lr_preds
            best_reg = lr
            
            if ridge_r2 >= best_score:
                best_model = "Ridge Regression"
                best_score = ridge_r2
                best_preds = ridge_preds
                best_reg = ridge
                
            if lasso_r2 >= best_score:
                best_model = "Lasso Regression"
                best_score = lasso_r2
                best_preds = lasso_preds
                best_reg = lasso
                
            if rf_r2_rf >= best_score:
                best_model = "Random Forest Regressor"
                best_score = rf_r2_rf
                best_preds = rf_preds
                best_reg = rf
                
            if xgb_r2_xgb >= best_score:
                best_model = "Tuned XGBoost Regressor"
                best_score = xgb_r2_xgb
                best_preds = xgb_preds
                best_reg = xgb
                
            if es_fit is not None and es_r2 >= best_score:
                best_model = "Holt-Winters ES"
                best_score = es_r2
                best_preds = es_preds
                best_reg = es_fit
                
            if arima_fit is not None and arima_r2 >= best_score:
                best_model = f"ARIMA {best_order}"
                best_score = arima_r2
                best_preds = arima_preds
                best_reg = arima_fit

            if prophet_fit is not None and prophet_r2 >= best_score:
                best_model = "Prophet"
                best_score = prophet_r2
                best_preds = prophet_preds
                best_reg = prophet_fit

            if lstm_fit is not None and lstm_r2 >= best_score:
                best_model = "LSTM Network"
                best_score = lstm_r2
                best_preds = lstm_preds
                best_reg = lstm_fit
                
            test_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
            
            trained_models = {
                "Linear Regression": lr,
                "Ridge Regression": ridge,
                "Lasso Regression": lasso,
                f"Random Forest Regressor (n={rf_best_n}, d={rf_best_d})": rf,
                f"Tuned XGBoost Regressor (n={xgb_best_n}, d={xgb_best_d}, lr={xgb_best_lr:.4f})": xgb
            }
            if es_fit is not None:
                trained_models["Holt-Winters ES"] = es_fit
            if arima_fit is not None:
                trained_models[f"ARIMA {best_order}"] = arima_fit
            if prophet_fit is not None:
                trained_models["Prophet"] = prophet_fit
            if lstm_fit is not None:
                trained_models["LSTM Network"] = lstm_fit

            models_compared = [
                {"name": "Linear Regression", "score": float(lr_r2), "metric": "R\u00b2 Score"},
                {"name": "Ridge Regression", "score": float(ridge_r2), "metric": "R\u00b2 Score"},
                {"name": "Lasso Regression", "score": float(lasso_r2), "metric": "R\u00b2 Score"},
                {"name": f"Random Forest Regressor (n={rf_best_n}, d={rf_best_d})", "score": float(rf_r2_rf), "metric": "R\u00b2 Score"},
                {"name": f"Tuned XGBoost Regressor (n={xgb_best_n}, d={xgb_best_d}, lr={xgb_best_lr:.4f})", "score": float(xgb_r2_xgb), "metric": "R\u00b2 Score"}
            ]
            if es_fit is not None:
                models_compared.append({"name": "Holt-Winters ES", "score": float(es_r2), "metric": "R\u00b2 Score"})
            if arima_fit is not None:
                models_compared.append({"name": f"ARIMA {best_order}", "score": float(arima_r2), "metric": "R\u00b2 Score"})
            if prophet_fit is not None:
                models_compared.append({"name": "Prophet", "score": float(prophet_r2), "metric": "R\u00b2 Score"})
            if lstm_fit is not None:
                models_compared.append({"name": "LSTM Network", "score": float(lstm_r2), "metric": "R\u00b2 Score"})
                
            metrics = {
                "model": best_model,
                "score_type": "R\u00b2 Score",
                "score": float(best_score),
                "additional_metrics": {
                    "RMSE": float(test_rmse)
                }
            }
            
        # Evaluate on Validation set if present
        if X_val is not None and len(X_val) > 0:
            try:
                if is_classification:
                    val_preds = best_clf.predict(X_val)
                    val_acc = accuracy_score(y_val, val_preds)
                    val_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
                    val_metrics = {
                        "model": best_model,
                        "score_type": "Accuracy",
                        "score": float(val_acc),
                        "additional_metrics": {
                            "F1 Score": float(val_f1)
                        }
                    }
                    
                    classes_list = sorted(list(np.unique(y_val)))
                    cm_val = confusion_matrix(y_val, val_preds, labels=classes_list)
                    val_confusion_matrix_data = {
                        "labels": [str(c) for c in classes_list],
                        "values": cm_val.tolist()
                    }
                else:
                    val_preds = best_reg.predict(X_val)
                    val_r2 = r2_score(y_val, val_preds)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                    val_metrics = {
                        "model": best_model,
                        "score_type": "R\u00b2 Score",
                        "score": float(val_r2),
                        "additional_metrics": {
                            "RMSE": float(val_rmse)
                        }
                    }
            except Exception as val_err:
                sys.stderr.write(f"Warning: Validation evaluation failed in time series: {str(val_err)}\n")

        # 5. Forecast Chart (True vs Predicted over time) with Historical context
        if "__is_test" in df.columns:
            time_test = df[time_col].loc[test_mask].dt.strftime('%Y-%m-%d').tolist()
        else:
            time_test = df[time_col].iloc[split_idx:].dt.strftime('%Y-%m-%d').tolist()
            
        # Historical training data points for the timeline (up to 200 points)
        historical_data = []
        if "__is_test" in df.columns:
            train_indices = np.where(train_mask)[0]
            recent_train_indices = train_indices[-200:] if len(train_indices) > 200 else train_indices
            time_train = df[time_col].iloc[recent_train_indices].dt.strftime('%Y-%m-%d').tolist()
            y_train_plot = y[recent_train_indices]
        else:
            recent_train_len = min(200, split_idx)
            time_train = df[time_col].iloc[split_idx - recent_train_len:split_idx].dt.strftime('%Y-%m-%d').tolist()
            y_train_plot = y[split_idx - recent_train_len:split_idx]
            
        for i in range(len(y_train_plot)):
            historical_data.append({
                "x_val": str(time_train[i]),
                "x_num": None,
                "y": float(y_train_plot[i]),
                "series": "Historical"
            })
            
        forecast_data = []
        forecast_data.extend(historical_data)
        
        for i in range(len(y_test)):
            time_str = str(time_test[i])
            forecast_data.append({
                "x_val": time_str,
                "x_num": None,
                "y": float(y_test[i]),
                "series": "Actual"
            })
            forecast_data.append({
                "x_val": time_str,
                "x_num": None,
                "y": float(best_preds[i]),
                "series": "Forecast"
            })
            
        charts.append({
            "type": "line",
            "title": "Time Series Forecast (Actual vs Predicted)",
            "x_label": f"Date ({time_col})",
            "y_label": f"Target: {target_col}",
            "data": forecast_data
        })
        
        # Feature Importance Chart
        if "Random Forest" in best_model or "XGBoost" in best_model:
            try:
                importances = []
                if "Classifier" in best_model:
                    importances = best_clf.feature_importances_
                else:
                    importances = best_reg.feature_importances_
                
                feat_imp = sorted(zip(X_df.columns, importances), key=lambda x: x[1], reverse=True)
                top_feat_imp = feat_imp[:10]
                
                charts.append({
                    "type": "bar",
                    "title": "Time Series Lag Feature Importance",
                    "x_label": "Feature",
                    "y_label": "Importance Score",
                    "data": [{"x_val": name, "x_num": None, "y": float(score)} for name, score in top_feat_imp]
                })
            except Exception as feat_err:
                sys.stderr.write(f"Warning: Failed to compute feature importances: {str(feat_err)}\n")

        # Residuals Plot (for regression)
        if not is_classification:
            try:
                residuals = y_test - best_preds
                residuals_data = []
                for i in range(len(y_test)):
                    residuals_data.append({
                        "x_val": str(time_test[i]),
                        "x_num": None,
                        "y": float(residuals[i])
                    })
                charts.append({
                    "type": "scatter",
                    "title": "Residuals Over Time",
                    "x_label": f"Date ({time_col})",
                    "y_label": "Residual Error",
                    "data": residuals_data
                })
            except Exception as res_err:
                sys.stderr.write(f"Warning: Failed to generate residuals chart: {str(res_err)}\n")

        # Rolling Volatility Chart (for regression)
        if not is_classification:
            try:
                rolling_vol = pd.Series(y).rolling(window=7, min_periods=1).std().fillna(0).tolist()
                vol_dates = df[time_col].dt.strftime('%Y-%m-%d').tolist()
                vol_data = []
                
                # Sample down to keep it fast
                step = max(1, len(rolling_vol) // 500)
                for i in range(0, len(rolling_vol), step):
                    vol_data.append({
                        "x_val": str(vol_dates[i]),
                        "x_num": None,
                        "y": float(rolling_vol[i])
                    })
                charts.append({
                    "type": "line",
                    "title": "Target Variable Rolling Volatility (7-Day std dev)",
                    "x_label": f"Date ({time_col})",
                    "y_label": "Standard Deviation",
                    "data": vol_data
                })
            except Exception as vol_err:
                sys.stderr.write(f"Warning: Failed to compute rolling volatility: {str(vol_err)}\n")
        
        # 6. Dummy baseline comparison
        dummy_score = None
        try:
            if is_classification:
                dummy = DummyClassifier(strategy="most_frequent")
                dummy.fit(X_train, y_train)
                dummy_preds = dummy.predict(X_test)
                dummy_score = float(accuracy_score(y_test, dummy_preds))
            else:
                dummy = DummyRegressor(strategy="mean")
                dummy.fit(X_train, y_train)
                dummy_preds = dummy.predict(X_test)
                dummy_score = float(r2_score(y_test, dummy_preds))
        except Exception:
            pass

        # 7. ADF Stationarity Test & Decomposition (Statsmodels features)
        if not is_classification:
            print_progress(0.75, "Reporting statistical stationarity test results...")
                
            # Autocorrelation (ACF) & Partial Autocorrelation (PACF)
            print_progress(0.82, "Computing ACF/PACF autocorrelation...")
            try:
                from statsmodels.tsa.stattools import acf, pacf
                # Calculate lags: min of 20 or half the dataset
                lags_count = min(20, len(y_series) // 2 - 1)
                if lags_count > 0:
                    acf_vals = acf(y_series, nlags=lags_count)
                    pacf_vals = pacf(y_series, nlags=lags_count)
                    
                    charts.append({
                        "type": "bar",
                        "title": "Autocorrelation Function (ACF)",
                        "x_label": "Lag",
                        "y_label": "ACF Value",
                        "data": [{"x_val": f"Lag {i}", "x_num": None, "y": float(acf_vals[i])} for i in range(len(acf_vals))]
                    })
                    charts.append({
                        "type": "bar",
                        "title": "Partial Autocorrelation Function (PACF)",
                        "x_label": "Lag",
                        "y_label": "PACF Value",
                        "data": [{"x_val": f"Lag {i}", "x_num": None, "y": float(pacf_vals[i])} for i in range(1, len(pacf_vals))]
                    })
            except Exception as acf_err:
                sys.stderr.write(f"ACF/PACF computation failed: {str(acf_err)}\n")
                
            # Seasonal Decomposition
            print_progress(0.86, "Decomposing time series components...")
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                period_guess = 7
                if len(y_series) >= period_guess * 2:
                    decomp = seasonal_decompose(y_series, model='additive', period=period_guess)
                    
                    # Convert components to line charts
                    times_str = df[time_col].dt.strftime('%Y-%m-%d').tolist()
                    
                    charts.append({
                        "type": "line",
                        "title": "Time Series Decomposition: Trend Component",
                        "x_label": "Date",
                        "y_label": "Trend",
                        "data": [{"x_val": str(times_str[i]), "x_num": None, "y": float(decomp.trend.iloc[i])} 
                                 for i in range(len(df)) if not np.isnan(decomp.trend.iloc[i])]
                    })
                    charts.append({
                        "type": "line",
                        "title": "Time Series Decomposition: Seasonal Component",
                        "x_label": "Date",
                        "y_label": "Seasonal",
                        "data": [{"x_val": str(times_str[i]), "x_num": None, "y": float(decomp.seasonal.iloc[i])} 
                                 for i in range(len(df)) if not np.isnan(decomp.seasonal.iloc[i])]
                    })
                    charts.append({
                        "type": "line",
                        "title": "Time Series Decomposition: Residual Component",
                        "x_label": "Date",
                        "y_label": "Residual",
                        "data": [{"x_val": str(times_str[i]), "x_num": None, "y": float(decomp.resid.iloc[i])} 
                                 for i in range(len(df)) if not np.isnan(decomp.resid.iloc[i])]
                    })
            except Exception as decomp_err:
                sys.stderr.write(f"Seasonal decomposition failed: {str(decomp_err)}\n")
                
            # Generate outlier boxplots
            try:
                boxplots = generate_boxplots(df, numeric_cols, target_col=target_col)
                charts.extend(boxplots)
            except Exception as box_err:
                sys.stderr.write(f"Failed to generate timeseries boxplots: {str(box_err)}\n")
                
        # 8. Data profiling
        print_progress(0.90, "Profiling columns & generating data statistics...")
        profiling = profile_dataset(df)
            
        # 9. Markdown report summary
        print_progress(0.95, "Compiling summary report & finalizing...")
        summary_sections = []
        overview = f"### 📊 Time Series Overview\n"
        overview += f"- **Rows:** {row_count:,} | **Columns:** {col_count:,}\n"
        overview += f"- **Time Column:** `{time_col}` (Chronological sorting applied)\n"
        overview += f"- **Missing Value Cells:** {sum(missing.values()):,} total across {len([k for k,v in missing.items() if v > 0])} columns."
        summary_sections.append(overview)
        
        if not is_classification and adf_stat is not None:
            adf_rep = f"### 📈 Stationarity Analysis (ADF Test)\n"
            adf_rep += f"- **ADF Statistic:** `{adf_stat:.4f}`\n"
            adf_rep += f"- **p-value:** `{adf_p:.4e}`\n"
            adf_rep += f"- **Conclusion:** Dataset is **{adf_conclusion}**."
            summary_sections.append(adf_rep)
            
        target_info = f"### 🎯 Target Variable Analysis (`{target_col}`)\n"
        target_info += f"- **Task Type:** {('Classification' if is_classification else 'Forecasting (Regression)')}\n"
        if not is_classification:
            target_info += f"- **Range:** `{y_series.min():.2f}` to `{y_series.max():.2f}` (Mean: `{y_series.mean():.2f}`, Median: `{y_series.median():.2f}`)"
        summary_sections.append(target_info)
        
        model_perf = f"### 🤖 Machine Learning Forecast Performance\n"
        model_perf += f"- **Best Model:** `{best_model}`\n"
        model_perf += f"- **Primary Score ({metrics['score_type']}):** `{metrics['score']:.4f}`\n"
        if 'additional_metrics' in metrics and metrics['additional_metrics']:
            for am_name, am_val in metrics['additional_metrics'].items():
                model_perf += f"- **{am_name}:** `{am_val:.4f}`"
        summary_sections.append(model_perf)
        
        summary = "\n\n".join(summary_sections)
        
        # Phase 1: Model & Code Export
        if model_export_path or code_export_path:
            model_to_save = best_clf if is_classification else best_reg
            feature_names = list(X_df.columns) if 'X_df' in locals() else []
            _export_model_and_code(
                model_to_save, model_export_path, code_export_path,
                file_path, "timeseries", target_col, None,
                "classification" if is_classification else "regression",
                feature_names, best_model, numeric_cols, categorical_cols, None, time_col=time_col
            )
            
            if model_export_path and 'trained_models' in locals():
                import os
                base_dir = os.path.dirname(model_export_path)
                base_name = os.path.basename(model_export_path)
                name_without_ext, ext = os.path.splitext(base_name)
                
                for m_name, m_obj in trained_models.items():
                    if m_obj is not None:
                        m_name_safe = m_name.replace(" ", "_").replace("²", "2").replace("(", "").replace(")", "").replace("=", "").replace(",", "")
                        sub_model_path = os.path.join(base_dir, f"{name_without_ext}_{m_name_safe}{ext}")
                        _export_model_and_code(
                            m_obj, sub_model_path, None,
                            file_path, "timeseries", target_col, None,
                            "classification" if is_classification else "regression",
                            feature_names, m_name, numeric_cols, categorical_cols, None, time_col=time_col
                        )

        return {
            "summary": summary,
            "columns": columns,
            "row_count": int(row_count),
            "col_count": int(col_count),
            "task_type": "forecast" if not is_classification else "classification",
            "numeric_col_count": len(numeric_cols),
            "categorical_col_count": len(categorical_cols),
            "text_col_count": 0,
            "missing_values": {k: int(v) for k, v in missing.items()},
            "correlations": [],
            "charts": charts,
            "metrics": metrics,
            "val_metrics": val_metrics,
            "val_confusion_matrix": val_confusion_matrix_data,
            "models_compared": models_compared,
            "target_column": target_col,
            "full_preview": full_preview,
            "dummy_baseline_score": dummy_score,
            "cv_scores": [],
            "cv_mean": None,
            "cv_std": None,
            "confusion_matrix": None,
            "profiling": profiling,
            "error": None
        }
    except Exception as e:
        import traceback
        return {"error": f"An error occurred during Time Series execution: {str(e)}\n{traceback.format_exc()}"}
