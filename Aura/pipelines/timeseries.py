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
            
            # 50-trial Optuna hyperparameter tuning for Random Forest (chronological split validation)
            print_progress(0.66, "Tuning Time Series Random Forest with Optuna...")
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            tuning_split_idx = int(len(X_train) * 0.8)
            if tuning_split_idx >= 2:
                tuning_X_tr, tuning_X_val = X_train[:tuning_split_idx], X_train[tuning_split_idx:]
                tuning_y_tr, tuning_y_val = y_train[:tuning_split_idx], y_train[tuning_split_idx:]
                
                def objective(trial):
                    n_estimators = trial.suggest_int("n_estimators", 10, 100)
                    max_depth = trial.suggest_int("max_depth", 3, 10)
                    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                    clf.fit(tuning_X_tr, tuning_y_tr)
                    preds = clf.predict(tuning_X_val)
                    return accuracy_score(tuning_y_val, preds)
                
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50, timeout=8.0)
                best_n = study.best_params.get("n_estimators", 100)
                best_d = study.best_params.get("max_depth", 5)
            else:
                best_n, best_d = 100, 5
                
            rf = RandomForestClassifier(n_estimators=best_n, max_depth=best_d, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_acc_rf = accuracy_score(y_test, rf_preds)
            
            if rf_acc_rf >= lr_acc:
                best_model = "Random Forest Classifier"
                best_score = rf_acc_rf
                best_preds = rf_preds
                best_clf = rf
            else:
                best_model = "Logistic Regression"
                best_score = lr_acc
                best_preds = lr_preds
                best_clf = lr
                
            models_compared = [
                {"name": "Logistic Regression", "score": float(lr_acc), "metric": "Accuracy"},
                {"name": f"Random Forest Classifier (n={best_n}, d={best_d})", "score": float(rf_acc_rf), "metric": "Accuracy"}
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
            
            # 50-trial Optuna hyperparameter tuning for Random Forest (chronological split validation)
            print_progress(0.66, "Tuning Time Series Random Forest with Optuna...")
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            tuning_split_idx = int(len(X_train) * 0.8)
            if tuning_split_idx >= 2:
                tuning_X_tr, tuning_X_val = X_train[:tuning_split_idx], X_train[tuning_split_idx:]
                tuning_y_tr, tuning_y_val = y_train[:tuning_split_idx], y_train[tuning_split_idx:]
                
                def objective(trial):
                    n_estimators = trial.suggest_int("n_estimators", 10, 100)
                    max_depth = trial.suggest_int("max_depth", 3, 10)
                    reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                    reg.fit(tuning_X_tr, tuning_y_tr)
                    preds = reg.predict(tuning_X_val)
                    return r2_score(tuning_y_val, preds)
                
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50, timeout=8.0)
                best_n = study.best_params.get("n_estimators", 100)
                best_d = study.best_params.get("max_depth", 5)
            else:
                best_n, best_d = 100, 5
                
            rf = RandomForestRegressor(n_estimators=best_n, max_depth=best_d, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_r2_rf = r2_score(y_test, rf_preds)
            
            # Holt-Winters Exponential Smoothing (Phase 4)
            es_fit = None
            es_preds = None
            es_r2 = -100.0
            try:
                from statsmodels.tsa.api import ExponentialSmoothing
                periods = 7 if len(y_train) > 14 else 2
                es = ExponentialSmoothing(y_train, seasonal_periods=periods, trend='add', seasonal='add', initialization_method="estimated")
                es_fit = es.fit()
                forecast_res = es_fit.forecast(len(y_test))
                es_preds = forecast_res.to_numpy() if hasattr(forecast_res, "to_numpy") else np.asarray(forecast_res)
                es_r2 = max(r2_score(y_test, es_preds), -100.0)
            except Exception as es_err:
                sys.stderr.write(f"Warning: Holt-Winters ES failed: {str(es_err)}. Trying simpler ES...\n")
                try:
                    es = ExponentialSmoothing(y_train, trend='add', seasonal=None)
                    es_fit = es.fit()
                    forecast_res = es_fit.forecast(len(y_test))
                    es_preds = forecast_res.to_numpy() if hasattr(forecast_res, "to_numpy") else np.asarray(forecast_res)
                    es_r2 = max(r2_score(y_test, es_preds), -100.0)
                except Exception:
                    es_fit = None
            
            # Compare models
            best_model = "Linear Regression"
            best_score = lr_r2
            best_preds = lr_preds
            best_reg = lr
            
            # ARIMA model (D6)
            arima_fit = None
            arima_preds = None
            arima_r2 = -100.0
            try:
                from statsmodels.tsa.arima.model import ARIMA
                d = 0 if (adf_p is not None and adf_p < 0.05) else 1
                arima_model = ARIMA(y_train, order=(1, d, 1))
                arima_fit = arima_model.fit()
                forecast_res = arima_fit.forecast(steps=len(y_test))
                arima_preds = forecast_res.to_numpy() if hasattr(forecast_res, "to_numpy") else np.asarray(forecast_res)
                arima_r2 = max(r2_score(y_test, arima_preds), -100.0)
            except Exception as arima_err:
                sys.stderr.write(f"Warning: ARIMA fitting failed: {str(arima_err)}\n")
            
            if rf_r2_rf >= best_score:
                best_model = "Random Forest Regressor"
                best_score = rf_r2_rf
                best_preds = rf_preds
                best_reg = rf
                
            if es_fit is not None and es_r2 >= best_score:
                best_model = "Holt-Winters ES"
                best_score = es_r2
                best_preds = es_preds
                best_reg = es_fit
                
            if arima_fit is not None and arima_r2 >= best_score:
                best_model = "ARIMA"
                best_score = arima_r2
                best_preds = arima_preds
                best_reg = arima_fit
                
            test_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
            models_compared = [
                {"name": "Linear Regression", "score": float(lr_r2), "metric": "R\u00b2 Score"},
                {"name": "Random Forest Regressor", "score": float(rf_r2_rf), "metric": "R\u00b2 Score"}
            ]
            if es_fit is not None:
                models_compared.append({"name": "Holt-Winters ES", "score": float(es_r2), "metric": "R\u00b2 Score"})
            if arima_fit is not None:
                models_compared.append({"name": "ARIMA", "score": float(arima_r2), "metric": "R\u00b2 Score"})
                
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

        # 5. Forecast Chart (True vs Predicted over time)
        forecast_data = []
        # We plot the test set predictions chronologically
        if "__is_test" in df.columns:
            time_test = df[time_col].loc[test_mask].dt.strftime('%Y-%m-%d').tolist()
        else:
            time_test = df[time_col].iloc[split_idx:].dt.strftime('%Y-%m-%d').tolist()
        
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
