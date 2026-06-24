import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor

# Import torch BEFORE XGBoost n_jobs=-1 training ever runs.
# On macOS both XGBoost and PyTorch ship their own libomp. Whichever library
# initialises the OpenMP runtime first wins; the second one deadlocks waiting
# for an already-held init mutex. Importing torch here ensures PyTorch claims
# the runtime before any n_jobs=-1 sklearn/XGBoost call happens.
try:
    import torch as _torch_preload  # noqa: F401 – side-effect import only
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    pass

from utils.cleaning import StatefulCleaner
from utils.helpers import print_progress, _export_model_and_code
from utils.profiler import profile_dataset
from utils.charts import generate_boxplots
from pipelines.deep_learning import train_and_evaluate_tabular_nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class PandasTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=15):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def fit(self, X, y=None):
        import pandas as pd
        if hasattr(X, "iloc"):
            if len(X.shape) == 2:
                X = X.iloc[:, 0]
        elif hasattr(X, "shape") and len(X.shape) == 2:
            X = X.ravel()
        X_clean = [str(val) if val is not None and not pd.isna(val) else "" for val in X]
        self.vectorizer.fit(X_clean)
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        import pandas as pd
        if hasattr(X, "iloc"):
            if len(X.shape) == 2:
                X = X.iloc[:, 0]
        elif hasattr(X, "shape") and len(X.shape) == 2:
            X = X.ravel()
        X_clean = [str(val) if val is not None and not pd.isna(val) else "" for val in X]
        sparse_res = self.vectorizer.transform(X_clean)
        return sparse_res.toarray()
        
    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.get_feature_names_out(input_features)
        
    def set_output(self, transform=None):
        return self

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

def preprocess(X_train, y_train, X_test, y_test, X_val, y_val,
               categorical_cols, numeric_cols, text_cols,
               target_encode_cols, one_hot_cols, is_classification):
    # Determine target encoding target type
    if target_encode_cols:
        from sklearn.preprocessing import TargetEncoder
        if is_classification:
            unique_y_train = np.unique(y_train)
            te_target_type = "binary" if len(unique_y_train) == 2 else "multiclass"
        else:
            te_target_type = "continuous"
    else:
        te_target_type = None

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    transformers = []
    
    # 1. Numeric columns
    numeric_cols_present = [c for c in numeric_cols if c in X_train.columns]
    if numeric_cols_present:
        transformers.append((
            'num',
            SimpleImputer(strategy='median'),
            numeric_cols_present
        ))
        
    # 2. One-hot columns
    one_hot_cols_present = [c for c in one_hot_cols if c in X_train.columns]
    if one_hot_cols_present:
        transformers.append((
            'cat_onehot',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]),
            one_hot_cols_present
        ))
        
    # 3. Target encode columns
    target_encode_cols_present = [c for c in target_encode_cols if c in X_train.columns]
    if target_encode_cols_present:
        transformers.append((
            'cat_target',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('target', TargetEncoder(target_type=te_target_type, random_state=42))
            ]),
            target_encode_cols_present
        ))
        
    # 4. Text columns
    text_cols_present = [c for c in text_cols if c in X_train.columns]
    for col in text_cols_present:
        transformers.append((
            f'text_{col}',
            Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('tfidf', PandasTfidfVectorizer(max_features=15))
            ]),
            [col]
        ))
        
    if not transformers:
        transformers.append((
            'pass',
            FunctionTransformer(lambda x: x, accept_sparse=False),
            list(X_train.columns)
        ))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    preprocessor.set_output(transform="pandas")
    
    preprocessor.fit(X_train, y_train)
    
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test) if X_test is not None else None
    X_val_proc = preprocessor.transform(X_val) if X_val is not None else None
    
    # Sanitize column names for XGBoost compatibility (no [, ] or <)
    def clean_cols(df):
        if df is None:
            return None
        new_cols = []
        for col in df.columns:
            c = str(col)
            c = c.replace('[', '_').replace(']', '_').replace('<', 'lt_').replace('>', 'gt_')
            new_cols.append(c)
        df.columns = new_cols
        return df

    X_train_proc = clean_cols(X_train_proc)
    X_test_proc = clean_cols(X_test_proc)
    X_val_proc = clean_cols(X_val_proc)
        
    return X_train_proc, X_test_proc, X_val_proc, preprocessor


def train_models(X_train, y_train, X_test, y_test, is_classification):
    """
    Trains classical baselines, tunes tree-models using Optuna (50 trials),
    and trains a Tabular Deep Learning PyTorch wrapper model on local GPU.
    """
    charts = []
    models_compared = []
    
    if is_classification:
        # 1. Logistic Regression Baseline
        print_progress(0.62, "Training Logistic Regression baseline...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_preds)
        
        # 2. Hyperparameter tuning split
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
        
        print_progress(0.66, "Tuning hyperparameters with Optuna...")
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
        
        # Train Tuned RF
        rf_best_n = best_params.get("rf_n_estimators", 100)
        rf_best_d = best_params.get("rf_max_depth", 5)
        print_progress(0.72, f"Training Tuned Random Forest (n_estimators={rf_best_n}, max_depth={rf_best_d})...")
        rf = RandomForestClassifier(n_estimators=rf_best_n, max_depth=rf_best_d, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_acc_rf = accuracy_score(y_test, rf_preds)
        
        # Train Tuned XGBoost
        xgb_best_n = best_params.get("xgb_n_estimators", 100)
        xgb_best_d = best_params.get("xgb_max_depth", 5)
        xgb_best_lr = best_params.get("xgb_learning_rate", 0.1)
        print_progress(0.76, f"Training Tuned XGBoost (n_estimators={xgb_best_n}, max_depth={xgb_best_d}, lr={xgb_best_lr:.4f})...")
        xgb = XGBClassifier(n_estimators=xgb_best_n, max_depth=xgb_best_d, learning_rate=xgb_best_lr, random_state=42, n_jobs=-1, eval_metric="mlogloss")
        xgb.fit(X_train, y_train_encoded)
        xgb_preds_encoded = xgb.predict(X_test)
        xgb_preds = le.inverse_transform(xgb_preds_encoded)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        
        # Setup initial classical best model
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
            
        models_compared = [
            {"name": "Logistic Regression", "score": float(lr_acc), "metric": "Accuracy"},
            {"name": "Tuned Random Forest", "score": float(rf_acc_rf), "metric": "Accuracy"},
            {"name": "Tuned XGBoost", "score": float(xgb_acc), "metric": "Accuracy"}
        ]
        
        # Train Tabular NN Classifier wrapper
        try:
            print_progress(0.82, "Training Tabular Deep Learning (CPU)...")
            dl_score, dl_preds, dl_model = train_and_evaluate_tabular_nn(
                X_train.to_numpy(), y_train, X_test.to_numpy(), y_test,
                is_classification=True, epochs=15, batch_size=128
            )
            models_compared.append({"name": "Tabular Deep Learning (CPU)", "score": float(dl_score), "metric": "Accuracy"})
            if dl_score >= best_score:
                best_model = "Tabular Deep Learning"
                best_score = dl_score
                best_preds = dl_preds
                best_clf = dl_model
        except Exception as dl_err:
            sys.stderr.write(f"Warning: Tabular Deep Learning training failed: {str(dl_err)}\n")
            
        return best_model, best_clf, models_compared, best_preds, rf, le
        
    else:
        # 1. Linear Regression Baseline
        print_progress(0.62, "Training Linear Regression baseline...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_r2 = r2_score(y_test, lr_preds)
        
        # 2. Hyperparameter tuning split
        tuning_X_tr, tuning_X_val, tuning_y_tr, tuning_y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print_progress(0.66, "Tuning hyperparameters with Optuna...")
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
        
        # Train Tuned RF
        rf_best_n = best_params.get("rf_n_estimators", 100)
        rf_best_d = best_params.get("rf_max_depth", 5)
        print_progress(0.72, f"Training Tuned Random Forest (n_estimators={rf_best_n}, max_depth={rf_best_d})...")
        rf = RandomForestRegressor(n_estimators=rf_best_n, max_depth=rf_best_d, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_r2_rf = r2_score(y_test, rf_preds)
        
        # Train Tuned XGBoost
        xgb_best_n = best_params.get("xgb_n_estimators", 100)
        xgb_best_d = best_params.get("xgb_max_depth", 5)
        xgb_best_lr = best_params.get("xgb_learning_rate", 0.1)
        print_progress(0.76, f"Training Tuned XGBoost (n_estimators={xgb_best_n}, max_depth={xgb_best_d}, lr={xgb_best_lr:.4f})...")
        xgb = XGBRegressor(n_estimators=xgb_best_n, max_depth=xgb_best_d, learning_rate=xgb_best_lr, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_preds)
        
        # Setup initial classical best model
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
            
        models_compared = [
            {"name": "Linear Regression", "score": float(lr_r2), "metric": "R² Score"},
            {"name": "Tuned Random Forest", "score": float(rf_r2_rf), "metric": "R² Score"},
            {"name": "Tuned XGBoost", "score": float(xgb_r2), "metric": "R² Score"}
        ]
        
        # Train Tabular NN Regressor wrapper
        try:
            print_progress(0.82, "Training Tabular Deep Learning (CPU)...")
            dl_score, dl_preds, dl_model = train_and_evaluate_tabular_nn(
                X_train.to_numpy(), y_train, X_test.to_numpy(), y_test,
                is_classification=False, epochs=15, batch_size=128
            )
            models_compared.append({"name": "Tabular Deep Learning (CPU)", "score": float(dl_score), "metric": "R² Score"})
            if dl_score >= best_score:
                best_model = "Tabular Deep Learning"
                best_score = dl_score
                best_preds = dl_preds
                best_reg = dl_model
        except Exception as dl_err:
            sys.stderr.write(f"Warning: Tabular Deep Learning training failed: {str(dl_err)}\n")
            
        return best_model, best_reg, models_compared, best_preds, rf, None

def compute_metrics(best_model_name, best_model_obj, X_train, y_train, X_test, y_test, X_val, y_val,
                    is_classification, le, X_processed, y, X_train_full, y_train_full,
                    categorical_cols, numeric_cols, text_cols, target_encode_cols, one_hot_cols):
    """
    Computes diagnostic metrics: dummy baselines, confusion matrices, validation evaluations,
    and runs 5-fold cross-validation with fold-level preprocessing to completely avoid data leakage.
    """
    cv_scores = []
    cv_mean = None
    cv_std = None
    dummy_score = None
    confusion_matrix_data = None
    val_metrics = None
    val_confusion_matrix_data = None
    
    # 1. 5-Fold Cross Validation (with fold-level preprocessing)
    n_samples = len(X_train_full)
    if n_samples >= 5:
        if is_classification:
            from collections import Counter
            class_counts = Counter(y_train_full)
            min_class_count = min(class_counts.values()) if class_counts else 0
            if min_class_count >= 2:
                n_splits = min(5, min_class_count)
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
        try:
            from sklearn.base import clone
            for train_idx, val_idx in cv.split(X_train_full, y_train_full):
                X_tr_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
                y_tr_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]
                
                # Preprocess fold (fit on training fold, transform validation fold)
                X_tr_proc, X_val_proc, _, _ = preprocess(
                    X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, None, None,
                    categorical_cols, numeric_cols, text_cols, target_encode_cols, one_hot_cols, is_classification
                )
                
                # Clone and train classical model (avoid cloning DL wrapper for now)
                if "Tabular Deep Learning" in best_model_name:
                    # For DL, train a fresh TabularNN inside the cross-validation
                    # Use fewer epochs for CV folds — only need approximate score estimates
                    try:
                        from pipelines.deep_learning import TabularNNClassifier, TabularNNRegressor
                    except ModuleNotFoundError:
                        from deep_learning import TabularNNClassifier, TabularNNRegressor
                    if is_classification:
                        fold_model = TabularNNClassifier(epochs=8, batch_size=256)
                    else:
                        fold_model = TabularNNRegressor(epochs=8, batch_size=256)
                else:
                    fold_model = clone(best_model_obj)
                
                y_tr_fold_fit = y_tr_fold
                if is_classification and le is not None:
                    y_tr_fold_fit = le.transform(y_tr_fold)
                    
                fold_model.fit(X_tr_proc.to_numpy() if hasattr(fold_model, "device") else X_tr_proc, y_tr_fold_fit)
                preds = fold_model.predict(X_val_proc.to_numpy() if hasattr(fold_model, "device") else X_val_proc)
                
                if is_classification and le is not None:
                    preds = le.inverse_transform(preds)
                
                if is_classification:
                    cv_scores.append(float(accuracy_score(y_val_fold, preds)))
                else:
                    cv_scores.append(float(r2_score(y_val_fold, preds)))
            
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception as cv_err:
            sys.stderr.write(f"Warning: Fold-level cross-validation failed: {str(cv_err)}\n")
            
    # 2. Evaluate Dummy baseline on test set
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
    except Exception as dummy_err:
        sys.stderr.write(f"Warning: Dummy baseline evaluation failed: {str(dummy_err)}\n")
        
    # 3. Confusion Matrix for classification (test set)
    if is_classification:
        try:
            # Predict using best model
            best_preds = best_model_obj.predict(X_test)
            if "XGBoost" in best_model_name and le is not None:
                best_preds = le.inverse_transform(best_preds)
            classes_list = sorted(list(np.unique(y_test)))
            cm = confusion_matrix(y_test, best_preds, labels=classes_list)
            confusion_matrix_data = {
                "labels": [str(c) for c in classes_list],
                "values": cm.tolist()
            }
        except Exception as cm_err:
            sys.stderr.write(f"Warning: Confusion matrix calculation failed: {str(cm_err)}\n")
            
    # 4. Evaluate on Validation set if present
    if X_val is not None and len(X_val) > 0:
        try:
            if is_classification:
                val_preds = best_model_obj.predict(X_val)
                if "XGBoost" in best_model_name and le is not None:
                    val_preds = le.inverse_transform(val_preds)
                val_acc = accuracy_score(y_val, val_preds)
                val_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
                val_metrics = {
                    "model": best_model_name,
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
                val_preds = best_model_obj.predict(X_val)
                val_r2 = r2_score(y_val, val_preds)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                val_metrics = {
                    "model": best_model_name,
                    "score_type": "R² Score",
                    "score": float(val_r2),
                    "additional_metrics": {
                        "RMSE": float(val_rmse)
                    }
                }
        except Exception as val_err:
            sys.stderr.write(f"Warning: Validation evaluation failed: {str(val_err)}\n")
            
    return cv_scores, cv_mean, cv_std, dummy_score, confusion_matrix_data, val_metrics, val_confusion_matrix_data

def build_charts(best_model_name, best_model_obj, X_train, y_train, X_test, y_test,
                 is_classification, le, target_series, target_col, numeric_cols,
                 categorical_cols, X_processed, best_preds, rf_classical_model):
    """
    Constructs explainability and diagnostic charts (Beeswarm, PDP/ICE, ROC/PR/Residuals, PCA).
    Surfaces the specific backing model (tree vs DL) clearly in chart titles.
    """
    charts = []
    
    # 1. Feature Importances (Tree model)
    importance_model = rf_classical_model
    model_name_label = "Random Forest approximation" if "Deep Learning" in best_model_name else "Random Forest"
    if "XGBoost" in best_model_name:
        importance_model = best_model_obj
        model_name_label = "XGBoost"
        
    if hasattr(importance_model, 'feature_importances_'):
        importances = importance_model.feature_importances_
        feat_imp = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)[:10]
        charts.append({
            "type": "bar",
            "title": f"Top Feature Importances ({model_name_label})",
            "x_label": "Feature",
            "y_label": "Importance",
            "data": [{"x_val": name, "x_num": None, "y": float(val)} for name, val in feat_imp]
        })
        
    # 2. Target Variable Distribution
    if is_classification:
        val_counts = target_series.value_counts()
        charts.append({
            "type": "bar",
            "title": "Target Class Distribution",
            "x_label": "Class",
            "y_label": "Count",
            "data": [{"x_val": str(k), "x_num": None, "y": float(v)} for k, v in val_counts.items()]
        })
    else:
        hist, bin_edges = np.histogram(target_series, bins=10)
        hist_data = []
        for i in range(len(hist)):
            mid_point = (bin_edges[i] + bin_edges[i+1]) / 2.0
            hist_data.append({"x_val": None, "x_num": float(mid_point), "y": float(hist[i])})
        charts.append({
            "type": "bar",
            "title": "Target Value Distribution",
            "x_label": "Value Range",
            "y_label": "Frequency",
            "data": hist_data
        })
        
    # 3. Predictions vs Actuals
    if is_classification:
        test_df = pd.DataFrame({'True': y_test, 'Predicted': best_preds})
        true_counts = test_df['True'].value_counts().to_dict()
        pred_counts = test_df['Predicted'].value_counts().to_dict()
        all_classes = sorted(list(set(true_counts.keys()).union(set(pred_counts.keys()))))
        
        comparison_data = []
        for cls in all_classes:
            comparison_data.append({"x_val": f"{cls} (True)", "x_num": None, "y": float(true_counts.get(cls, 0))})
            comparison_data.append({"x_val": f"{cls} (Pred)", "x_num": None, "y": float(pred_counts.get(cls, 0))})
        charts.append({
            "type": "bar",
            "title": f"Test Set Predictions vs Actual - {best_model_name}",
            "x_label": "Class Type",
            "y_label": "Count",
            "data": comparison_data
        })
    else:
        scatter_data = []
        sample_size = min(len(y_test), 200)
        if sample_size > 0:
            indices = np.random.choice(len(y_test), sample_size, replace=False)
            for idx in indices:
                scatter_data.append({
                    "x_val": None,
                    "x_num": float(y_test[idx]),
                    "y": float(best_preds[idx])
                })
            charts.append({
                "type": "scatter",
                "title": f"Predicted vs Actual Scatter Plot - {best_model_name} (Test Set)",
                "x_label": "Actual Value",
                "y_label": "Predicted Value",
                "data": scatter_data
            })
            
    # 4. Model Explanations (Permutation Importance, ROC, PR, Residuals)
    try:
        if len(X_test) > 5:
            # D2: Permutation Importance
            from sklearn.inspection import permutation_importance
            pi_result = permutation_importance(
                best_model_obj, X_test, y_test, n_repeats=5, random_state=42
            )
            pi_importances = pi_result.importances_mean
            pi_features = X_train.columns.tolist()
            sorted_pi = sorted(zip(pi_features, pi_importances), key=lambda x: x[1], reverse=True)[:10]
            
            charts.append({
                "type": "bar",
                "title": f"Permutation Feature Importances ({best_model_name})",
                "x_label": "Feature",
                "y_label": "Decrease in Score",
                "data": [{"x_val": name, "x_num": None, "y": float(val)} for name, val in sorted_pi]
            })
            
            # ROC / PR curves (Binary Classification)
            if is_classification and len(np.unique(y_test)) == 2:
                from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
                from sklearn.calibration import calibration_curve
                
                if hasattr(best_model_obj, "predict_proba"):
                    y_probs = best_model_obj.predict_proba(X_test)[:, 1]
                    
                    y_test_for_curves = y_test
                    if hasattr(best_model_obj, "classes_") and len(best_model_obj.classes_) > 0:
                        is_model_numeric = isinstance(best_model_obj.classes_[0], (int, np.integer, float, np.floating))
                        is_test_numeric = isinstance(y_test[0], (int, np.integer, float, np.floating)) if len(y_test) > 0 else True
                        if is_model_numeric and not is_test_numeric and le is not None:
                            y_test_for_curves = le.transform(y_test)
                            
                    pos_label = best_model_obj.classes_[1] if hasattr(best_model_obj, "classes_") else np.unique(y_test_for_curves)[1]
                    y_test_bin = (y_test_for_curves == pos_label).astype(int)
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test_bin, y_probs)
                    roc_auc = auc(fpr, tpr)
                    step = max(1, len(fpr) // 100)
                    roc_data = []
                    for idx in range(0, len(fpr), step):
                        roc_data.append({"x_val": None, "x_num": float(fpr[idx]), "y": float(tpr[idx]), "series": f"Model ({best_model_name})"})
                    roc_data.append({"x_val": None, "x_num": 1.0, "y": 1.0, "series": f"Model ({best_model_name})"})
                    roc_data.append({"x_val": None, "x_num": 0.0, "y": 0.0, "series": "Random Guess"})
                    roc_data.append({"x_val": None, "x_num": 1.0, "y": 1.0, "series": "Random Guess"})
                    
                    charts.append({
                        "type": "line",
                        "title": f"ROC Curve - {best_model_name} (AUC: {roc_auc:.3f})",
                        "x_label": "False Positive Rate",
                        "y_label": "True Positive Rate",
                        "data": roc_data
                    })
                    
                    # PR Curve
                    precision, recall, _ = precision_recall_curve(y_test_bin, y_probs)
                    ap_score = average_precision_score(y_test_bin, y_probs)
                    step_pr = max(1, len(precision) // 100)
                    pr_data = []
                    for idx in range(0, len(precision), step_pr):
                        pr_data.append({"x_val": None, "x_num": float(recall[idx]), "y": float(precision[idx]), "series": f"Model ({best_model_name})"})
                    pr_data.append({"x_val": None, "x_num": 1.0, "y": float(precision[-1]), "series": f"Model ({best_model_name})"})
                    
                    pos_ratio = float(np.sum(y_test_bin == 1) / len(y_test_bin))
                    pr_data.append({"x_val": None, "x_num": 0.0, "y": pos_ratio, "series": "Baseline"})
                    pr_data.append({"x_val": None, "x_num": 1.0, "y": pos_ratio, "series": "Baseline"})
                    
                    charts.append({
                        "type": "line",
                        "title": f"Precision-Recall Curve - {best_model_name} (AP: {ap_score:.3f})",
                        "x_label": "Recall",
                        "y_label": "Precision",
                        "data": pr_data
                    })
                    
                    # Calibration Curve
                    prob_true, prob_pred = calibration_curve(y_test_bin, y_probs, n_bins=10)
                    cal_data = []
                    for idx in range(len(prob_true)):
                        cal_data.append({"x_val": None, "x_num": float(prob_pred[idx]), "y": float(prob_true[idx]), "series": f"Model ({best_model_name})"})
                    cal_data.append({"x_val": None, "x_num": 0.0, "y": 0.0, "series": "Perfect Calibration"})
                    cal_data.append({"x_val": None, "x_num": 1.0, "y": 1.0, "series": "Perfect Calibration"})
                    
                    charts.append({
                        "type": "line",
                        "title": f"Probability Calibration Curve - {best_model_name}",
                        "x_label": "Mean Predicted Probability",
                        "y_label": "Fraction of Positives",
                        "data": cal_data
                    })
                    
            # Residuals Plot (Regression)
            elif not is_classification:
                residuals = y_test - best_preds
                sample_size = min(len(y_test), 300)
                res_data = []
                if sample_size > 0:
                    indices = np.random.choice(len(y_test), sample_size, replace=False)
                    for idx in indices:
                        res_data.append({
                            "x_val": None,
                            "x_num": float(best_preds[idx]),
                            "y": float(residuals[idx]),
                            "series": "Residuals"
                        })
                    charts.append({
                        "type": "scatter",
                        "title": f"Residuals Plot - {best_model_name} (Predicted vs. Error)",
                        "x_label": "Predicted Value",
                        "y_label": "Residual (Actual - Predicted)",
                        "data": res_data
                    })
    except Exception as explanation_err:
        sys.stderr.write(f"Warning: Failed to compute advanced diagnostic charts: {str(explanation_err)}\n")
        
    # 5. SHAP values (fitted on classical Random Forest model for robustTree explainability)
    if rf_classical_model is not None and len(X_test) > 5:
        print_progress(0.78, "Calculating SHAP values for feature importance...")
        try:
            import shap
            explainer = shap.TreeExplainer(rf_classical_model)
            
            shap_samples = X_test
            if len(shap_samples) > 100:
                shap_samples = shap_samples.sample(n=100, random_state=42)
                
            shap_vals = explainer.shap_values(shap_samples)
            
            if isinstance(shap_vals, list):
                shap_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
            elif len(shap_vals.shape) == 3:
                shap_abs = np.mean(np.abs(shap_vals), axis=-1)
            else:
                shap_abs = np.abs(shap_vals)
                
            mean_shap = shap_abs.mean(axis=0)
            feature_names = X_test.columns.tolist()
            shap_importances = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)[:10]
            
            charts.append({
                "type": "bar",
                "title": "Global Feature Importance (SHAP)",
                "x_label": "Feature",
                "y_label": "Mean |SHAP Value|",
                "data": [{"x_val": name, "x_num": None, "y": float(val)} for name, val in shap_importances]
            })
            
            # Beeswarm
            if isinstance(shap_vals, list):
                shap_plot_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
            elif len(shap_vals.shape) == 3:
                shap_plot_vals = shap_vals[:, :, 1] if shap_vals.shape[-1] == 2 else shap_vals[:, :, 0]
            else:
                shap_plot_vals = shap_vals
                
            top_features = [name for name, _ in shap_importances]
            beeswarm_data = []
            for feat in top_features:
                f_idx = feature_names.index(feat)
                col_vals = shap_samples[feat].values
                min_val = col_vals.min()
                max_val = col_vals.max()
                denom = (max_val - min_val) if max_val != min_val else 1.0
                
                for i in range(len(shap_samples)):
                    shap_val = shap_plot_vals[i, f_idx]
                    norm_val = (col_vals[i] - min_val) / denom
                    beeswarm_data.append({
                        "x_val": feat,
                        "x_num": float(shap_val),
                        "y": float(norm_val)
                    })
                    
            charts.append({
                "type": "shap_beeswarm",
                "title": "SHAP Feature Impact (Beeswarm)",
                "x_label": "SHAP Value (Impact on Model Output)",
                "y_label": "Feature",
                "data": beeswarm_data
            })
            
            # PDP & ICE
            for feat in top_features[:3]:
                try:
                    pdp_ice_points = calculate_pdp_ice(
                        rf_classical_model,
                        shap_samples,
                        feat,
                        grid_resolution=20,
                        num_ice_samples=10
                    )
                    if pdp_ice_points:
                        charts.append({
                            "type": "pdp_ice",
                            "title": f"PDP & ICE: {feat} ({model_name_label})",
                            "x_label": f"{feat} Value",
                            "y_label": "Model Prediction / Probability",
                            "data": pdp_ice_points
                        })
                except Exception as pdp_err:
                    sys.stderr.write(f"Warning: PDP/ICE for {feat} failed: {str(pdp_err)}\n")
        except Exception as shap_err:
            sys.stderr.write(f"Warning: SHAP evaluation failed: {str(shap_err)}\n")
            
    # 6. PCA explained variance (if >= 3 numeric features)
    numeric_features = [c for c in numeric_cols if c != target_col]
    if len(numeric_features) >= 3:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            df_pca = X_processed[numeric_features].copy()
            scaler = StandardScaler()
            scaled_feats = scaler.fit_transform(df_pca)
            
            pca = PCA()
            pca.fit(scaled_feats)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            
            pca_data = []
            for idx, val in enumerate(cum_var):
                pca_data.append({
                    "x_val": f"PC {idx+1}",
                    "x_num": None,
                    "y": float(val)
                })
            charts.append({
                "type": "line",
                "title": "PCA Cumulative Explained Variance",
                "x_label": "Principal Components",
                "y_label": "Cumulative Variance Ratio",
                "data": pca_data
            })
        except Exception as pca_err:
            sys.stderr.write(f"PCA error: {str(pca_err)}\n")
            
    return charts

def analyze_tabular(df, target_col, task_type_override,
                    row_count, col_count, columns, full_preview, missing,
                    numeric_cols, categorical_cols,
                    file_path=None, model_export_path=None, code_export_path=None,
                    smart_sample=False, cleaning_actions=None,
                    test_df=None, val_df=None, has_test_set=False, has_val_set=False,
                    test_info=None, val_info=None, cleaner=None, feature_selection=False):
    """
    Main entry point for the Tabular pipeline analysis. Handles preprocessing,
    modeling, metrics evaluation, charts construction, and code export.
    """
    try:
        # Determine classification or regression
        is_classification = False
        target_series = df[target_col].dropna()
        unique_targets = target_series.nunique()
        
        if task_type_override and task_type_override != "auto":
            is_classification = (task_type_override == "classification")
        else:
            if target_col in categorical_cols:
                is_classification = True
            elif unique_targets <= 10:
                dtype = target_series.dtype
                if pd.api.types.is_integer_dtype(dtype):
                    vals = sorted(target_series.unique().tolist())
                    is_contiguous = (vals == list(range(int(vals[0]), int(vals[-1]) + 1)))
                    if is_contiguous and int(vals[0]) in [0, 1]:
                        is_classification = True
                else:
                    is_classification = True
                    
        task_type = "classification" if is_classification else "regression"
        
        # Prepare targets
        df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)
        if df_clean.empty:
            return {"error": f"No rows left after dropping records where target '{target_col}' is missing."}
            
        y_raw = df_clean[target_col]
        if is_classification:
            y = y_raw.fillna(y_raw.mode()[0] if not y_raw.mode().empty else "missing").astype(str).to_numpy()
        else:
            y = y_raw.fillna(y_raw.median()).to_numpy()
            
        X = df_clean.drop(columns=[target_col])
        
        # Identify text columns and columns to drop
        cols_to_drop = []
        text_cols = []
        for col in X.columns:
            if col == "__is_test":
                continue
            nunique = X[col].nunique()
            if nunique == 0:
                cols_to_drop.append(col)
            elif col in categorical_cols:
                sample_series = X[col].dropna().astype(str)
                if sample_series.empty:
                    cols_to_drop.append(col)
                    continue
                avg_len = sample_series.str.len().mean()
                col_lower = col.lower()
                is_id = any(kw in col_lower for kw in ["id", "uuid", "key", "index", "code", "url", "link"])
                if nunique > 100:
                    if not is_id:
                        text_cols.append(col)
                    else:
                        cols_to_drop.append(col)
                        
        X = X.drop(columns=cols_to_drop)
        
        # Prepare list of standard categorical / target encode categories
        X_categorical = [c for c in X.columns if c in categorical_cols and c not in text_cols and c != "__is_test"]
        target_encode_cols = []
        one_hot_cols = []
        
        actions_list = []
        if cleaning_actions:
            try:
                actions_list = json.loads(cleaning_actions)
            except Exception:
                pass
                
        for col in X_categorical:
            is_target_encoded = any(act.get("column") == col and act.get("actionType") == "target_encode" for act in actions_list)
            if is_target_encoded:
                target_encode_cols.append(col)
            else:
                one_hot_cols.append(col)
                
        # Split train/test
        X_val_split, y_val_split = None, None
        
        if "__is_test" in X.columns:
            train_mask = (X["__is_test"] == 0).to_numpy()
            test_mask = (X["__is_test"] == 1).to_numpy()
            val_mask = (X["__is_test"] == 2).to_numpy()
            
            X_train_full = X[train_mask].drop(columns=["__is_test"], errors='ignore')
            y_train_full = y[train_mask]
            
            if np.any(val_mask):
                X_val_split = X[val_mask].drop(columns=["__is_test"], errors='ignore')
                y_val_split = y[val_mask]
                
            if np.any(test_mask):
                X_train = X_train_full
                y_train = y_train_full
                X_test = X[test_mask].drop(columns=["__is_test"], errors='ignore')
                y_test = y[test_mask]
            else:
                use_stratify = False
                if is_classification:
                    from collections import Counter
                    counts = Counter(y_train_full)
                    if counts and min(counts.values()) >= 2:
                        use_stratify = True
                        
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train_full, y_train_full, test_size=0.2, random_state=42,
                    stratify=y_train_full if use_stratify else None
                )
        else:
            X_train_full = X
            y_train_full = y
            use_stratify = False
            if is_classification:
                from collections import Counter
                counts = Counter(y)
                if counts and min(counts.values()) >= 2:
                    use_stratify = True
                    
            if use_stratify:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
        # Cap sizes
        if len(X_train) > 10000:
            np.random.seed(42)
            indices = np.random.choice(len(X_train), 10000, replace=False)
            X_train = X_train.iloc[indices]
            y_train = y_train[indices]
        if len(X_test) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(X_test), 5000, replace=False)
            X_test = X_test.iloc[indices]
            y_test = y_test[indices]
        if X_val_split is not None and len(X_val_split) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(X_val_split), 5000, replace=False)
            X_val_split = X_val_split.iloc[indices]
            y_val_split = y_val_split[indices]
            
        # Run preprocessing
        print_progress(0.42, "Preprocessing features & imputing values...")
        X_train_proc, X_test_proc, X_val_proc, preprocessor = preprocess(
            X_train, y_train, X_test, y_test, X_val_split, y_val_split,
            categorical_cols, numeric_cols, text_cols, target_encode_cols, one_hot_cols, is_classification
        )
        
        # Globally preprocessed full X for PCA / SHAP / etc
        X_proc_full, _, _, _ = preprocess(
            X_train_full, y_train_full, None, None, None, None,
            categorical_cols, numeric_cols, text_cols, target_encode_cols, one_hot_cols, is_classification
        )
        
        if feature_selection:
            print_progress(0.50, "Running Recursive Feature Elimination (RFE) for feature selection...")
            n_features = X_train_proc.shape[1]
            n_to_select = max(10, int(n_features * 0.5))
            if n_to_select < n_features:
                from sklearn.feature_selection import RFE
                from sklearn.preprocessing import LabelEncoder
                if is_classification:
                    estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                    le_temp = LabelEncoder()
                    y_train_fit = le_temp.fit_transform(y_train)
                else:
                    estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                    y_train_fit = y_train
                
                rfe = RFE(estimator=estimator, n_features_to_select=n_to_select, step=0.2)
                rfe.fit(X_train_proc, y_train_fit)
                
                selected_cols = X_train_proc.columns[rfe.support_]
                X_train_proc = X_train_proc[selected_cols]
                if X_test_proc is not None:
                    X_test_proc = X_test_proc[selected_cols]
                if X_val_proc is not None:
                    X_val_proc = X_val_proc[selected_cols]
                if X_proc_full is not None:
                    X_proc_full = X_proc_full[selected_cols]
                
                sys.stderr.write(f"RFE selected {len(selected_cols)} features out of {n_features}: {list(selected_cols)}\n")
        
        # Fit models
        best_model_name, best_model_obj, models_compared, best_preds, rf_classical_model, le = train_models(
            X_train_proc, y_train, X_test_proc, y_test, is_classification
        )
        
        # Compute metrics
        cv_scores, cv_mean, cv_std, dummy_score, confusion_matrix_data, val_metrics, val_confusion_matrix_data = compute_metrics(
            best_model_name, best_model_obj, X_train_proc, y_train, X_test_proc, y_test, X_val_proc, y_val_split,
            is_classification, le, X_proc_full, y, X_train_full, y_train_full,
            categorical_cols, numeric_cols, text_cols, target_encode_cols, one_hot_cols
        )
        
        # Build charts
        charts = build_charts(
            best_model_name, best_model_obj, X_train_proc, y_train, X_test_proc, y_test,
            is_classification, le, target_series, target_col, numeric_cols,
            categorical_cols, X_proc_full, best_preds, rf_classical_model
        )
        
        # Final primary score dictionary
        if is_classification:
            metrics_dict = {
                "model": best_model_name,
                "score_type": "Accuracy",
                "score": float(accuracy_score(y_test, best_preds)),
                "additional_metrics": {
                    "F1 Score": float(f1_score(y_test, best_preds, average='weighted', zero_division=0))
                }
            }
        else:
            test_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
            metrics_dict = {
                "model": best_model_name,
                "score_type": "R² Score",
                "score": float(r2_score(y_test, best_preds)),
                "additional_metrics": {
                    "RMSE": float(test_rmse)
                }
            }
            
        # Re-build summary text
        summary_sections = []
        overview = f"### 📊 Dataset Overview\n"
        overview += f"- **Rows:** {row_count:,} | **Columns:** {col_count:,}\n"
        overview += f"- **Column Types:** {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(text_cols)} text\n"
        overview += f"- **Missing Value Cells:** {sum(missing.values()):,} total across {len([k for k,v in missing.items() if v > 0])} columns."
        summary_sections.append(overview)
        
        # DQ alerts
        dq_alerts = []
        for col in numeric_cols:
            try:
                sk = df_clean[col].skew()
                if abs(sk) > 1.5:
                    dq_alerts.append(f"- **Highly Skewed Column:** `{col}` has a skewness of `{sk:.2f}` (consider Log/Power transform).")
            except Exception:
                pass
                
            try:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                outliers_count = df_clean[(df_clean[col] < q1 - 1.5 * iqr) | (df_clean[col] > q3 + 1.5 * iqr)][col].count()
                if outliers_count > 0:
                    pct = (outliers_count / row_count) * 100
                    if pct > 1.0:
                        dq_alerts.append(f"- **Outliers Detected:** `{col}` contains `{outliers_count:,}` outliers (`{pct:.1f}%` of values).")
            except Exception:
                pass
                
        for col, miss_val in missing.items():
            miss_pct = (miss_val / row_count) * 100
            if miss_pct > 30.0:
                dq_alerts.append(f"- **High Missingness:** `{col}` has `{miss_pct:.1f}%` missing values (imputation may be unreliable).")
                
        if dq_alerts:
            summary_sections.append("### ⚠️ Data Quality & Warnings\n" + "\n".join(dq_alerts))
            
        target_info = f"### 🎯 Target Variable Analysis (`{target_col}`)\n"
        target_info += f"- **Task Type:** {task_type.capitalize()}\n"
        if is_classification:
            target_info += f"- **Unique Classes:** {unique_targets}\n"
            class_counts = target_series.value_counts(normalize=True)
            if not class_counts.empty:
                target_info += f"- **Majority Class:** `{class_counts.index[0]}` represents `{class_counts.iloc[0]*100:.1f}%` of the labels."
                if class_counts.max() > 0.7:
                    target_info += " **(Highly Imbalanced)**"
        else:
            target_info += f"- **Range:** `{target_series.min():.2f}` to `{target_series.max():.2f}` (Mean: `{target_series.mean():.2f}`, Median: `{target_series.median():.2f}`)"
        summary_sections.append(target_info)
        
        model_perf = f"### 🤖 Machine Learning Model Performance\n"
        model_perf += f"- **Best Model:** `{best_model_name}`\n"
        model_perf += f"- **Primary Score ({metrics_dict['score_type']}):** `{metrics_dict['score']:.4f}`\n"
        if 'additional_metrics' in metrics_dict and metrics_dict['additional_metrics']:
            for am_name, am_val in metrics_dict['additional_metrics'].items():
                model_perf += f"- **{am_name}:** `{am_val:.4f}`"
        summary_sections.append(model_perf)
        
        summary = "\n\n".join(summary_sections)
        
        print_progress(0.88, "Profiling columns & generating data statistics...")
        profiling = profile_dataset(df)
        
        # Target leakage detections
        data_leakage_warnings = []
        if target_col in df.columns:
            target_lower = target_col.lower()
            for col in df.columns:
                if col != target_col:
                    if target_lower in col.lower() and not (col.lower().startswith("is_") or col.lower().startswith("has_")):
                        data_leakage_warnings.append(
                            f"Column '{col}' name contains target name '{target_col}' (possible label encoding/leakage)."
                        )
                    if col in numeric_cols and target_col in numeric_cols:
                        try:
                            corr = df[col].corr(df[target_col])
                            if abs(corr) >= 0.98:
                                data_leakage_warnings.append(
                                    f"Column '{col}' has a correlation of {corr:.3f} with target '{target_col}' (highly likely target leakage)."
                                )
                        except Exception:
                            pass
                            
        # Recommendations
        cleaning_recommendations = []
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            if missing_pct >= 0.5:
                cleaning_recommendations.append({
                    "column": col,
                    "issue": f"{missing_pct*100:.1f}% missing values",
                    "recommendation": "Drop this column as it contains mostly nulls.",
                    "impact": "High"
                })
            elif missing_pct > 0.0:
                cleaning_recommendations.append({
                    "column": col,
                    "issue": f"{missing_pct*100:.1f}% missing values",
                    "recommendation": "Impute missing values using median (numeric) or mode (categorical).",
                    "impact": "Low"
                })
            if col != target_col and col in categorical_cols:
                nunique = df[col].nunique()
                if nunique > 100:
                    cleaning_recommendations.append({
                        "column": col,
                        "issue": f"High cardinality categorical column ({nunique} unique categories)",
                        "recommendation": "Group rare categories or drop if not highly predictive.",
                        "impact": "Medium"
                    })
            if col != target_col:
                nunique = df[col].nunique()
                if nunique == 1:
                    cleaning_recommendations.append({
                        "column": col,
                        "issue": "Constant value column (only 1 unique value)",
                        "recommendation": "Drop this column as it has zero variance.",
                        "impact": "Medium"
                    })
                elif nunique == 2 and col in numeric_cols:
                    val_counts = df[col].value_counts(normalize=True)
                    if val_counts.max() > 0.99:
                        cleaning_recommendations.append({
                            "column": col,
                            "issue": f"Near-constant column ({val_counts.max()*100:.1f}% identical values)",
                            "recommendation": "Consider dropping this column.",
                            "impact": "Low"
                        })
            if col != target_col and col in numeric_cols:
                try:
                    q25 = df[col].quantile(0.25)
                    q75 = df[col].quantile(0.75)
                    iqr = q75 - q25
                    if iqr > 0:
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                        outlier_pct = len(outliers) / len(df)
                        if outlier_pct > 0.05:
                            cleaning_recommendations.append({
                                "column": col,
                                "issue": f"Outliers detected ({outlier_pct*100:.1f}% of values outside whiskers)",
                                "recommendation": "Apply outlier capping/clipping or log transformation.",
                                "impact": "Medium"
                            })
                except Exception:
                    pass
                    
        # Export model and code
        if model_export_path or code_export_path:
            model_to_save = best_model_obj
            feature_names = list(X_train_proc.columns)
            _export_model_and_code(
                model_to_save, model_export_path, code_export_path,
                file_path, "tabular", target_col, None,
                "classification" if is_classification else "regression",
                feature_names, best_model_name, numeric_cols, categorical_cols, text_cols,
                cleaner=cleaner, preprocessor=preprocessor, label_encoder=le
            )
            
        result = {
            "summary": summary,
            "columns": columns,
            "row_count": int(row_count),
            "col_count": int(col_count),
            "original_row_count": int(row_count),
            "sampled_row_count": int(row_count),
            "task_type": task_type,
            "numeric_col_count": len(numeric_cols),
            "categorical_col_count": len(categorical_cols),
            "text_col_count": len(text_cols),
            "missing_values": {k: int(v) for k, v in missing.items()},
            "correlations": [], # computed in analyze.py main if needed, or empty
            "charts": charts,
            "metrics": metrics_dict,
            "val_metrics": val_metrics,
            "val_confusion_matrix": val_confusion_matrix_data,
            "models_compared": models_compared,
            "target_column": target_col,
            "full_preview": full_preview,
            "dummy_baseline_score": dummy_score,
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "confusion_matrix": confusion_matrix_data,
            "profiling": profiling,
            "data_leakage_warnings": data_leakage_warnings,
            "cleaning_recommendations": cleaning_recommendations,
            "error": None
        }
        if test_info:
            result.update(test_info)
        if val_info:
            result.update(val_info)
        return result
        
    except Exception as e:
        import traceback
        return {"error": f"An error occurred during Tabular execution: {str(e)}\n{traceback.format_exc()}"}
