import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor

# Ensure local import paths work correctly in both dev and Xcode compiled bundle contexts
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# If running in a flattened Xcode bundle where "utils" or "pipelines" subfolders don't exist,
# we dynamically create namespace packages and map them in sys.modules to route imports correctly.
if not os.path.isdir(os.path.join(script_dir, "utils")) and os.path.exists(os.path.join(script_dir, "helpers.py")):
    import types
    # 1. Register namespace package 'utils'
    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = []
    sys.modules["utils"] = utils_mod
    
    # 2. Load and register helpers first (since loader, profiler, charts depend on it)
    import helpers
    sys.modules["utils.helpers"] = helpers
    utils_mod.helpers = helpers
    
    # 3. Load others sequentially
    import loader
    sys.modules["utils.loader"] = loader
    utils_mod.loader = loader
    
    import profiler
    sys.modules["utils.profiler"] = profiler
    utils_mod.profiler = profiler
    
    import charts
    sys.modules["utils.charts"] = charts
    utils_mod.charts = charts

if not os.path.isdir(os.path.join(script_dir, "pipelines")) and os.path.exists(os.path.join(script_dir, "timeseries.py")):
    import types
    # 1. Register namespace package 'pipelines'
    pipelines_mod = types.ModuleType("pipelines")
    pipelines_mod.__path__ = []
    sys.modules["pipelines"] = pipelines_mod
    
    # 2. Load and register pipelines sequentially
    import timeseries
    sys.modules["pipelines.timeseries"] = timeseries
    pipelines_mod.timeseries = timeseries
    
    import image
    sys.modules["pipelines.image"] = image
    pipelines_mod.image = image
    
    import nlp
    sys.modules["pipelines.nlp"] = nlp
    pipelines_mod.nlp = nlp
    
    import preview
    sys.modules["pipelines.preview"] = preview
    pipelines_mod.preview = preview

# Import submodules
from utils.helpers import print_progress, clean_nan, _generate_reproduction_code, _export_model_and_code
from utils.loader import download_dataset, load_dataset, _infer_dataset_type
from utils.profiler import profile_dataset
from utils.charts import generate_boxplots, load_images_from_tabular, get_image_preview

from pipelines.timeseries import analyze_timeseries
from pipelines.nlp import analyze_nlp
from pipelines.image import analyze_image
from pipelines.preview import analyze_preview

def analyze(file_path, target_col=None, dataset_type="tabular",
            task_type_override="auto", time_col=None, exclude_cols=None, test_file_path=None, val_file_path=None,
            model_export_path=None, code_export_path=None, smart_sample=False, cleaning_actions=None):
    if exclude_cols is None:
        exclude_cols = set()
    try:
        # Check if URL input
        if file_path.startswith("http://") or file_path.startswith("https://"):
            try:
                print_progress(0.05, "Downloading remote dataset...")
                file_path = download_dataset(file_path)
            except Exception as download_err:
                return {"error": f"Failed to download dataset: {str(download_err)}"}

        if dataset_type == "image":
            return analyze_image(file_path, task_type_override, target_col, test_file_path=test_file_path, model_export_path=model_export_path, code_export_path=code_export_path)

        print_progress(0.15, "Loading dataset file...")
        df = load_dataset(file_path)

        if df.empty:
            return {"error": "The dataset is empty."}

        original_row_count = len(df)
        sampled_row_count = None

        # Apply column exclusion BEFORE any analysis
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
            print_progress(0.17, f"Excluded columns: {list(exclude_cols)}...")

        # Apply cleaning actions (Phase 3)
        if cleaning_actions:
            try:
                import json
                actions = json.loads(cleaning_actions)
                for act in actions:
                    col = act.get("column")
                    act_type = act.get("actionType")
                    if col in df.columns:
                        if act_type == "drop":
                            df = df.drop(columns=[col])
                        elif act_type == "impute_mean":
                            if pd.api.types.is_numeric_dtype(df[col].dtype):
                                df[col] = df[col].fillna(df[col].mean())
                        elif act_type == "impute_median":
                            if pd.api.types.is_numeric_dtype(df[col].dtype):
                                df[col] = df[col].fillna(df[col].median())
                        elif act_type == "impute_mode":
                            mode_val = df[col].mode()
                            if not mode_val.empty:
                                df[col] = df[col].fillna(mode_val[0])
                        elif act_type == "clip_outliers":
                            if pd.api.types.is_numeric_dtype(df[col].dtype):
                                q25 = df[col].quantile(0.25)
                                q75 = df[col].quantile(0.75)
                                iqr = q75 - q25
                                lower = q25 - 1.5 * iqr
                                upper = q75 + 1.5 * iqr
                                df[col] = df[col].clip(lower, upper)
                print_progress(0.173, "Applied interactive cleaning recommendations...")
            except Exception as clean_err:
                sys.stderr.write(f"Warning: Failed to apply cleaning actions: {str(clean_err)}\n")

        # Smart Sampling (Phase 2)
        if smart_sample and original_row_count > 100000:
            print_progress(0.175, f"Smart sampling dataset from {original_row_count} to 100,000 rows...")
            stratify_col = None
            if target_col and target_col in df.columns:
                y_series = df[target_col].dropna()
                if y_series.nunique() > 1 and (y_series.nunique() < 50 or y_series.dtype == object or y_series.dtype == bool):
                    from collections import Counter
                    counts = Counter(y_series)
                    if min(counts.values()) >= 2:
                        stratify_col = target_col
            
            if stratify_col:
                try:
                    from sklearn.model_selection import train_test_split
                    df, _ = train_test_split(df, train_size=100000, random_state=42, stratify=df[stratify_col])
                except Exception:
                    df = df.sample(n=100000, random_state=42)
            else:
                df = df.sample(n=100000, random_state=42)
            df = df.reset_index(drop=True)
            sampled_row_count = len(df)

        # Load separate test set if provided
        test_df = None
        has_test_set = False
        if test_file_path and os.path.exists(test_file_path):
            try:
                print_progress(0.18, "Loading separate test dataset...")
                test_df = load_dataset(test_file_path)
                if test_df is not None and not test_df.empty:
                    if exclude_cols:
                        test_df = test_df.drop(columns=[c for c in exclude_cols if c in test_df.columns], errors='ignore')
                    # Apply same cleaning actions to test set
                    if cleaning_actions and not test_df.empty:
                        for act in actions:
                            col = act.get("column")
                            act_type = act.get("actionType")
                            if col in test_df.columns:
                                if act_type == "drop":
                                    test_df = test_df.drop(columns=[col])
                                elif act_type == "impute_mean":
                                    if pd.api.types.is_numeric_dtype(test_df[col].dtype):
                                        test_df[col] = test_df[col].fillna(test_df[col].mean())
                                elif act_type == "impute_median":
                                    if pd.api.types.is_numeric_dtype(test_df[col].dtype):
                                        test_df[col] = test_df[col].fillna(test_df[col].median())
                                elif act_type == "impute_mode":
                                    mode_val = test_df[col].mode()
                                    if not mode_val.empty:
                                        test_df[col] = test_df[col].fillna(mode_val[0])
                                elif act_type == "clip_outliers":
                                    if pd.api.types.is_numeric_dtype(test_df[col].dtype):
                                        q25 = test_df[col].quantile(0.25)
                                        q75 = test_df[col].quantile(0.75)
                                        iqr = q75 - q25
                                        lower = q25 - 1.5 * iqr
                                        upper = q75 + 1.5 * iqr
                                        test_df[col] = test_df[col].clip(lower, upper)
                    # Apply smart sampling to test set
                    if smart_sample and len(test_df) > 100000:
                        test_df = test_df.sample(n=100000, random_state=42).reset_index(drop=True)
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to load/preprocess test dataset: {str(e)}\n")

        # Load separate validation set if provided
        val_df = None
        has_val_set = False
        if val_file_path and os.path.exists(val_file_path):
            try:
                print_progress(0.19, "Loading separate validation dataset...")
                val_df = load_dataset(val_file_path)
                if val_df is not None and not val_df.empty:
                    if exclude_cols:
                        val_df = val_df.drop(columns=[c for c in exclude_cols if c in val_df.columns], errors='ignore')
                    # Apply same cleaning actions to validation set
                    if cleaning_actions and not val_df.empty:
                        for act in actions:
                            col = act.get("column")
                            act_type = act.get("actionType")
                            if col in val_df.columns:
                                if act_type == "drop":
                                    val_df = val_df.drop(columns=[col])
                                elif act_type == "impute_mean":
                                    if pd.api.types.is_numeric_dtype(val_df[col].dtype):
                                        val_df[col] = val_df[col].fillna(val_df[col].mean())
                                elif act_type == "impute_median":
                                    if pd.api.types.is_numeric_dtype(val_df[col].dtype):
                                        val_df[col] = val_df[col].fillna(val_df[col].median())
                                elif act_type == "impute_mode":
                                    mode_val = val_df[col].mode()
                                    if not mode_val.empty:
                                        val_df[col] = val_df[col].fillna(mode_val[0])
                                elif act_type == "clip_outliers":
                                    if pd.api.types.is_numeric_dtype(val_df[col].dtype):
                                        q25 = val_df[col].quantile(0.25)
                                        q75 = val_df[col].quantile(0.75)
                                        iqr = q75 - q25
                                        lower = q25 - 1.5 * iqr
                                        upper = q75 + 1.5 * iqr
                                        val_df[col] = val_df[col].clip(lower, upper)
                    # Apply smart sampling to validation set
                    if smart_sample and len(val_df) > 100000:
                        val_df = val_df.sample(n=100000, random_state=42).reset_index(drop=True)
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to load/preprocess validation dataset: {str(e)}\n")

        print_progress(0.25, "Analyzing columns and missing values...")
        row_count, col_count = df.shape
        columns = list(df.columns)

        # Capture full preview for the Data tab (up to 500 rows, convert to strings)
        preview_df = df.head(500).fillna("").astype(str)
        full_preview = {
            "columns": list(preview_df.columns),
            "rows": preview_df.values.tolist(),
            "total_rows": int(row_count),
        }

        # Missing values
        missing = df.isnull().sum().to_dict()

        # Identify numeric, categorical, and datetime columns
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        for col in columns:
            col_series = df[col]
            
            is_datetime = False
            if pd.api.types.is_datetime64_any_dtype(col_series.dtype):
                is_datetime = True
            else:
                col_lower = col.lower()
                date_keywords = ["date", "time", "timestamp", "datetime", "year", "month", "day"]
                if any(kw in col_lower for kw in date_keywords):
                    try:
                        non_nulls = col_series.dropna()
                        if not non_nulls.empty:
                            sample = non_nulls.head(100)
                            parsed = pd.to_datetime(sample, errors='coerce')
                            if parsed.notnull().sum() / len(sample) >= 0.8:
                                is_datetime = True
                    except Exception:
                        pass
            
            if is_datetime:
                datetime_cols.append(col)
                continue
                
            is_raw_num = pd.api.types.is_numeric_dtype(col_series.dtype)
            is_categorical_num = False
            if is_raw_num:
                nunique = col_series.nunique()
                col_lower = col.lower()
                non_nulls = col_series.dropna()
                is_integer_like = False
                if pd.api.types.is_integer_dtype(col_series.dtype):
                    is_integer_like = True
                elif not non_nulls.empty:
                    try:
                        is_integer_like = non_nulls.apply(lambda x: float(x).is_integer()).all()
                    except Exception:
                        pass
                
                cat_keywords = ["class", "label", "target", "category", "group", "outcome", "gender", "sex", "style", "type", "status", "state", "mode", "stage", "grade", "phase", "tier", "priority", "choice", "option", "flag", "indicator", "ind", "is_"]
                has_cat_name = any(kw in col_lower for kw in cat_keywords)
                
                continuous_keywords = ["age", "price", "amount", "score", "value", "val", "count", "num", "temp", "rate", "pct", "percent", "date", "year", "month", "day", "time"]
                has_cont_name = any(kw in col_lower for kw in continuous_keywords)
                
                if nunique <= 2:
                    is_categorical_num = True
                elif nunique <= 15 and has_cat_name:
                    is_categorical_num = True
                elif nunique <= 5 and is_integer_like and not has_cont_name:
                    is_categorical_num = True
            
            if is_raw_num and not is_categorical_num:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Target columns resolution
        target_cols = []
        if target_col:
            if "," in target_col:
                target_cols = [t.strip() for t in target_col.split(",") if t.strip()]
            else:
                target_cols = [target_col]
        else:
            target_keywords = ["target", "label", "class", "price", "outcome", "y", "survived"]
            for col in columns:
                if any(kw in col.lower() for kw in target_keywords):
                    target_col = col
                    break
            if not target_col:
                target_col = columns[-1] # Default to last column
            target_cols = [target_col]

        # Identify identifier columns to exclude them from modeling features, correlations, and analytics
        identifier_cols = []
        for col in columns:
            if col in target_cols:
                continue
            col_lower = col.lower()
            col_series = df[col].dropna()
            non_null_count = len(col_series)
            if non_null_count > 0:
                nunique = df[col].nunique()
                is_unique_key = nunique == non_null_count or (nunique / non_null_count) >= 0.98
                is_id_name = col_lower in ["id", "index", "no", "number", "num", "row", "rowid"] or \
                             col_lower.endswith("_id") or col_lower.endswith("id") or col_lower.startswith("id_")
                             
                if is_unique_key and (is_id_name or col_series.dtype == object):
                    identifier_cols.append(col)
                    
        # Exclude identifier columns from training features, correlations, and plots
        numeric_cols = [c for c in numeric_cols if c not in identifier_cols]
        categorical_cols = [c for c in categorical_cols if c not in identifier_cols]

        for t in target_cols:
            if t not in df.columns:
                return {"error": f"Target column '{t}' not found in the dataset."}

        # Check separate test set validity
        has_test_set = False
        if test_df is not None and not test_df.empty:
            has_test_set = True
            for t in target_cols:
                col_mapping = {c.lower(): c for c in test_df.columns}
                if t in test_df.columns:
                    pass
                elif t.lower() in col_mapping:
                    test_df = test_df.rename(columns={col_mapping[t.lower()]: t})
                else:
                    sys.stderr.write(f"Warning: Target column '{t}' not found in test dataset. Falling back to train-test split.\n")
                    has_test_set = False

        # Check separate validation set validity
        has_val_set = False
        if val_df is not None and not val_df.empty:
            has_val_set = True
            for t in target_cols:
                col_mapping = {c.lower(): c for c in val_df.columns}
                if t in val_df.columns:
                    pass
                elif t.lower() in col_mapping:
                    val_df = val_df.rename(columns={col_mapping[t.lower()]: t})
                else:
                    sys.stderr.write(f"Warning: Target column '{t}' not found in validation dataset. Ignoring validation set.\n")
                    has_val_set = False

        # Compute test set profiling & statistics if loaded
        test_info = {}
        if has_test_set:
            try:
                t_row_count, t_col_count = test_df.shape
                t_missing = test_df.isnull().sum().to_dict()
                
                # Compute test correlations for numeric cols
                t_numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
                t_correlations = []
                if len(t_numeric_cols) > 1:
                    t_corr_matrix = test_df[t_numeric_cols].corr()
                    for i in range(len(t_numeric_cols)):
                        for j in range(i + 1, len(t_numeric_cols)):
                            val = t_corr_matrix.iloc[i, j]
                            if not np.isnan(val):
                                t_correlations.append({
                                    "x": t_numeric_cols[i],
                                    "y": t_numeric_cols[j],
                                    "value": float(val)
                                })
                    t_correlations.sort(key=lambda x: abs(x["value"]), reverse=True)
                
                # Profiling
                t_profiling = profile_dataset(test_df)
                
                # Full preview
                t_preview_df = test_df.head(500).fillna("").astype(str)
                t_full_preview = {
                    "columns": list(t_preview_df.columns),
                    "rows": t_preview_df.values.tolist(),
                    "total_rows": int(t_row_count)
                }
                
                test_info = {
                    "test_row_count": int(t_row_count),
                    "test_col_count": int(t_col_count),
                    "test_missing_values": {k: int(v) for k, v in t_missing.items()},
                    "test_correlations": t_correlations,
                    "test_profiling": t_profiling,
                    "test_full_preview": t_full_preview
                }
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to profile test dataset: {str(e)}\n")

        # Compute validation set profiling & statistics if loaded
        val_info = {}
        if has_val_set:
            try:
                v_row_count, v_col_count = val_df.shape
                v_missing = val_df.isnull().sum().to_dict()
                
                # Compute validation correlations for numeric cols
                v_numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
                v_correlations = []
                if len(v_numeric_cols) > 1:
                    v_corr_matrix = val_df[v_numeric_cols].corr()
                    for i in range(len(v_numeric_cols)):
                        for j in range(i + 1, len(v_numeric_cols)):
                            val = v_corr_matrix.iloc[i, j]
                            if not np.isnan(val):
                                v_correlations.append({
                                    "x": v_numeric_cols[i],
                                    "y": v_numeric_cols[j],
                                    "value": float(val)
                                })
                    v_correlations.sort(key=lambda x: abs(x["value"]), reverse=True)
                
                # Profiling
                v_profiling = profile_dataset(val_df)
                
                # Full preview
                v_preview_df = val_df.head(500).fillna("").astype(str)
                v_full_preview = {
                    "columns": list(v_preview_df.columns),
                    "rows": v_preview_df.values.tolist(),
                    "total_rows": int(v_row_count)
                }
                
                val_info = {
                    "val_row_count": int(v_row_count),
                    "val_col_count": int(v_col_count),
                    "val_missing_values": {k: int(v) for k, v in v_missing.items()},
                    "val_correlations": v_correlations,
                    "val_profiling": v_profiling,
                    "val_full_preview": v_full_preview
                }
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to profile validation dataset: {str(e)}\n")

        # Combine datasets if separate test or validation set is active
        if has_test_set or has_val_set:
            df_train_temp = df.copy()
            df_train_temp["__is_test"] = 0
            parts = [df_train_temp]
            
            if has_test_set:
                df_test_temp = test_df.copy()
                df_test_temp["__is_test"] = 1
                for col in df_train_temp.columns:
                    if col not in df_test_temp.columns:
                        df_test_temp[col] = np.nan
                df_test_temp = df_test_temp[df_train_temp.columns]
                parts.append(df_test_temp)
                
            if has_val_set:
                df_val_temp = val_df.copy()
                df_val_temp["__is_test"] = 2
                for col in df_train_temp.columns:
                    if col not in df_val_temp.columns:
                        df_val_temp[col] = np.nan
                df_val_temp = df_val_temp[df_train_temp.columns]
                parts.append(df_val_temp)
                
            combined_df = pd.concat(parts, axis=0, ignore_index=True)
        else:
            combined_df = df

        if dataset_type == "timeseries":
            targets_result = {}
            first_target_res = None
            for t in target_cols:
                is_first = (t == target_cols[0])
                print_progress(0.35 + 0.5 * (target_cols.index(t) / len(target_cols)), f"Analyzing time series for target {t}...")
                
                m_path = model_export_path if is_first else None
                c_path = code_export_path if is_first else None
                
                t_res = analyze_timeseries(
                    combined_df, t, time_col, task_type_override,
                    row_count, col_count, columns, full_preview, missing,
                    numeric_cols, categorical_cols,
                    file_path=file_path, model_export_path=m_path, code_export_path=c_path
                )
                if is_first:
                    first_target_res = t_res
                targets_result[t] = {
                    "metrics": t_res.get("metrics"),
                    "models_compared": t_res.get("models_compared"),
                    "charts": t_res.get("charts"),
                    "dummy_baseline_score": t_res.get("dummy_baseline_score"),
                    "cv_scores": t_res.get("cv_scores"),
                    "cv_mean": t_res.get("cv_mean"),
                    "cv_std": t_res.get("cv_std"),
                    "confusion_matrix": t_res.get("confusion_matrix"),
                    "val_metrics": t_res.get("val_metrics"),
                    "val_confusion_matrix": t_res.get("val_confusion_matrix")
                }
            res = first_target_res.copy() if first_target_res else {}
            res["targets"] = targets_result
            res.update(test_info)
            res.update(val_info)
            return res

        if dataset_type == "nlp":
            res = analyze_nlp(
                combined_df, target_col, task_type_override,
                row_count, col_count, columns, full_preview, missing,
                numeric_cols, categorical_cols,
                file_path=file_path, model_export_path=model_export_path, code_export_path=code_export_path
            )
            res.update(test_info)
            res.update(val_info)
            return res

        # Determine classification or regression
        is_classification = False
        target_series = df[target_col].dropna()
        unique_targets = target_series.nunique()

        if task_type_override and task_type_override != "auto":
            # User explicitly chose — always respect it
            is_classification = (task_type_override == "classification")
        else:
            # Heuristic: string/object/bool → classification;
            # numeric with <=10 unique contiguous integers starting at 0/1 → classification
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

        # Basic summary text
        task_type = "classification" if is_classification else "regression"
        summary = f"Loaded dataset with {row_count} rows and {col_count} columns. "
        summary += f"Detected target column: '{target_col}' (task: {task_type}, unique values: {unique_targets}). "

        
        # Compute correlations
        print_progress(0.35, "Computing correlation matrices...")
        correlations = []
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().round(3)
            # Find top correlations (exclude self-correlation and duplicates)
            pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    val = corr_matrix.iloc[i, j]
                    if not np.isnan(val):
                        pairs.append((numeric_cols[i], numeric_cols[j], val))
            
            # Sort by absolute correlation value
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for x, y, val in pairs[:15]: # return top 15 correlations
                correlations.append({"x": x, "y": y, "value": float(val)})
                
        # Drop columns with all nulls in df_clean
        df_clean = combined_df.dropna(subset=[target_col]).reset_index(drop=True)
        if df_clean.empty:
            return {"error": f"No rows left after dropping records where target '{target_col}' is missing."}
        
        # Setup features X and target y
        y_raw = df_clean[target_col]
        # Impute target missing values using pandas directly to avoid PyArrow/NumPy reshape issues
        if is_classification:
            y = y_raw.fillna(y_raw.mode()[0] if not y_raw.mode().empty else "missing").astype(str).to_numpy()
        else:
            y = y_raw.fillna(y_raw.median()).to_numpy()
            
        X = df_clean.drop(columns=[target_col])
        
        # Preprocessing of X:
        # Categorize columns: standard categorical, text, or drop (ID-like)
        cols_to_drop = []
        text_cols = []
        for col in X.columns:
            if col == "__is_test":
                continue
            if col in datetime_cols:
                cols_to_drop.append(col)
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
                    # If the column has high cardinality and is not named like an ID,
                    # treat it as a text column for vectorization (TF-IDF) instead of dropping it.
                    if not is_id:
                        text_cols.append(col)
                    else:
                        cols_to_drop.append(col)
        X = X.drop(columns=cols_to_drop)
        
        print_progress(0.42, "Preprocessing features & imputing values...")
        X_numeric = [c for c in X.columns if c in numeric_cols and c != "__is_test"]
        X_categorical = [c for c in X.columns if c in categorical_cols and c not in text_cols and c != "__is_test"]
        
        # Impute missing values using pandas directly
        X_imputed = X.copy()
        for col in X_numeric:
            X_imputed[col] = X[col].fillna(X[col].median())
        for col in (X_categorical + text_cols):
            mode_val = X[col].mode()
            mode_choice = mode_val[0] if not mode_val.empty else "missing"
            X_imputed[col] = X[col].fillna(mode_choice).astype(str)
            
        # Process parts (numerical as-is, categorical one-hot encoded, text TF-IDF vectorized)
        processed_parts = []
        if X_numeric:
            df_num = X_imputed[X_numeric]
            processed_parts.append(df_num)
            
        if X_categorical:
            # One hot encode
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_cat_encoded = encoder.fit_transform(X_imputed[X_categorical])
            encoded_names = encoder.get_feature_names_out(X_categorical)
            df_cat = pd.DataFrame(X_cat_encoded, columns=encoded_names, index=X.index)
            processed_parts.append(df_cat)
            
        if text_cols:
            print_progress(0.48, "Vectorizing text features (TF-IDF)...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            for col in text_cols:
                try:
                    # Extract top 15 words to keep feature space small and fast
                    vectorizer = TfidfVectorizer(max_features=15, stop_words=None)
                    X_text_encoded = vectorizer.fit_transform(X_imputed[col]).toarray()
                    encoded_names = [f"{col}_tfidf_{word}" for word in vectorizer.get_feature_names_out()]
                    df_text = pd.DataFrame(X_text_encoded, columns=encoded_names, index=X.index)
                    processed_parts.append(df_text)
                except Exception:
                    pass
            
        if has_test_set or has_val_set:
            processed_parts.append(X_imputed[["__is_test"]])

        if not processed_parts:
            return {"error": "No features left after preprocessing."}
            
        X_processed = pd.concat(processed_parts, axis=1)
        # Re-build DataFrame using numpy representation to avoid PyArrow indexing bugs in scikit-learn
        X_processed = pd.DataFrame(X_processed.to_numpy(), columns=X_processed.columns, index=X_processed.index)
        
        print_progress(0.55, "Splitting dataset & training baseline models...")
        
        X_val, y_val = None, None
        val_metrics = None
        val_confusion_matrix_data = None

        if "__is_test" in X_processed.columns:
            train_mask = (X_processed["__is_test"] == 0).to_numpy()
            test_mask = (X_processed["__is_test"] == 1).to_numpy()
            val_mask = (X_processed["__is_test"] == 2).to_numpy()
            
            X_train_full = X_processed[train_mask].drop(columns=["__is_test"])
            y_train_full = y[train_mask]
            
            if np.any(val_mask):
                X_val = X_processed[val_mask].drop(columns=["__is_test"])
                y_val = y[val_mask]
                
            if np.any(test_mask):
                X_train = X_train_full
                y_train = y_train_full
                X_test = X_processed[test_mask].drop(columns=["__is_test"])
                y_test = y[test_mask]
            else:
                use_stratify = False
                if is_classification:
                    from collections import Counter
                    counts = Counter(y_train_full)
                    if counts and min(counts.values()) >= 2:
                        use_stratify = True
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train_full, y_train_full,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_train_full if use_stratify else None
                )
        else:
            # Split train/test (stratified if possible for classification)
            use_stratify = False
            if is_classification:
                from collections import Counter
                counts = Counter(y)
                if counts and min(counts.values()) >= 2:
                    use_stratify = True
                    
            from sklearn.model_selection import train_test_split
            if use_stratify:
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        
        # Cap dataset size for model training to avoid slow execution/OOM
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
        if X_val is not None and len(X_val) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(X_val), 5000, replace=False)
            X_val = X_val.iloc[indices]
            y_val = y_val[indices]
        
        # Models
        charts = []
        metrics = {}
        
        if is_classification:
            # 1. Logistic Regression
            print_progress(0.62, "Training Logistic Regression classifier...")
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_acc = accuracy_score(y_test, lr_preds)
            
            # 2. Random Forest
            print_progress(0.70, "Training Random Forest classifier...")
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_acc_rf = accuracy_score(y_test, rf_preds)
            
            # Choose best
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
                {"name": "Random Forest Classifier", "score": float(rf_acc_rf), "metric": "Accuracy"}
            ]
            metrics = {
                "model": best_model,
                "score_type": "Accuracy",
                "score": float(best_score),
                "additional_metrics": {
                    "F1 Score": float(f1_score(y_test, best_preds, average='weighted', zero_division=0))
                }
            }
            summary += f"Best model is {best_model} with an accuracy of {best_score:.2f}."
            
            # Feature Importances (Random Forest)
            if hasattr(rf, 'feature_importances_'):
                importances = rf.feature_importances_
                feat_imp = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)[:10]
                charts.append({
                    "type": "bar",
                    "title": "Top Feature Importances (Random Forest)",
                    "x_label": "Feature",
                    "y_label": "Importance",
                    "data": [{"x_val": name, "x_num": None, "y": float(val)} for name, val in feat_imp]
                })
                
            # Chart: Target distribution
            val_counts = target_series.value_counts()
            charts.append({
                "type": "bar",
                "title": "Target Class Distribution",
                "x_label": "Class",
                "y_label": "Count",
                "data": [{"x_val": str(k), "x_num": None, "y": float(v)} for k, v in val_counts.items()]
            })
            
            # Chart: True vs Predicted counts (for visual validation)
            # Find distribution in test set
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
                "title": "Test Set Predictions vs Actual Counts",
                "x_label": "Class Type",
                "y_label": "Count",
                "data": comparison_data
            })
            
        else:
            # 1. Linear Regression
            print_progress(0.62, "Training Linear Regression model...")
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_r2 = r2_score(y_test, lr_preds)
            
            # 2. Random Forest
            print_progress(0.70, "Training Random Forest regressor...")
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_r2_rf = r2_score(y_test, rf_preds)
            
            # Choose best
            if rf_r2_rf >= lr_r2:
                best_model = "Random Forest Regressor"
                best_score = rf_r2_rf
                best_preds = rf_preds
                best_reg = rf
            else:
                best_model = "Linear Regression"
                best_score = lr_r2
                best_preds = lr_preds
                best_reg = lr
                
            test_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
            
            models_compared = [
                {"name": "Linear Regression", "score": float(lr_r2), "metric": "R\u00b2 Score"},
                {"name": "Random Forest Regressor", "score": float(rf_r2_rf), "metric": "R\u00b2 Score"}
            ]
            metrics = {
                "model": best_model,
                "score_type": "R\u00b2 Score",
                "score": float(best_score),
                "additional_metrics": {
                    "RMSE": float(test_rmse)
                }
            }
            summary += f"Best model is {best_model} with R\u00b2 score of {best_score:.2f} (RMSE: {test_rmse:.2f})."
            
            # Feature Importances (Random Forest)
            if hasattr(rf, 'feature_importances_'):
                importances = rf.feature_importances_
                feat_imp = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)[:10]
                charts.append({
                    "type": "bar",
                    "title": "Top Feature Importances (Random Forest)",
                    "x_label": "Feature",
                    "y_label": "Importance",
                    "data": [{"x_val": name, "x_num": None, "y": float(val)} for name, val in feat_imp]
                })
                
            # Chart: Target distribution (Histogram binned)
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
            
            # Chart: True vs Predicted scatter plot (subsampled to fit)
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
                    "title": "Predicted vs Actual Scatter Plot (Test Set)",
                    "x_label": "Actual Value",
                    "y_label": "Predicted Value",
                    "data": scatter_data
                })

        # 5-Fold Cross Validation & Dummy Baseline Evaluation
        cv_scores = []
        cv_mean = None
        cv_std = None
        dummy_score = None
        
        n_samples = len(X_processed)
        if n_samples >= 5:
            # Determine CV splits
            if is_classification:
                from collections import Counter
                class_counts = Counter(y)
                min_class_count = min(class_counts.values()) if class_counts else 0
                if min_class_count >= 2:
                    n_splits = min(5, min_class_count)
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
            try:
                model_to_cv = best_clf if is_classification else best_reg
                metric_name = "accuracy" if is_classification else "r2"
                scores = cross_val_score(model_to_cv, X_processed, y, cv=cv, scoring=metric_name)
                cv_scores = [float(s) for s in scores]
                cv_mean = float(np.mean(scores))
                cv_std = float(np.std(scores))
            except Exception as cv_err:
                sys.stderr.write(f"Warning: Cross-validation failed: {str(cv_err)}\n")
                
        # Train & evaluate dummy baseline on test set
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
            
        # Confusion matrix for classification
        confusion_matrix_data = None
        if is_classification:
            try:
                classes_list = sorted(list(np.unique(y_test)))
                cm = confusion_matrix(y_test, best_preds, labels=classes_list)
                confusion_matrix_data = {
                    "labels": [str(c) for c in classes_list],
                    "values": cm.tolist()
                }
            except Exception as cm_err:
                sys.stderr.write(f"Warning: Confusion matrix calculation failed: {str(cm_err)}\n")
            
        # Evaluate on Validation set if present
        val_metrics = None
        val_confusion_matrix_data = None
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
                sys.stderr.write(f"Warning: Validation evaluation failed: {str(val_err)}\n")
            
        # === SHAP VALUE CALCULATION (D1) ===
        if 'rf' in locals() and 'X_test' in locals() and rf is not None and X_test is not None and not X_test.empty:
            print_progress(0.78, "Calculating SHAP values for feature importance...")
            try:
                import shap
                explainer = shap.TreeExplainer(rf)
                
                # Subsample X_test to keep execution fast (max 100 samples)
                shap_samples = X_test
                if len(shap_samples) > 100:
                    shap_samples = shap_samples.sample(n=100, random_state=42)
                    
                shap_vals = explainer.shap_values(shap_samples)
                
                # Normalize shap_values to a 2D array of shape (N, features)
                if isinstance(shap_vals, list):
                    # Multiclass: take mean of absolute SHAP values across all classes
                    shap_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
                elif len(shap_vals.shape) == 3:
                    # Multiclass 3D array: average absolute values over classes axis (last axis)
                    shap_abs = np.mean(np.abs(shap_vals), axis=-1)
                else:
                    # Regression / Binary: absolute values
                    shap_abs = np.abs(shap_vals)
                    
                # Compute mean absolute SHAP value per feature
                mean_shap = shap_abs.mean(axis=0)
                
                # Map features to their SHAP values
                feature_names = X_test.columns.tolist()
                shap_importances = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)[:10]
                
                charts.append({
                    "type": "bar",
                    "title": "Global Feature Importance (SHAP)",
                    "x_label": "Feature",
                    "y_label": "Mean |SHAP Value|",
                    "data": [{"x_val": name, "x_num": None, "y": float(val)} for name, val in shap_importances]
                })
            except Exception as shap_err:
                sys.stderr.write(f"Warning: SHAP evaluation failed: {str(shap_err)}\n")

        # === ENHANCED EDA AND ADDITIONAL CHARTS ===
        print_progress(0.80, "Running advanced metrics and PCA...")
        
        # 1. PCA Explained Variance Line Chart (if >= 3 numeric features)
        numeric_features = [c for c in numeric_cols if c != target_col]
        if len(numeric_features) >= 3:
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                # Impute and scale numeric features
                df_pca = df_clean[numeric_features].copy()
                for c in numeric_features:
                    df_pca[c] = df_pca[c].fillna(df_pca[c].median())
                
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

        # 2. Correlation of Numeric Features with Target (for Regression)
        if not is_classification and len(numeric_features) > 0:
            target_corr = []
            for col in numeric_features:
                try:
                    val = df_clean[col].corr(df_clean[target_col])
                    if not np.isnan(val):
                        target_corr.append((col, val))
                except Exception:
                    pass
            if target_corr:
                target_corr.sort(key=lambda x: abs(x[1]), reverse=True)
                charts.append({
                    "type": "bar",
                    "title": "Correlation of Numeric Features with Target",
                    "x_label": "Feature Name",
                    "y_label": "Pearson Correlation",
                    "data": [{"x_val": name, "x_num": None, "y": float(val)} for name, val in target_corr[:10]]
                })

        # 3. Find top feature from Random Forest importances
        top_feature = None
        if hasattr(rf, 'feature_importances_'):
            importances = rf.feature_importances_
            feat_imp_sorted = sorted(zip(X_processed.columns, importances), key=lambda x: x[1], reverse=True)
            if feat_imp_sorted:
                cand = feat_imp_sorted[0][0]
                top_feature = cand
                # Map back to original X column if one-hot encoded
                for original_col in X.columns:
                    if cand.startswith(original_col):
                        top_feature = original_col
                        break

        # 4. Grouped Analysis Charts for Top Feature
        if top_feature and top_feature in df_clean.columns:
            if is_classification:
                # Target distribution based on top feature (numeric)
                if top_feature in numeric_cols:
                    grouped = df_clean.groupby(target_col)[top_feature].mean().reset_index()
                    charts.append({
                        "type": "bar",
                        "title": f"Mean of Top Feature ({top_feature}) by Target Class",
                        "x_label": "Target Class",
                        "y_label": f"Average {top_feature}",
                        "data": [{"x_val": str(row[target_col]), "x_num": None, "y": float(row[top_feature])} for _, row in grouped.iterrows()]
                    })
                # Target distribution based on top feature (categorical)
                elif top_feature in categorical_cols:
                    counts = df_clean.groupby([top_feature, target_col]).size().unstack(fill_value=0)
                    top_cats = counts.sum(axis=1).nlargest(5).index
                    grouped_data = []
                    for cat in top_cats:
                        for target_val in counts.columns:
                            val_count = counts.loc[cat, target_val]
                            if val_count > 0:
                                grouped_data.append({
                                    "x_val": str(cat),
                                    "x_num": None,
                                    "y": float(val_count),
                                    "series": str(target_val)
                                })
                    charts.append({
                        "type": "bar",
                        "title": f"Target Class Counts by {top_feature}",
                        "x_label": "Category",
                        "y_label": "Count",
                        "data": grouped_data
                    })
            else:
                # Regression: Top Feature vs Target scatter
                if top_feature in numeric_cols:
                    sub_df = df_clean[[top_feature, target_col]].dropna()
                    sample_size = min(len(sub_df), 150)
                    if sample_size > 0:
                        sampled = sub_df.sample(n=sample_size, random_state=42)
                        charts.append({
                            "type": "scatter",
                            "title": f"Target vs Top Feature ({top_feature})",
                            "x_label": f"{top_feature} Value",
                            "y_label": f"Target: {target_col}",
                            "data": [{"x_val": None, "x_num": float(row[top_feature]), "y": float(row[target_col])} for _, row in sampled.iterrows()]
                        })

        # 5. Distribution (Histogram) of the Top Numeric Feature
        dist_feat = top_feature if (top_feature in numeric_features) else (numeric_features[0] if numeric_features else None)
        if dist_feat:
            series = df_clean[dist_feat].dropna()
            if not series.empty:
                hist, bin_edges = np.histogram(series, bins=10)
                charts.append({
                    "type": "bar",
                    "title": f"Feature Distribution: {dist_feat}",
                    "x_label": f"{dist_feat} Range",
                    "y_label": "Frequency",
                    "data": [{"x_val": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}", "x_num": None, "y": float(hist[i])} for i in range(len(hist))]
                })

        # Generate outlier boxplots
        try:
            boxplots = generate_boxplots(df_clean, numeric_cols, target_col=target_col)
            charts.extend(boxplots)
        except Exception as box_err:
            sys.stderr.write(f"Failed to generate boxplots: {str(box_err)}\n")

        print_progress(0.92, "Compiling summary report & finalizing...")
        # 6. Rebuild 'summary' as a highly detailed, professional markdown report
        summary_sections = []
        
        # Overview
        overview = f"### 📊 Dataset Overview\n"
        overview += f"- **Rows:** {row_count:,} | **Columns:** {col_count:,}\n"
        overview += f"- **Column Types:** {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(text_cols)} text\n"
        overview += f"- **Missing Value Cells:** {sum(missing.values()):,} total across {len([k for k,v in missing.items() if v > 0])} columns."
        summary_sections.append(overview)
        
        # Distribution / DQ alerts
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
            
        # Target Variable
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
        
        # ML Models
        model_perf = f"### 🤖 Machine Learning Model Performance\n"
        model_perf += f"- **Best Model:** `{best_model}`\n"
        model_perf += f"- **Primary Score ({metrics['score_type']}):** `{metrics['score']:.4f}`\n"
        if 'additional_metrics' in metrics and metrics['additional_metrics']:
            for am_name, am_val in metrics['additional_metrics'].items():
                model_perf += f"- **{am_name}:** `{am_val:.4f}`"
        summary_sections.append(model_perf)
        
        # Data profiling
        print_progress(0.88, "Profiling columns & generating data statistics...")
        profiling = profile_dataset(df)

        # Data Leakage Warnings (D7)
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

        # Auto-Cleaning Recommendations (D8)
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

        # Model Interpretability & Diagnostics (D2, D3, D4, D5)
        try:
            model_to_explain = None
            if 'best_clf' in locals() and best_clf is not None:
                model_to_explain = best_clf
            elif 'best_reg' in locals() and best_reg is not None:
                model_to_explain = best_reg
                
            if model_to_explain is not None and len(X_test) > 5:
                # D2: Permutation Importance
                from sklearn.inspection import permutation_importance
                pi_result = permutation_importance(
                    model_to_explain, X_test, y_test, n_repeats=5, random_state=42
                )
                pi_importances = pi_result.importances_mean
                pi_features = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(X_processed.shape[1])]
                sorted_pi = sorted(zip(pi_features, pi_importances), key=lambda x: x[1], reverse=True)[:10]
                
                charts.append({
                    "type": "bar",
                    "title": "Permutation Feature Importances (Test Set)",
                    "x_label": "Feature",
                    "y_label": "Decrease in Score",
                    "data": [{"x_val": name, "x_num": None, "y": float(val)} for name, val in sorted_pi]
                })

                # D3 & D5: ROC/PR/Calibration curves (Binary Classification)
                if is_classification and len(np.unique(y_test)) == 2:
                    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
                    from sklearn.calibration import calibration_curve
                    
                    if hasattr(model_to_explain, "predict_proba"):
                        y_probs = model_to_explain.predict_proba(X_test)[:, 1]
                        
                        if hasattr(model_to_explain, "classes_") and len(model_to_explain.classes_) == 2:
                            pos_label = model_to_explain.classes_[1]
                        else:
                            pos_label = np.unique(y_test)[1]
                        
                        y_test_bin = (y_test == pos_label).astype(int)
                        
                        # ROC Curve
                        fpr, tpr, _ = roc_curve(y_test_bin, y_probs)
                        roc_auc = auc(fpr, tpr)
                        step = max(1, len(fpr) // 100)
                        roc_data = []
                        for idx in range(0, len(fpr), step):
                            roc_data.append({"x_val": None, "x_num": float(fpr[idx]), "y": float(tpr[idx]), "series": "Model"})
                        roc_data.append({"x_val": None, "x_num": 1.0, "y": 1.0, "series": "Model"})
                        roc_data.append({"x_val": None, "x_num": 0.0, "y": 0.0, "series": "Random Guess"})
                        roc_data.append({"x_val": None, "x_num": 1.0, "y": 1.0, "series": "Random Guess"})
                        
                        charts.append({
                            "type": "line",
                            "title": f"ROC Curve (AUC: {roc_auc:.3f})",
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
                            pr_data.append({"x_val": None, "x_num": float(recall[idx]), "y": float(precision[idx]), "series": "Model"})
                        pr_data.append({"x_val": None, "x_num": 1.0, "y": float(precision[-1]), "series": "Model"})
                        
                        # PR Baseline
                        pos_ratio = float(np.sum(y_test_bin == 1) / len(y_test_bin))
                        pr_data.append({"x_val": None, "x_num": 0.0, "y": pos_ratio, "series": "Baseline"})
                        pr_data.append({"x_val": None, "x_num": 1.0, "y": pos_ratio, "series": "Baseline"})
                        
                        charts.append({
                            "type": "line",
                            "title": f"Precision-Recall Curve (AP: {ap_score:.3f})",
                            "x_label": "Recall",
                            "y_label": "Precision",
                            "data": pr_data
                        })
 
                        # Probability Calibration Curve
                        prob_true, prob_pred = calibration_curve(y_test_bin, y_probs, n_bins=10)
                        cal_data = []
                        for idx in range(len(prob_true)):
                            cal_data.append({"x_val": None, "x_num": float(prob_pred[idx]), "y": float(prob_true[idx]), "series": "Model"})
                        cal_data.append({"x_val": None, "x_num": 0.0, "y": 0.0, "series": "Perfect Calibration"})
                        cal_data.append({"x_val": None, "x_num": 1.0, "y": 1.0, "series": "Perfect Calibration"})
                        
                        charts.append({
                            "type": "line",
                            "title": "Probability Calibration Curve",
                            "x_label": "Mean Predicted Probability",
                            "y_label": "Fraction of Positives",
                            "data": cal_data
                        })

                # D4: Residuals Plot (Regression)
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
                            "title": "Residuals Plot (Predicted vs. Error)",
                            "x_label": "Predicted Value",
                            "y_label": "Residual (Actual - Predicted)",
                            "data": res_data
                        })
        except Exception as explanation_err:
            sys.stderr.write(f"Warning: Failed to compute advanced ML metrics: {str(explanation_err)}\n")

        summary = "\n\n".join(summary_sections)

        # Phase 1: Model & Code Export
        if model_export_path or code_export_path:
            model_to_save = best_clf if is_classification else best_reg
            feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else []
            _export_model_and_code(
                model_to_save, model_export_path, code_export_path,
                file_path, "tabular", target_col, exclude_cols,
                "classification" if is_classification else "regression",
                feature_names, best_model, numeric_cols, categorical_cols, text_cols
            )

        # Format the response
        result = {
            "summary": summary,
            "columns": columns,
            "row_count": int(row_count),
            "col_count": int(col_count),
            "original_row_count": int(original_row_count) if 'original_row_count' in locals() else None,
            "sampled_row_count": int(sampled_row_count) if ('sampled_row_count' in locals() and sampled_row_count is not None) else None,
            "task_type": task_type,
            "numeric_col_count": len(numeric_cols),
            "categorical_col_count": len(categorical_cols),
            "text_col_count": len(text_cols),
            "missing_values": {k: int(v) for k, v in missing.items()},
            "correlations": correlations,
            "charts": charts,
            "metrics": metrics,
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
        result.update(test_info)
        result.update(val_info)
        return result
        
    except Exception as e:
        import traceback
        return {"error": f"An error occurred during execution: {str(e)}\n{traceback.format_exc()}"}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aura Python Analysis Pipeline")
    parser.add_argument("file", help="Path or URL to the dataset file")
    # Legacy positional target (kept for backward compat); overridden by --target
    parser.add_argument("legacy_target", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--dataset-type", default="tabular",
                        choices=["tabular", "timeseries", "image", "nlp"],
                        help="Dataset type (tabular|timeseries|image|nlp)")
    parser.add_argument("--task-type", default="auto",
                        choices=["auto", "classification", "regression", "forecast"],
                        help="ML task type override")
    parser.add_argument("--time-col", default=None, help="Datetime column (for timeseries)")
    parser.add_argument("--exclude-cols", default=None,
                        help="Comma-separated column names to exclude")
    parser.add_argument("--test-file", default=None, help="Path or URL to test dataset file")
    parser.add_argument("--val-file", default=None, help="Path or URL to validation dataset file")
    parser.add_argument("--preview", action="store_true", help="Run in preview mode")
    parser.add_argument("--model-export-path", default=None, help="Path to save the best model (.joblib)")
    parser.add_argument("--code-export-path", default=None, help="Path to save the reproduction code (.py)")
    parser.add_argument("--smart-sample", action="store_true", help="Enable smart sampling for large datasets")
    parser.add_argument("--cleaning-actions", default=None, help="JSON string of cleaning actions to apply")

    args = parser.parse_args()

    # Resolve target: --target wins over legacy positional
    target = args.target or args.legacy_target

    # Parse excluded columns
    exclude_cols = set()
    if args.exclude_cols:
        exclude_cols = set(c.strip() for c in args.exclude_cols.split(",") if c.strip())

    if args.preview:
        analysis = analyze_preview(args.file, dataset_type=args.dataset_type)
    else:
        analysis = analyze(
            args.file,
            target_col=target,
            dataset_type=args.dataset_type,
            task_type_override=args.task_type,
            time_col=args.time_col,
            exclude_cols=exclude_cols,
            test_file_path=args.test_file,
            val_file_path=args.val_file,
            model_export_path=args.model_export_path,
            code_export_path=args.code_export_path,
            smart_sample=args.smart_sample,
            cleaning_actions=args.cleaning_actions
        )

    analysis = clean_nan(analysis)
    print(json.dumps(analysis))

