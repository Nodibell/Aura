import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np
import pandas as pd


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
    
    # 2. Load and register helpers and event_bus first (since others depend on them)
    import helpers
    sys.modules["utils.helpers"] = helpers
    utils_mod.helpers = helpers

    if os.path.exists(os.path.join(script_dir, "event_bus.py")):
        import event_bus
        sys.modules["utils.event_bus"] = event_bus
        utils_mod.event_bus = event_bus
    
    # 3. Load others sequentially
    if os.path.exists(os.path.join(script_dir, "cleaning.py")):
        import cleaning
        sys.modules["utils.cleaning"] = cleaning
        utils_mod.cleaning = cleaning

    import loader
    sys.modules["utils.loader"] = loader
    utils_mod.loader = loader
    
    import profiler
    sys.modules["utils.profiler"] = profiler
    utils_mod.profiler = profiler
    
    import charts
    sys.modules["utils.charts"] = charts
    utils_mod.charts = charts

    if os.path.exists(os.path.join(script_dir, "data_engine.py")):
        import data_engine
        sys.modules["utils.data_engine"] = data_engine
        utils_mod.data_engine = data_engine

    if os.path.exists(os.path.join(script_dir, "ai_analyst.py")):
        import ai_analyst
        sys.modules["utils.ai_analyst"] = ai_analyst
        utils_mod.ai_analyst = ai_analyst

if not os.path.isdir(os.path.join(script_dir, "pipelines")) and os.path.exists(os.path.join(script_dir, "timeseries.py")):
    import types
    # 1. Register namespace package 'pipelines'
    pipelines_mod = types.ModuleType("pipelines")
    pipelines_mod.__path__ = []
    sys.modules["pipelines"] = pipelines_mod
    
    # 2. Load deep_learning, model_engine, and cv_nlp_engine first (since other pipelines import from them)
    if os.path.exists(os.path.join(script_dir, "deep_learning.py")):
        import deep_learning
        sys.modules["pipelines.deep_learning"] = deep_learning
        pipelines_mod.deep_learning = deep_learning

    if os.path.exists(os.path.join(script_dir, "model_engine.py")):
        import model_engine
        sys.modules["pipelines.model_engine"] = model_engine
        pipelines_mod.model_engine = model_engine

    if os.path.exists(os.path.join(script_dir, "cv_nlp_engine.py")):
        import cv_nlp_engine
        sys.modules["pipelines.cv_nlp_engine"] = cv_nlp_engine
        pipelines_mod.cv_nlp_engine = cv_nlp_engine

    # 3. Load other pipelines sequentially
    import timeseries
    sys.modules["pipelines.timeseries"] = timeseries
    pipelines_mod.timeseries = timeseries
    
    import image
    sys.modules["pipelines.image"] = image
    pipelines_mod.image = image
    
    import nlp
    sys.modules["pipelines.nlp"] = nlp
    pipelines_mod.nlp = nlp
    
    import object_detection
    sys.modules["pipelines.object_detection"] = object_detection
    pipelines_mod.object_detection = object_detection
    
    import preview
    sys.modules["pipelines.preview"] = preview
    pipelines_mod.preview = preview
    
    if os.path.exists(os.path.join(script_dir, "tabular.py")):
        import tabular
        sys.modules["pipelines.tabular"] = tabular
        pipelines_mod.tabular = tabular

    if os.path.exists(os.path.join(script_dir, "clustering.py")):
        import clustering
        sys.modules["pipelines.clustering"] = clustering
        pipelines_mod.clustering = clustering

# Disable Metal API validation layer for PyTorch MPS subprocess execution to prevent assertion crashes when run from Xcode
os.environ["MTL_DEBUG_LAYER"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


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

def compute_mixed_correlation(df, col1, col2):
    try:
        import scipy.stats as ss
        
        temp_df = df[[col1, col2]].dropna()
        if len(temp_df) < 5:
            return np.nan
            
        s1 = temp_df[col1]
        s2 = temp_df[col2]
        
        is_num1 = pd.api.types.is_numeric_dtype(s1.dtype) and s1.nunique() > 2
        is_num2 = pd.api.types.is_numeric_dtype(s2.dtype) and s2.nunique() > 2
        
        if is_num1 and is_num2:
            val, _ = ss.pearsonr(s1, s2)
            return val
        elif not is_num1 and not is_num2:
            confusion_matrix = pd.crosstab(s1, s2)
            if confusion_matrix.empty or min(confusion_matrix.shape) <= 1:
                return 0.0
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = len(s1)
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            if min(r - 1, k - 1) == 0:
                return 0.0
            return float(np.sqrt(phi2 / min(r - 1, k - 1)))
        else:
            num_s = s1 if is_num1 else s2
            cat_s = s2 if is_num1 else s1
            
            cats = cat_s.unique()
            if len(cats) == 2:
                binary_s = (cat_s == cats[0]).astype(int)
                val, _ = ss.pointbiserialr(binary_s, num_s)
                return val
            elif len(cats) > 2:
                groups = [num_s[cat_s == cat] for cat in cats]
                overall_mean = num_s.mean()
                ss_total = ((num_s - overall_mean) ** 2).sum()
                if ss_total == 0:
                    return 0.0
                ss_between = sum([len(group) * ((group.mean() - overall_mean) ** 2) for group in groups])
                return float(np.sqrt(ss_between / ss_total))
            else:
                return np.nan
    except Exception:
        try:
            temp_df = df[[col1, col2]].dropna()
            if len(temp_df) < 5:
                return np.nan
            s1 = temp_df[col1]
            s2 = temp_df[col2]
            if not pd.api.types.is_numeric_dtype(s1.dtype):
                s1 = pd.factorize(s1)[0]
            if not pd.api.types.is_numeric_dtype(s2.dtype):
                s2 = pd.factorize(s2)[0]
            return float(pd.Series(s1).corr(pd.Series(s2)))
        except Exception:
            return np.nan

def analyze(file_path, target_col=None, dataset_type="tabular",
            task_type_override="auto", time_col=None, exclude_cols=None, test_file_path=None, val_file_path=None,
            model_export_path=None, code_export_path=None, notebook_export_path=None, smart_sample=False, cleaning_actions=None,
            feature_selection=False, column_type_overrides=None,
            time_range_start=None, time_range_end=None):
    from utils.helpers import print_progress
    from utils.loader import download_dataset, load_dataset
    from utils.cleaning import StatefulCleaner
    from utils.profiler import profile_dataset

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

        # Run Smart Ingestion Adapter to detect format & standardize input paths
        try:
            from utils.ingestion import ingest_dataset
            file_path, dataset_type = ingest_dataset(file_path, dataset_type)
        except Exception as ingest_err:
            sys.stderr.write(f"Warning: Smart Ingestion failed: {str(ingest_err)}\n")

        if dataset_type == "image":
            from pipelines.image import analyze_image
            res = analyze_image(file_path, task_type_override, target_col, test_file_path=test_file_path, model_export_path=model_export_path, code_export_path=code_export_path)
            res["file_path"] = file_path
            return res

        if dataset_type == "object_detection":
            from pipelines.object_detection import analyze_object_detection
            res = analyze_object_detection(file_path, task_type_override, target_col, test_file_path=test_file_path, model_export_path=model_export_path, code_export_path=code_export_path)
            res["file_path"] = file_path
            return res


        print_progress(0.15, "Loading dataset file...")
        df = load_dataset(file_path)

        if df.empty:
            return {"error": "The dataset is empty."}

        # ── Generate rich dataset context for AI chat (lightweight RAG) ──
        _dataset_context = ""
        try:
            from utils.ai_analyst import AIAnalyst
            _dataset_context = AIAnalyst().generate_dataset_context(df, target_col or "")
        except Exception as _ctx_err:
            sys.stderr.write(f"Warning: dataset context generation failed: {_ctx_err}\n")


        # Apply Time Range Filtering for Time Series datasets
        if dataset_type == "timeseries" and (time_range_start or time_range_end):
            resolved_time_col = time_col
            if not resolved_time_col:
                ts_keywords = ["date", "time", "timestamp", "datetime", "year", "month", "period", "week"]
                for col in df.columns:
                    if any(kw in col.lower() for kw in ts_keywords):
                        resolved_time_col = col
                        break
                if not resolved_time_col:
                    resolved_time_col = df.columns[0]
            
            try:
                temp_time = pd.to_datetime(df[resolved_time_col], errors='coerce')
                mask = pd.Series(True, index=df.index)
                if time_range_start:
                    mask &= (temp_time >= pd.to_datetime(time_range_start))
                if time_range_end:
                    mask &= (temp_time <= pd.to_datetime(time_range_end))
                df = df[mask].reset_index(drop=True)
                sys.stderr.write(f"Filtered time series dataset to {len(df)} rows between {time_range_start} and {time_range_end}.\n")
            except Exception as filter_err:
                sys.stderr.write(f"Warning: Failed to filter by time range: {str(filter_err)}\n")

        original_row_count = len(df)
        sampled_row_count = None

        # Apply column exclusion BEFORE any analysis
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
            print_progress(0.17, f"Excluded columns: {list(exclude_cols)}...")

        # Initialize, fit and apply the stateful cleaner
        cleaner = None
        if cleaning_actions:
            try:
                actions = json.loads(cleaning_actions)
                cleaner = StatefulCleaner(actions)
                cleaner.fit(df)
                df = cleaner.transform(df, is_training=True)
                print_progress(0.173, "Applied interactive cleaning recommendations...")
            except Exception as clean_err:
                sys.stderr.write(f"Warning: Failed to fit/apply cleaning actions: {str(clean_err)}\n")

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
                    if cleaner is not None and not test_df.empty:
                        test_df = cleaner.transform(test_df, is_training=False)
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
                    if cleaner is not None and not val_df.empty:
                        val_df = cleaner.transform(val_df, is_training=False)
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
            if column_type_overrides and col in column_type_overrides:
                override_type = column_type_overrides[col]
                if override_type == "numeric":
                    numeric_cols.append(col)
                elif override_type == "categorical" or override_type == "text":
                    categorical_cols.append(col)
                elif override_type == "datetime":
                    datetime_cols.append(col)
                elif override_type == "identifier":
                    exclude_cols.discard(col) # Remove from active list
                    # Wait, excluded columns are filtered out from `columns` already, but we need to make sure it is added to exclude_cols.
                    exclude_cols.add(col)
                continue

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
                t_profiling = profile_dataset(test_df, column_type_overrides=column_type_overrides)
                
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
                v_profiling = profile_dataset(val_df, column_type_overrides=column_type_overrides)
                
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
            from pipelines.timeseries import analyze_timeseries
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
            res["dataset_context"] = _dataset_context
            res["file_path"] = file_path
            return res


        if dataset_type == "nlp":
            from pipelines.nlp import analyze_nlp
            res = analyze_nlp(
                combined_df, target_col, task_type_override,
                row_count, col_count, columns, full_preview, missing,
                numeric_cols, categorical_cols,
                file_path=file_path, model_export_path=model_export_path, code_export_path=code_export_path
            )
            res.update(test_info)
            res.update(val_info)
            res["dataset_context"] = _dataset_context
            res["file_path"] = file_path
            return res



        if task_type_override == "clustering" or (dataset_type == "tabular" and (not target_col or target_col == "")):
            if "pipelines.clustering" in sys.modules:
                analyze_clustering = sys.modules["pipelines.clustering"].analyze_clustering
            else:
                from pipelines.clustering import analyze_clustering
            res = analyze_clustering(
                combined_df, row_count, col_count, columns, full_preview, missing,
                numeric_cols, categorical_cols, file_path=file_path
            )
            res.update(test_info)
            res.update(val_info)
            res["dataset_context"] = _dataset_context
            res["file_path"] = file_path
            return res


        # Delegate tabular analysis to the pipelines/tabular.py module
        from pipelines.tabular import analyze_tabular
        res = analyze_tabular(
            combined_df, target_col, task_type_override,
            row_count, col_count, columns, full_preview, missing,
            numeric_cols, categorical_cols,
            file_path=file_path, model_export_path=model_export_path, code_export_path=code_export_path,
            smart_sample=smart_sample, cleaning_actions=cleaning_actions,
            test_df=test_df, val_df=val_df, has_test_set=has_test_set, has_val_set=has_val_set,
            test_info=test_info, val_info=val_info,
            cleaner=cleaner, feature_selection=feature_selection,
            column_type_overrides=column_type_overrides
        )

        
        # Calculate mixed correlation matrix
        correlations = []
        all_corr_cols = [c for c in numeric_cols + categorical_cols if c in df.columns]
        if len(all_corr_cols) > 30:
            prio_cols = []
            if target_col in all_corr_cols:
                prio_cols.append(target_col)
            for c in all_corr_cols:
                if c != target_col and len(prio_cols) < 30:
                    prio_cols.append(c)
            all_corr_cols = prio_cols
            
        if len(all_corr_cols) > 1:
            try:
                pairs = []
                for i in range(len(all_corr_cols)):
                    for j in range(i + 1, len(all_corr_cols)):
                        col1, col2 = all_corr_cols[i], all_corr_cols[j]
                        val = compute_mixed_correlation(df, col1, col2)
                        if not np.isnan(val):
                            pairs.append((col1, col2, val))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for x, y, val in pairs[:100]:
                    correlations.append({"x": x, "y": y, "value": float(val)})
            except Exception as corr_err:
                sys.stderr.write(f"Warning: Failed to compute mixed correlation matrix: {str(corr_err)}\n")
        
        if "error" not in res or res["error"] is None:
            res["correlations"] = correlations
            res["dataset_context"] = _dataset_context
            res["file_path"] = file_path

        # ── Optional: export as Jupyter Notebook ──
        if notebook_export_path and ("error" not in res or res["error"] is None):
            try:
                from utils.notebook_exporter import generate_notebook
                config_dict = {
                    "train_file_path": file_path,
                    "target_column": target_col or "",
                    "dataset_type": dataset_type,
                    "excluded_columns": list(exclude_cols) if exclude_cols else [],
                    "cleaning_actions": json.loads(cleaning_actions) if cleaning_actions else [],
                    "model_export_path": model_export_path,
                }
                generate_notebook(config_dict, res, notebook_export_path)
                sys.stderr.write(f"Notebook exported to: {notebook_export_path}\n")
            except Exception as nb_err:
                sys.stderr.write(f"Warning: Notebook export failed: {nb_err}\n")

        return res

        
    except Exception as e:
        import traceback
        return {"error": f"An error occurred during execution: {str(e)}\n{traceback.format_exc()}"}

def run_predict(model_path, input_data_json):
    import joblib
    import json
    import sys
    import pandas as pd
    import numpy as np

    def _log(msg):
        sys.stderr.write(f"[run_predict] {msg}\n")
        sys.stderr.flush()

    _log(f"Loading pipeline from: {model_path}")
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        _log(f"ERROR loading model: {e}")
        raise

    # Backward-compatibility: old cached files were saved as raw model objects
    if not isinstance(pipeline, dict):
        _log(f"Legacy raw model detected (type={type(pipeline).__name__}). Wrapping in standard dict.")
        pipeline = {
            'cleaner': None,
            'preprocessor': None,
            'model': pipeline,
            'feature_names': None,
            'target_col': None,
            'label_encoder': None,
        }

    pipeline_keys = list(pipeline.keys())
    _log(f"Pipeline keys: {pipeline_keys}")

    _log(f"Parsing input JSON: {input_data_json[:300]}")
    input_dict = json.loads(input_data_json)
    _log(f"Input columns: {list(input_dict.keys())}")

    # Build DataFrame
    df = pd.DataFrame({k: [v] for k, v in input_dict.items()})
    _log(f"Input DataFrame shape: {df.shape}, dtypes: {df.dtypes.to_dict()}")

    # Apply cleaner
    if 'cleaner' in pipeline and pipeline['cleaner'] is not None:
        _log("Applying StatefulCleaner.transform() ...")
        try:
            cleaner = pipeline['cleaner']
            df = cleaner.transform(df)
            _log(f"After cleaner shape: {df.shape}")
        except Exception as e:
            _log(f"ERROR in cleaner.transform: {e}")
            import traceback; _log(traceback.format_exc())
            raise
    else:
        _log("No cleaner in pipeline, skipping.")

    # Apply preprocessor
    X_proc = df
    if 'preprocessor' in pipeline and pipeline['preprocessor'] is not None:
        preprocessor = pipeline['preprocessor']
        _log(f"Applying preprocessor: {type(preprocessor).__name__}")
        if hasattr(preprocessor, "feature_names_in_"):
            expected = list(preprocessor.feature_names_in_)
            for col in expected:
                if col not in df.columns:
                    _log(f"  Adding missing column '{col}' = None")
                    df[col] = None
        try:
            X_proc = preprocessor.transform(df)
            _log(f"After preprocessor type: {type(X_proc).__name__}, shape: {X_proc.shape}")
        except Exception as e:
            _log(f"ERROR in preprocessor.transform: {e}")
            import traceback; _log(traceback.format_exc())
            raise

        if hasattr(X_proc, "columns"):
            new_cols = []
            for col in X_proc.columns:
                c = str(col)
                c = c.replace('[', '_').replace(']', '_').replace('<', 'lt_').replace('>', 'gt_')
                new_cols.append(c)
            X_proc.columns = new_cols
    else:
        _log("No preprocessor in pipeline, using raw DataFrame.")

    # Re-align features
    from sklearn.pipeline import Pipeline as _SKPipeline
    _is_sklearn_pipeline = isinstance(pipeline.get('model'), _SKPipeline)
    if not _is_sklearn_pipeline and 'feature_names' in pipeline and pipeline['feature_names'] is not None:
        feat_names = list(pipeline['feature_names'])
        _log(f"Re-aligning to {len(feat_names)} feature names")
        if hasattr(X_proc, "columns"):
            for col in feat_names:
                if col not in X_proc.columns:
                    X_proc[col] = 0.0
            X_proc = X_proc[feat_names]
    elif _is_sklearn_pipeline:
        _log("sklearn Pipeline detected — skipping feature_names re-alignment")

    model = pipeline['model']
    model_type_name = type(model).__name__
    _log(f"Model type: {model_type_name}")

    res = {}

    # ---------------------------------------------------------
    # ROUTING: statsmodels Time Series vs. scikit-learn
    # ---------------------------------------------------------
    if model_type_name in ["ARIMAResultsWrapper", "SARIMAXResultsWrapper", "HoltWintersResultsWrapper"]:
        _log("Routing to statsmodels forecast API...")
        try:
            # statsmodels needs strictly numeric exogenous features (drop datetime strings)
            exog_data = X_proc.select_dtypes(include=[np.number]) if isinstance(X_proc, pd.DataFrame) else pd.DataFrame(X_proc).select_dtypes(include=[np.number])
            
            if exog_data.shape[1] > 0:
                _log(f"Forecasting with exogenous features: {list(exog_data.columns)}")
                preds = model.forecast(steps=len(df), exog=exog_data)
            else:
                _log("Forecasting univariately (no exogenous features).")
                preds = model.forecast(steps=len(df))
                
            # Safely handle both pandas Series and raw numpy arrays
            preds_array = np.asarray(preds)
            res["prediction"] = preds_array[0]
            _log(f"forecast() -> {res['prediction']}")
            
        except Exception as e:
            _log(f"ERROR in statsmodels forecast: {e}")
            import traceback; _log(traceback.format_exc())
            raise

    else:
        # Standard scikit-learn / XGBoost / TabularNN prediction path
        import scipy.sparse
        if isinstance(model, _SKPipeline):
            if X_proc.shape[1] == 1:
                X_input = X_proc.iloc[:, 0]
                _log(f"sklearn Pipeline detected (NLP): passing 1D Series of dtype {X_input.dtype}")
            else:
                X_input = X_proc
                _log(f"sklearn Pipeline detected (multi-col): passing DataFrame shape {X_proc.shape}")
        elif scipy.sparse.issparse(X_proc):
            X_input = X_proc
            _log(f"Sparse matrix input, shape: {X_proc.shape}")
        else:
            X_input = X_proc.to_numpy() if hasattr(X_proc, "to_numpy") else X_proc
            _log(f"X_input shape: {X_input.shape if hasattr(X_input, 'shape') else 'unknown'}")

        if hasattr(model, "predict_proba"):
            _log("Running predict() + predict_proba() ...")
            try:
                preds = model.predict(X_input)
                res["prediction"] = preds[0]
                _log(f"predict() -> {preds[0]}")
            except Exception as e:
                _log(f"ERROR in model.predict: {e}")
                import traceback; _log(traceback.format_exc())
                raise
            try:
                probs = model.predict_proba(X_input)
                if 'label_encoder' in pipeline and pipeline['label_encoder'] is not None:
                    le = pipeline['label_encoder']
                    res["probabilities"] = {str(c): float(p) for c, p in zip(le.classes_, probs[0])}
                elif hasattr(model, "classes_"):
                    res["probabilities"] = {str(c): float(p) for c, p in zip(model.classes_, probs[0])}
                else:
                    res["probabilities"] = [float(p) for p in probs[0]]
                _log(f"predict_proba() -> {list(res['probabilities'].keys())[:5]}")
            except Exception as e:
                _log(f"WARNING in predict_proba (non-fatal): {e}")
        else:
            _log("Running predict() (no predict_proba) ...")
            try:
                preds = model.predict(X_input)
                res["prediction"] = preds[0]
                _log(f"predict() -> {preds[0]}")
            except Exception as e:
                _log(f"ERROR in model.predict: {e}")
                import traceback; _log(traceback.format_exc())
                raise

    # Apply LabelEncoder / MultiLabelBinarizer inverse transform
    if 'label_encoder' in pipeline and pipeline['label_encoder'] is not None:
        try:
            le = pipeline['label_encoder']
            from sklearn.preprocessing import MultiLabelBinarizer
            if isinstance(le, MultiLabelBinarizer):
                # inverse_transform takes 2D binarized array, e.g. [res["prediction"]]
                pred_label_tuple = le.inverse_transform([res["prediction"]])[0]
                pred_label = ", ".join(str(lbl) for lbl in pred_label_tuple)
                _log(f"MultiLabelBinarizer inverse_transform: {res['prediction']} -> {pred_label}")
                res["prediction"] = pred_label
            else:
                pred_label = le.inverse_transform([res["prediction"]])[0]
                _log(f"LabelEncoder inverse_transform: {res['prediction']} -> {pred_label}")
                res["prediction"] = pred_label
        except Exception as e:
            _log(f"WARNING: LabelEncoder/MultiLabelBinarizer inverse_transform failed: {e}")

    # Convert numpy types for JSON serialization
    if hasattr(res["prediction"], "ndim") and res["prediction"].ndim > 0:
        res["prediction"] = res["prediction"].tolist()
    elif hasattr(res["prediction"], "item"):
        res["prediction"] = res["prediction"].item()

    _log(f"Final prediction: {res['prediction']}")
    return res

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aura Python Analysis Pipeline")
    parser.add_argument("file", nargs="?", default=None, help="Path or URL to the dataset file")
    # Legacy positional target (kept for backward compat); overridden by --target
    parser.add_argument("legacy_target", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--predict", action="store_true", help="Run inference on model")
    parser.add_argument("--model-path", default=None, help="Path to the serialized model (.joblib)")
    parser.add_argument("--input-data", default=None, help="JSON string representing the input features")
    parser.add_argument("--dataset-type", default="tabular",
                        choices=["tabular", "timeseries", "image", "nlp", "object_detection"],
                        help="Dataset type (tabular|timeseries|image|nlp|object_detection)")
    parser.add_argument("--task-type", default="auto",
                        choices=["auto", "classification", "regression", "forecast", "clustering"],
                        help="ML task type override")
    parser.add_argument("--time-col", default=None, help="Datetime column (for timeseries)")
    parser.add_argument("--exclude-cols", default=None,
                        help="Comma-separated column names to exclude")
    parser.add_argument("--test-file", default=None, help="Path or URL to test dataset file")
    parser.add_argument("--val-file", default=None, help="Path or URL to validation dataset file")
    parser.add_argument("--preview", action="store_true", help="Run in preview mode")
    parser.add_argument("--model-export-path", default=None, help="Path to save the best model (.joblib)")
    parser.add_argument("--code-export-path", default=None, help="Path to save the reproduction code (.py)")
    parser.add_argument("--notebook-export-path", default=None, help="Path to save the Jupyter Notebook (.ipynb)")

    parser.add_argument("--smart-sample", action="store_true", help="Enable smart sampling for large datasets")
    parser.add_argument("--cleaning-actions", default=None, help="JSON string of cleaning actions to apply")
    parser.add_argument("--feature-selection", action="store_true", help="Enable automatic feature selection (RFE)")
    parser.add_argument("--column-type-overrides", default=None, help="JSON string of column type overrides")
    parser.add_argument("--time-range-start", default=None, help="Start date/time for time series filtering")
    parser.add_argument("--time-range-end", default=None, help="End date/time for time series filtering")
    
    # Merge options
    parser.add_argument("--merge", action="store_true", help="Merge two files and exit")
    parser.add_argument("--file2", default=None, help="Second file to merge")
    parser.add_argument("--key1", default=None, help="Join key for first file")
    parser.add_argument("--key2", default=None, help="Join key for second file")
    parser.add_argument("--join-type", default="inner", help="Type of join")
    parser.add_argument("--output-merge-path", default=None, help="Output path for merged file")

    args = parser.parse_args()

    from utils.event_bus import ProgressSubject, StderrProgressObserver
    ProgressSubject.get_instance().attach(StderrProgressObserver())

    if args.predict:
        if not args.model_path or not args.input_data:
            print(json.dumps({"error": "Prediction requires --model-path and --input-data."}))
            sys.exit(1)
        try:
            prediction_result = run_predict(args.model_path, args.input_data)
            print(json.dumps(prediction_result))
            sys.exit(0)
        except Exception as pred_err:
            import traceback
            print(json.dumps({"error": f"Prediction failed: {str(pred_err)}\n{traceback.format_exc()}"}))
            sys.exit(1)

    # Resolve target: --target wins over legacy positional
    target = args.target or args.legacy_target

    # Parse excluded columns
    exclude_cols = set()
    if args.exclude_cols:
        exclude_cols = set(c.strip() for c in args.exclude_cols.split(",") if c.strip())

    if args.merge:
        try:
            from utils.data_engine import DataEngine
            merged = DataEngine.merge_datasets(
                args.file, args.file2, args.key1, args.key2, args.join_type, args.output_merge_path
            )
            result = {
                "success": True,
                "row_count": len(merged),
                "columns": list(merged.columns)
            }
            print(json.dumps(result))
            sys.exit(0)
        except Exception as merge_err:
            result = {
                "success": False,
                "error": str(merge_err)
            }
            print(json.dumps(result))
            sys.exit(1)

    if args.preview:
        from pipelines.preview import analyze_preview
        analysis = analyze_preview(args.file, dataset_type=args.dataset_type)
    else:
        column_type_overrides = {}
        if args.column_type_overrides:
            try:
                column_type_overrides = json.loads(args.column_type_overrides)
            except Exception:
                pass

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
            notebook_export_path=args.notebook_export_path,
            smart_sample=args.smart_sample,
            cleaning_actions=args.cleaning_actions,
            feature_selection=args.feature_selection,
            column_type_overrides=column_type_overrides,
            time_range_start=args.time_range_start,
            time_range_end=args.time_range_end
        )


    from utils.helpers import clean_nan
    analysis = clean_nan(analysis)
    print(json.dumps(analysis))


