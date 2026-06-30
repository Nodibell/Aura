import numpy as np
import pandas as pd

def profile_dataset(df, column_type_overrides=None):
    profiling = {
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {}
    }
    for col in df.columns:
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
                    
        is_raw_num = pd.api.types.is_numeric_dtype(col_series.dtype)
        
        is_categorical_num = False
        if is_raw_num:
            nunique = int(col_series.nunique())
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
                
        is_num = is_raw_num and not is_categorical_num
        
        # Detect if it is an identifier column
        is_identifier = False
        col_lower = col.lower()
        col_non_null = col_series.dropna()
        non_null_count = len(col_non_null)
        if non_null_count > 0:
            nunique = int(col_series.nunique())
            is_unique_key = nunique == non_null_count or (nunique / non_null_count) >= 0.98
            is_id_name = col_lower in ["id", "index", "no", "number", "num", "row", "rowid"] or \
                         col_lower.endswith("_id") or col_lower.endswith("id") or col_lower.startswith("id_") or \
                         col_lower.endswith("key") or col_lower.startswith("key_")
                         
            if is_unique_key:
                if is_id_name:
                    is_identifier = True
                elif col_series.dtype == object or pd.api.types.is_string_dtype(col_series.dtype):
                    try:
                        sample_vals = col_non_null.head(100).astype(str)
                        has_spaces = sample_vals.str.contains(r'\s').any()
                        avg_val_len = sample_vals.str.len().mean()
                        if not has_spaces and avg_val_len < 40 and not col_lower == "password":
                            is_identifier = True
                    except Exception:
                        pass

        # Detect if it is text/NLP column
        is_text = False
        if not is_num and not is_identifier:
            try:
                # Calculate average string length of non-null values
                non_null_strs = col_series.dropna().astype(str)
                if not non_null_strs.empty:
                    avg_len = non_null_strs.str.len().mean()
                    if avg_len > 50:
                        is_text = True
                    else:
                        # High cardinality string columns (like names, passwords) are text/NLP, not categorical
                        nunique = int(col_series.nunique())
                        if nunique > 100 and (pd.api.types.is_string_dtype(col_series.dtype) or col_series.dtype == object):
                            is_text = True
            except Exception:
                pass
        
        if column_type_overrides and col in column_type_overrides:
            col_type = column_type_overrides[col]
        else:
            col_type = "identifier" if is_identifier else ("datetime" if is_datetime else ("numeric" if is_num else ("text" if is_text else "categorical")))
        
        # Check for multi-label target format (comma-separated strings)
        is_multi_label = False
        if col_type in ["categorical", "text"]:
            try:
                non_nulls = col_series.dropna().astype(str)
                if not non_nulls.empty:
                    comma_count = non_nulls.str.contains(',').sum()
                    if comma_count / len(non_nulls) >= 0.1:
                        split_lens = non_nulls.str.split(',').apply(len)
                        if split_lens.mean() > 1.1:
                            is_multi_label = True
            except Exception:
                pass

        col_profile = {
            "nunique": int(col_series.nunique()),
            "missing": int(col_series.isnull().sum()),
            "type": col_type,
            "is_multi_label": is_multi_label
        }
        
        if col_type == "numeric":
            try:
                num_series = pd.to_numeric(col_series, errors='coerce')
                desc = num_series.describe()
                col_profile["stats"] = {
                    "min": float(desc.get("min", 0.0)) if not np.isnan(desc.get("min", 0.0)) else 0.0,
                    "max": float(desc.get("max", 0.0)) if not np.isnan(desc.get("max", 0.0)) else 0.0,
                    "mean": float(desc.get("mean", 0.0)) if not np.isnan(desc.get("mean", 0.0)) else 0.0,
                    "std": float(desc.get("std", 0.0)) if not np.isnan(desc.get("std", 0.0)) else 0.0,
                    "p25": float(num_series.quantile(0.25)) if not np.isnan(num_series.quantile(0.25)) else 0.0,
                    "p50": float(num_series.quantile(0.50)) if not np.isnan(num_series.quantile(0.50)) else 0.0,
                    "p75": float(num_series.quantile(0.75)) if not np.isnan(num_series.quantile(0.75)) else 0.0
                }
                non_nulls = num_series.dropna()
                is_int = False
                if pd.api.types.is_integer_dtype(num_series.dtype):
                    is_int = True
                elif not non_nulls.empty:
                    try:
                        is_int = non_nulls.apply(lambda x: float(x).is_integer()).all()
                    except Exception:
                        pass
                col_profile["is_integer"] = bool(is_int)
            except Exception:
                pass
        elif col_type == "text":
            # Stats for text column (character length stats)
            try:
                lengths = col_series.dropna().astype(str).str.len()
                if not lengths.empty:
                    desc = lengths.describe()
                    col_profile["stats"] = {
                        "min": float(desc.get("min", 0.0)) if not np.isnan(desc.get("min", 0.0)) else 0.0,
                        "max": float(desc.get("max", 0.0)) if not np.isnan(desc.get("max", 0.0)) else 0.0,
                        "mean": float(desc.get("mean", 0.0)) if not np.isnan(desc.get("mean", 0.0)) else 0.0,
                        "std": float(desc.get("std", 0.0)) if not np.isnan(desc.get("std", 0.0)) else 0.0,
                        "p25": float(lengths.quantile(0.25)) if not np.isnan(lengths.quantile(0.25)) else 0.0,
                        "p50": float(lengths.quantile(0.50)) if not np.isnan(lengths.quantile(0.50)) else 0.0,
                        "p75": float(lengths.quantile(0.75)) if not np.isnan(lengths.quantile(0.75)) else 0.0
                    }
            except Exception:
                pass
        else:
            top_cats = col_series.value_counts().head(5)
            col_profile["top_categories"] = [
                {"value": str(k), "count": int(v)} for k, v in top_cats.items()
            ]
            
        profiling["columns"][col] = col_profile
    return profiling
