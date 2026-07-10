import os
import sys
import pandas as pd
from utils.loader import download_dataset, load_dataset, _infer_dataset_type
from utils.charts import get_image_preview
from utils.helpers import print_progress

def _get_datetime_bounds(file_path, columns):
    ts_keywords = {"date", "time", "timestamp", "datetime", "year", "month", "period", "week"}
    candidate_cols = [c for c in columns if any(kw in c.lower() for kw in ts_keywords)]
    
    if not candidate_cols:
        return {}
        
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            df_dates = pd.read_csv(file_path, usecols=candidate_cols, sep=sep)
        elif ext == ".parquet":
            df_dates = pd.read_parquet(file_path, columns=candidate_cols)
        else:
            df_dates = load_dataset(file_path)
            
        bounds = {}
        for col in candidate_cols:
            if col in df_dates.columns:
                series = pd.to_datetime(df_dates[col], errors='coerce').dropna()
                if not series.empty:
                    bounds[col] = {
                        "min": series.min().strftime("%Y-%m-%d"),
                        "max": series.max().strftime("%Y-%m-%d")
                    }
        return bounds
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to compute datetime bounds: {str(e)}\n")
        return {}

def analyze_preview(file_path, dataset_type=None, cleaning_actions=None):
    try:
        print_progress(0.01, "Initializing dataset preview...")
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

        # Determine if it's object detection
        print_progress(0.20, "Detecting dataset format...")
        is_object_detection = False
        resolved_path = file_path
        if os.path.isfile(file_path):
            parent = os.path.dirname(file_path)
            has_direct_yolo = False
            for name in ("dataset.yaml", "data.yaml", "yolo.yaml"):
                if os.path.exists(os.path.join(parent, name)):
                    has_direct_yolo = True
                    break
            if os.path.exists(os.path.join(parent, "images")) and os.path.exists(os.path.join(parent, "labels")):
                has_direct_yolo = True
            
            if has_direct_yolo:
                resolved_path = parent
                is_object_detection = True

        if dataset_type == "object_detection":
            is_object_detection = True
            if os.path.isfile(file_path):
                file_path = os.path.dirname(file_path)
        elif is_object_detection:
            file_path = resolved_path
        elif not dataset_type or dataset_type == "tabular":
            if os.path.isdir(file_path):
                from pipelines.object_detection import _find_yaml, _find_splits
                if _find_yaml(file_path) is not None or _find_splits(file_path, {}):
                    is_object_detection = True

        # Determine if it's image dataset
        is_image = False
        if not is_object_detection:
            if dataset_type == "image":
                is_image = True
            elif not dataset_type or dataset_type == "tabular":
                if os.path.isdir(file_path):
                    is_image = True
                else:
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == ".npz":
                        is_image = True

        if is_object_detection:
            print_progress(0.30, "Analyzing YOLO dataset structure...")
            from pipelines.object_detection import preview_yolo
            res = preview_yolo(file_path)
            print_progress(0.95, "Finalizing YOLO preview...")
        elif is_image:
            print_progress(0.30, "Analyzing image files...")
            res = get_image_preview(file_path, nrows=15)
            print_progress(0.95, "Finalizing image preview...")
        else:
            print_progress(0.30, "Scanning files...")
            # Estimate or calculate total rows in the source file
            total_rows = None
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".parquet":
                try:
                    print_progress(0.40, "Reading Parquet metadata...")
                    import pyarrow.parquet as pq
                    meta = pq.read_metadata(file_path)
                    total_rows = int(meta.num_rows)
                except Exception:
                    pass
            
            if total_rows is None and ext in [".csv", ".tsv"]:
                try:
                    print_progress(0.40, "Estimating file line count...")
                    with open(file_path, 'rb') as f:
                        lines = 0
                        buf_size = 1024 * 1024
                        read_f = f.read
                        buf = read_f(buf_size)
                        while buf:
                            lines += buf.count(b'\n')
                            buf = read_f(buf_size)
                        total_rows = max(0, lines - 1)
                except Exception:
                    pass
            
            if total_rows is None:
                try:
                    print_progress(0.50, "Loading full dataset...")
                    df_full = load_dataset(file_path)
                    total_rows = len(df_full)
                except Exception:
                    total_rows = 15
            
            print_progress(0.60, "Loading preview rows...")
            if cleaning_actions:
                df_to_clean = load_dataset(file_path, nrows=50000)
                from utils.cleaning import StatefulCleaner
                try:
                    import json
                    actions = json.loads(cleaning_actions)
                    cleaner = StatefulCleaner(actions)
                    cleaner.fit(df_to_clean)
                    df_cleaned = cleaner.transform(df_to_clean, is_training=True)
                    df = df_cleaned.head(15)
                    if total_rows is not None and total_rows > 50000:
                        ratio = len(df_cleaned) / 50000.0
                        total_rows = int(total_rows * ratio)
                    else:
                        total_rows = len(df_cleaned)
                except Exception as e_clean:
                    sys.stderr.write(f"Warning: Failed to apply cleaning actions in preview: {e_clean}\n")
                    df = load_dataset(file_path, nrows=15)
            else:
                df = load_dataset(file_path, nrows=15)

            columns = list(df.columns)
            # Stringify all values and replace NaN/None with empty string so the
            # output is always valid JSON (bare `NaN` is not valid JSON).
            df_preview = df.fillna("").astype(str)
            preview_rows = df_preview.values.tolist()

            # Infer dataset type if not provided by caller
            print_progress(0.85, "Inferring column types...")
            if not dataset_type or dataset_type == "tabular":
                inferred = _infer_dataset_type(df, file_path)
            else:
                inferred = dataset_type

            # Fast column type profiling for the preview screen
            column_types = {}
            try:
                from utils.profiler import profile_dataset
                df_profile = load_dataset(file_path, nrows=1000)
                profile = profile_dataset(df_profile)
                for col_name, col_prof in profile.get("columns", {}).items():
                    column_types[col_name] = col_prof.get("type", "categorical")
            except Exception:
                # Fallback to categorical if profiling fails
                for col_name in columns:
                    column_types[col_name] = "categorical"

            # Extract top unique category values for categorical columns (up to 30 values)
            column_categories = {}
            try:
                df_sample = load_dataset(file_path, nrows=5000)
                for col in columns:
                    c_type = column_types.get(col, "categorical")
                    if c_type in ["categorical", "text"] and col in df_sample.columns:
                        u_vals = df_sample[col].dropna().unique()
                        u_str_vals = [str(v) for v in u_vals if str(v).strip() != ""]
                        column_categories[col] = sorted(u_str_vals[:30])
            except Exception as e_cats:
                sys.stderr.write(f"Warning: Failed to extract column categories: {e_cats}\n")

            res = {
                "columns": columns,
                "preview_rows": preview_rows,
                "inferred_dataset_type": inferred,
                "local_path": file_path,
                "total_rows": total_rows,
                "column_types": column_types,
                "column_categories": column_categories,
                "datetime_range": _get_datetime_bounds(file_path, columns)
            }
            print_progress(0.95, "Finalizing tabular preview...")

        # Inject available_files
        if res and ("error" not in res or res.get("error") is None):
            available_files = []
            if "aura_cache" in file_path:
                # Find the root of this cached dataset directory (the direct child of aura_cache)
                current = file_path
                while True:
                    parent = os.path.dirname(current)
                    if os.path.basename(parent) == "aura_cache":
                        # Walk from the root cached directory (current) to find all files
                        for root, _, filenames in os.walk(current):
                            for f in filenames:
                                if f.endswith(".csv") or f.endswith(".parquet") or f.endswith(".tsv") or f.endswith(".npz") or f.endswith(".xlsx") or f.endswith(".xls"):
                                    available_files.append(os.path.join(root, f))
                        break
                    if parent == current:
                        available_files.append(file_path)
                        break
                    current = parent
            def _sort_key(path):
                name = os.path.basename(path).lower()
                if "train" in name:
                    return (0, name)
                elif "val" in name or "valid" in name:
                    return (2, name)
                elif "test" in name:
                    return (3, name)
                else:
                    return (1, name)
            res["available_files"] = sorted(list(set(available_files)), key=_sort_key)
        else:
            if res is None:
                res = {"error": "Preview generation returned None"}

        return res
    except Exception as e:
        import traceback
        return {"error": f"Failed to generate dataset preview: {str(e)}\n{traceback.format_exc()}"}
