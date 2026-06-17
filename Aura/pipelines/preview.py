import os
import sys
import pandas as pd
from utils.loader import download_dataset, load_dataset, _infer_dataset_type
from utils.charts import get_image_preview

def analyze_preview(file_path, dataset_type=None):
    try:
        # Check if URL input
        if file_path.startswith("http://") or file_path.startswith("https://"):
            try:
                file_path = download_dataset(file_path)
            except Exception as download_err:
                return {"error": f"Failed to download dataset: {str(download_err)}"}

        # Determine if it's image dataset
        is_image = False
        if dataset_type == "image":
            is_image = True
        elif not dataset_type or dataset_type == "tabular":
            if os.path.isdir(file_path):
                is_image = True
            else:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".npz":
                    is_image = True

        if is_image:
            res = get_image_preview(file_path, nrows=15)
        else:
            # Estimate or calculate total rows in the source file
            total_rows = None
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".parquet":
                try:
                    import pyarrow.parquet as pq
                    meta = pq.read_metadata(file_path)
                    total_rows = int(meta.num_rows)
                except Exception:
                    pass
            
            if total_rows is None and ext in [".csv", ".tsv"]:
                try:
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
                    df_full = load_dataset(file_path)
                    total_rows = len(df_full)
                except Exception:
                    total_rows = 15
            
            df = load_dataset(file_path, nrows=15)

            columns = list(df.columns)
            # Convert df values to list of lists, handling NaN/None
            df_preview = df.where(pd.notnull(df), None)
            preview_rows = df_preview.values.tolist()

            # Infer dataset type if not provided by caller
            if not dataset_type or dataset_type == "tabular":
                inferred = _infer_dataset_type(df, file_path)
            else:
                inferred = dataset_type

            res = {
                "columns": columns,
                "preview_rows": preview_rows,
                "inferred_dataset_type": inferred,
                "local_path": file_path,
                "total_rows": total_rows
            }

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
                                if f.endswith(".csv") or f.endswith(".parquet") or f.endswith(".tsv") or f.endswith(".npz"):
                                    available_files.append(os.path.join(root, f))
                        break
                    if parent == current:
                        available_files.append(file_path)
                        break
                    current = parent
            else:
                available_files.append(file_path)
            res["available_files"] = sorted(list(set(available_files)))
        else:
            if res is None:
                res = {"error": "Preview generation returned None"}

        return res
    except Exception as e:
        import traceback
        return {"error": f"Failed to generate dataset preview: {str(e)}\n{traceback.format_exc()}"}
