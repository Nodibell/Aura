import os
import polars as pl
from utils.loader import download_dataset
from utils.event_bus import publish_progress

class DataEngine:
    @staticmethod
    def load_data(file_path: str) -> pl.DataFrame:
        if file_path.startswith("http://") or file_path.startswith("https://"):
            publish_progress(0.05, "Downloading remote dataset...")
            file_path = download_dataset(file_path)
        
        publish_progress(0.15, "Loading dataset file...")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".parquet":
            return pl.read_parquet(file_path)
        elif ext in [".xlsx", ".xls"]:
            try:
                return pl.read_excel(file_path)
            except Exception:
                # Fallback to pandas if excel parsing libraries for polars are missing
                import pandas as pd
                return pl.from_pandas(pd.read_excel(file_path))
        elif ext == ".json":
            return pl.read_json(file_path)
        elif ext == ".jsonl":
            return pl.read_ndjson(file_path)
        elif ext == ".tsv":
            return pl.read_csv(file_path, separator="\t")
        else:
            # Default to CSV
            try:
                return pl.read_csv(file_path)
            except Exception:
                import pandas as pd
                return pl.from_pandas(pd.read_csv(file_path))

    @staticmethod
    def merge_datasets(file1: str, file2: str, key1: str, key2: str, join_type: str, output_path: str) -> pl.DataFrame:
        df1 = DataEngine.load_data(file1)
        df2 = DataEngine.load_data(file2)
        
        # Clean column names (strip whitespace)
        df1 = df1.rename({col: col.strip() for col in df1.columns})
        df2 = df2.rename({col: col.strip() for col in df2.columns})
        
        how = join_type.lower()
        if how == "right":
            try:
                merged = df1.join(df2, left_on=key1, right_on=key2, how="right")
            except Exception:
                # Fallback to swapped left join if right join is unsupported
                merged = df2.join(df1, left_on=key2, right_on=key1, how="left")
        else:
            merged = df1.join(df2, left_on=key1, right_on=key2, how=how)
            
        merged.write_csv(output_path)
        return merged
