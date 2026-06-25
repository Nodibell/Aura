import pandas as pd
from utils.loader import load_dataset, download_dataset
from utils.event_bus import publish_progress

class DataEngine:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        if file_path.startswith("http://") or file_path.startswith("https://"):
            publish_progress(0.05, "Downloading remote dataset...")
            file_path = download_dataset(file_path)
        
        publish_progress(0.15, "Loading dataset file...")
        return load_dataset(file_path)

    @staticmethod
    def merge_datasets(file1: str, file2: str, key1: str, key2: str, join_type: str, output_path: str) -> pd.DataFrame:
        df1 = load_dataset(file1)
        df2 = load_dataset(file2)
        
        df1.columns = df1.columns.str.strip()
        df2.columns = df2.columns.str.strip()
        
        merged = pd.merge(df1, df2, left_on=key1, right_on=key2, how=join_type)
        merged.to_csv(output_path, index=False)
        return merged
