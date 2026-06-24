import numpy as np
import pandas as pd

class StatefulCleaner:
    def __init__(self, actions_list):
        self.actions = actions_list
        self.imputers = {}
        self.outlier_bounds = {}
        
    def fit(self, df):
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType")
            if col not in df.columns:
                continue
            
            # Imputation fitting
            if act_type == "impute_mean":
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    self.imputers[col] = df[col].mean()
            elif act_type == "impute_median":
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    self.imputers[col] = df[col].median()
            elif act_type == "impute_mode":
                mode_val = df[col].mode()
                self.imputers[col] = mode_val[0] if not mode_val.empty else "missing"
            elif act_type == "impute_knn":
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    from sklearn.impute import KNNImputer
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c != "__is_test"]
                    imputer = KNNImputer(n_neighbors=5)
                    imputer.fit(df[numeric_cols])
                    self.imputers[col] = (imputer, numeric_cols)
            elif act_type == "impute_mice":
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c != "__is_test"]
                    imputer = IterativeImputer(random_state=42, max_iter=10)
                    imputer.fit(df[numeric_cols])
                    self.imputers[col] = (imputer, numeric_cols)
                    
            # Outlier fitting
            if act_type == "clip_outliers" or act_type == "drop_outliers":
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    q25 = df[col].quantile(0.25)
                    q75 = df[col].quantile(0.75)
                    iqr = q75 - q25
                    self.outlier_bounds[col] = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
            elif act_type == "isolation_forest":
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    from sklearn.ensemble import IsolationForest
                    temp_col = df[col].fillna(df[col].median())
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    iso.fit(temp_col.to_frame())
                    self.outlier_bounds[col] = iso
                    
    def transform(self, df, is_training=True):
        df_out = df.copy()
        # 1. Drop actions
        drop_cols = [act.get("column") for act in self.actions if act.get("actionType") == "drop" and act.get("column") in df_out.columns]
        if drop_cols:
            df_out = df_out.drop(columns=drop_cols)
            
        # 2. Imputation actions
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType")
            if col not in df_out.columns:
                continue
            
            if act_type in ["impute_mean", "impute_median", "impute_mode"]:
                if col in self.imputers:
                    df_out[col] = df_out[col].fillna(self.imputers[col])
            elif act_type == "impute_knn":
                if col in self.imputers:
                    imputer, numeric_cols = self.imputers[col]
                    missing_cols = [c for c in numeric_cols if c not in df_out.columns]
                    for mc in missing_cols:
                        df_out[mc] = np.nan
                    df_out[numeric_cols] = imputer.transform(df_out[numeric_cols])
            elif act_type == "impute_mice":
                if col in self.imputers:
                    imputer, numeric_cols = self.imputers[col]
                    missing_cols = [c for c in numeric_cols if c not in df_out.columns]
                    for mc in missing_cols:
                        df_out[mc] = np.nan
                    df_out[numeric_cols] = imputer.transform(df_out[numeric_cols])
                    
        # 3. Outlier actions
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType")
            if col not in df_out.columns:
                continue
            
            if act_type == "clip_outliers":
                if col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    df_out[col] = df_out[col].clip(lower, upper)
            elif act_type == "drop_outliers":
                if is_training and col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
            elif act_type == "isolation_forest":
                if col in self.outlier_bounds:
                    iso = self.outlier_bounds[col]
                    temp_col = df_out[col].fillna(df_out[col].median() if col in df_out.columns else 0)
                    preds = iso.predict(temp_col.to_frame())
                    if is_training:
                        df_out = df_out[preds == 1]
                        
        # 4. Feature engineering transformations
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType", "")
            if col not in df_out.columns:
                continue
                
            if act_type == "transform_log":
                if pd.api.types.is_numeric_dtype(df_out[col].dtype):
                    df_out[f"{col}_log"] = np.log1p(np.maximum(0.0, df_out[col]))
            elif act_type == "transform_power":
                if pd.api.types.is_numeric_dtype(df_out[col].dtype):
                    df_out[f"{col}_power"] = np.square(df_out[col])
            elif act_type.startswith("transform_interaction:"):
                other_col = act_type.split(":", 1)[1]
                if other_col in df_out.columns:
                    if pd.api.types.is_numeric_dtype(df_out[col].dtype) and pd.api.types.is_numeric_dtype(df_out[other_col].dtype):
                        df_out[f"{col}_x_{other_col}"] = df_out[col] * df_out[other_col]
            elif act_type == "transform_date":
                try:
                    dt_series = pd.to_datetime(df_out[col], errors='coerce')
                    df_out[f"{col}_year"] = dt_series.dt.year.fillna(2000).astype(int)
                    df_out[f"{col}_month"] = dt_series.dt.month.fillna(1).astype(int)
                    df_out[f"{col}_day"] = dt_series.dt.day.fillna(1).astype(int)
                    df_out[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
                    df_out[f"{col}_hour"] = dt_series.dt.hour.fillna(0).astype(int)
                except Exception:
                    pass
        return df_out
