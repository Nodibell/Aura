import numpy as np
import pandas as pd

class CleaningStrategy:
    def fit(self, df, col):
        pass
    def transform(self, df, col, is_training):
        return df

class DropColumnStrategy(CleaningStrategy):
    def transform(self, df, col, is_training):
        if col in df.columns:
            return df.drop(columns=[col])
        return df

class MeanImputation(CleaningStrategy):
    def __init__(self):
        self.mean_val = None
    def fit(self, df, col):
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            self.mean_val = df[col].mean()
    def transform(self, df, col, is_training):
        if self.mean_val is not None and col in df.columns:
            df[col] = df[col].fillna(self.mean_val)
        return df

class MedianImputation(CleaningStrategy):
    def __init__(self):
        self.median_val = None
    def fit(self, df, col):
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            self.median_val = df[col].median()
    def transform(self, df, col, is_training):
        if self.median_val is not None and col in df.columns:
            df[col] = df[col].fillna(self.median_val)
        return df

class ModeImputation(CleaningStrategy):
    def __init__(self):
        self.mode_val = None
    def fit(self, df, col):
        mode_series = df[col].mode()
        self.mode_val = mode_series[0] if not mode_series.empty else "missing"
    def transform(self, df, col, is_training):
        if self.mode_val is not None and col in df.columns:
            df[col] = df[col].fillna(self.mode_val)
        return df

class KNNImputerStrategy(CleaningStrategy):
    def __init__(self):
        self.imputer = None
        self.numeric_cols = []
    def fit(self, df, col):
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            from sklearn.impute import KNNImputer
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.numeric_cols = [c for c in self.numeric_cols if c != "__is_test"]
            self.imputer = KNNImputer(n_neighbors=5)
            self.imputer.fit(df[self.numeric_cols])
    def transform(self, df, col, is_training):
        if self.imputer is not None and col in df.columns:
            missing_cols = [c for c in self.numeric_cols if c not in df.columns]
            for mc in missing_cols:
                df[mc] = np.nan
            df[self.numeric_cols] = self.imputer.transform(df[self.numeric_cols])
        return df

class MiceImputerStrategy(CleaningStrategy):
    def __init__(self):
        self.imputer = None
        self.numeric_cols = []
    def fit(self, df, col):
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.numeric_cols = [c for c in self.numeric_cols if c != "__is_test"]
            self.imputer = IterativeImputer(random_state=42, max_iter=10)
            self.imputer.fit(df[self.numeric_cols])
    def transform(self, df, col, is_training):
        if self.imputer is not None and col in df.columns:
            missing_cols = [c for c in self.numeric_cols if c not in df.columns]
            for mc in missing_cols:
                df[mc] = np.nan
            df[self.numeric_cols] = self.imputer.transform(df[self.numeric_cols])
        return df

class ClipOutliers(CleaningStrategy):
    def __init__(self):
        self.lower = None
        self.upper = None
    def fit(self, df, col):
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = q75 - q25
            self.lower = q25 - 1.5 * iqr
            self.upper = q75 + 1.5 * iqr
    def transform(self, df, col, is_training):
        if self.lower is not None and col in df.columns:
            df[col] = df[col].clip(self.lower, self.upper)
        return df

class DropOutliers(CleaningStrategy):
    def __init__(self):
        self.lower = None
        self.upper = None
    def fit(self, df, col):
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = q75 - q25
            self.lower = q25 - 1.5 * iqr
            self.upper = q75 + 1.5 * iqr
    def transform(self, df, col, is_training):
        if is_training and self.lower is not None and col in df.columns:
            df = df[(df[col] >= self.lower) & (df[col] <= self.upper)]
        return df

class IsolationForestOutliers(CleaningStrategy):
    def __init__(self):
        self.iso = None
    def fit(self, df, col):
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            from sklearn.ensemble import IsolationForest
            temp_col = df[col].fillna(df[col].median())
            self.iso = IsolationForest(contamination=0.05, random_state=42)
            self.iso.fit(temp_col.to_frame())
    def transform(self, df, col, is_training):
        if self.iso is not None and col in df.columns:
            temp_col = df[col].fillna(df[col].median() if col in df.columns else 0)
            preds = self.iso.predict(temp_col.to_frame())
            if is_training:
                df = df[preds == 1]
        return df

class LogTransform(CleaningStrategy):
    def transform(self, df, col, is_training):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype):
            df[f"{col}_log"] = np.log1p(np.maximum(0.0, df[col]))
        return df

class PowerTransform(CleaningStrategy):
    def transform(self, df, col, is_training):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype):
            df[f"{col}_power"] = np.square(df[col])
        return df

class InteractionTransform(CleaningStrategy):
    def __init__(self, other_col):
        self.other_col = other_col
    def transform(self, df, col, is_training):
        if col in df.columns and self.other_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col].dtype) and pd.api.types.is_numeric_dtype(df[self.other_col].dtype):
                df[f"{col}_x_{self.other_col}"] = df[col] * df[self.other_col]
        return df

class DateTransform(CleaningStrategy):
    def transform(self, df, col, is_training):
        if col in df.columns:
            try:
                dt_series = pd.to_datetime(df[col], errors='coerce')
                df[f"{col}_year"] = dt_series.dt.year.fillna(2000).astype(int)
                df[f"{col}_month"] = dt_series.dt.month.fillna(1).astype(int)
                df[f"{col}_day"] = dt_series.dt.day.fillna(1).astype(int)
                df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
                df[f"{col}_hour"] = dt_series.dt.hour.fillna(0).astype(int)
            except Exception:
                pass
        return df

class StatefulCleaner:
    def __init__(self, actions_list):
        self.actions = actions_list
        self.strategies = {}

    @property
    def imputers(self):
        imp_dict = {}
        for (col, act_type), strat in self.strategies.items():
            if act_type == "impute_mean":
                imp_dict[col] = strat.mean_val
            elif act_type == "impute_median":
                imp_dict[col] = strat.median_val
            elif act_type == "impute_mode":
                imp_dict[col] = strat.mode_val
        return imp_dict
        
    def _create_strategy(self, action_type):
        if action_type == "drop":
            return DropColumnStrategy()
        elif action_type == "impute_mean":
            return MeanImputation()
        elif action_type == "impute_median":
            return MedianImputation()
        elif action_type == "impute_mode":
            return ModeImputation()
        elif action_type == "impute_knn":
            return KNNImputerStrategy()
        elif action_type == "impute_mice":
            return MiceImputerStrategy()
        elif action_type == "clip_outliers":
            return ClipOutliers()
        elif action_type == "drop_outliers":
            return DropOutliers()
        elif action_type == "isolation_forest":
            return IsolationForestOutliers()
        elif action_type == "transform_log":
            return LogTransform()
        elif action_type == "transform_power":
            return PowerTransform()
        elif action_type.startswith("transform_interaction:"):
            other_col = action_type.split(":", 1)[1]
            return InteractionTransform(other_col)
        elif action_type == "transform_date":
            return DateTransform()
        return CleaningStrategy()

    def fit(self, df):
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType")
            if col not in df.columns:
                continue
            
            strategy = self._create_strategy(act_type)
            strategy.fit(df, col)
            self.strategies[(col, act_type)] = strategy

    def transform(self, df, is_training=True):
        df_out = df.copy()
        
        # 0. Rename actions (run first)
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType")
            if act_type and act_type.startswith("rename:") and col in df_out.columns:
                new_name = act_type.split(":", 1)[1].strip()
                if new_name:
                    df_out = df_out.rename(columns={col: new_name})
                    # Update column name for any subsequent actions on this column
                    for sub_act in self.actions:
                        if sub_act.get("column") == col:
                            sub_act["column"] = new_name
                            
        # 1. Drop actions
        drop_cols = [act.get("column") for act in self.actions if act.get("actionType") == "drop" and act.get("column") in df_out.columns]
        if drop_cols:
            df_out = df_out.drop(columns=drop_cols)
            
        # 2. Imputation actions
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType")
            if col not in df_out.columns or not act_type.startswith("impute_"):
                continue
            strategy = self.strategies.get((col, act_type))
            if strategy:
                df_out = strategy.transform(df_out, col, is_training)
                
        # 3. Outlier actions
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType")
            if col not in df_out.columns or act_type not in ["clip_outliers", "drop_outliers", "isolation_forest"]:
                continue
            strategy = self.strategies.get((col, act_type))
            if strategy:
                df_out = strategy.transform(df_out, col, is_training)
                
        # 4. Feature engineering transformations
        for act in self.actions:
            col = act.get("column")
            act_type = act.get("actionType", "")
            if col not in df_out.columns or not act_type.startswith("transform_"):
                continue
            strategy = self.strategies.get((col, act_type))
            if strategy:
                df_out = strategy.transform(df_out, col, is_training)
                
        return df_out
