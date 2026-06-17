import sys
import json
import numpy as np
import pandas as pd

def print_progress(fraction, message):
    sys.stderr.write(f"PROGRESS: {fraction:.2f}:{message}\n")
    sys.stderr.flush()

def clean_nan(obj):
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return f"<Binary Data: {len(obj)} bytes>"
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_nan(x) for x in obj)
    elif isinstance(obj, float) or isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj

def _generate_reproduction_code(dataset_path, dataset_type, target_col, exclude_cols,
                               task_type, feature_names, model_name, model_path,
                               numeric_cols, categorical_cols, text_cols, time_col=None):
    import os
    exclude_list = list(exclude_cols) if exclude_cols else []
    num_list = numeric_cols if numeric_cols else []
    cat_list = categorical_cols if categorical_cols else []
    txt_list = text_cols if text_cols else []
    model_file_name = os.path.basename(model_path) if model_path else "model.joblib"

    if dataset_type == "timeseries":
        code = f'''# Aura - Time Series Reproduction Pipeline
# Generated for target: '{target_col}' ({task_type})

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# 1. Load the dataset
DATA_PATH = r"{dataset_path}"
if not os.path.exists(DATA_PATH):
    print(f"Warning: File not found at {{DATA_PATH}}. Please update DATA_PATH to your local file path.")

df = pd.read_csv(DATA_PATH) if not DATA_PATH.endswith('.parquet') else pd.read_parquet(DATA_PATH)

# Exclude columns
exclude_cols = {exclude_list}
df = df.drop(columns=exclude_cols, errors='ignore')

# Sort chronologically by time column
time_col = "{time_col}"
if time_col in df.columns:
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)

# Preprocess target and features
target_col = "{target_col}"
y_raw = df[target_col]
if "{task_type}" == "classification":
    y = y_raw.ffill().bfill().astype(str).to_numpy()
else:
    y_series = y_raw.interpolate(method='linear').ffill().bfill().fillna(0)
    y = y_series.to_numpy()

# 2. Create lag features
X_df = pd.DataFrame(index=df.index)
if "{task_type}" != "classification":
    for lag in [1, 2, 3, 7]:
        if len(df) > lag:
            X_df[f"target_lag_{{lag}}"] = y_series.shift(lag)
    for roll in [3, 7]:
        if len(df) > roll:
            X_df[f"target_roll_mean_{{roll}}"] = y_series.shift(1).rolling(roll).mean()
            
numeric_cols = {num_list}
for col in numeric_cols:
    if col != target_col and col != time_col and col in df.columns:
        col_series = df[col].interpolate(method='linear').ffill().bfill().fillna(0)
        for lag in [1, 3]:
            if len(df) > lag:
                X_df[f"{{col}}_lag_{{lag}}"] = col_series.shift(lag)

X_df = X_df.bfill().ffill().fillna(0)
X_processed = X_df.to_numpy()

# 3. Load Trained Model and Predict
MODEL_PATH = r"{model_file_name}"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Successfully loaded model from:", MODEL_PATH)
    preds = model.predict(X_processed)
    print("First 10 predictions:", preds[:10])
    
    if y is not None:
        if "{task_type}" == "classification":
            score = accuracy_score(y, preds)
            print("Reproduction Accuracy:", score)
        else:
            score = r2_score(y, preds)
            print("Reproduction R2:", score)
else:
    print(f"Model file not found at {{MODEL_PATH}}. Please place it in the same directory.")
'''
        return code

    elif dataset_type == "nlp":
        code = f'''# Aura - Text/NLP Reproduction Pipeline
# Generated for target: '{target_col}' ({task_type})

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, r2_score, f1_score

# 1. Load the dataset
DATA_PATH = r"{dataset_path}"
if not os.path.exists(DATA_PATH):
    print(f"Warning: File not found at {{DATA_PATH}}.")

df = pd.read_csv(DATA_PATH) if not DATA_PATH.endswith('.parquet') else pd.read_parquet(DATA_PATH)

# Exclude columns
exclude_cols = {exclude_list}
df = df.drop(columns=exclude_cols, errors='ignore')

# Drop target missing values
target_col = "{target_col}"
if target_col in df.columns:
    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)
    y_raw = df_clean[target_col]
    if "{task_type}" == "classification":
        y = y_raw.ffill().bfill().astype(str).to_numpy()
    else:
        y = y_raw.fillna(y_raw.median()).to_numpy()
    X = df_clean.drop(columns=[target_col])
else:
    X = df
    y = None

# Identify text columns (exclude target)
text_cols = [c for c in X.select_dtypes(exclude=[np.number]).columns if c != target_col]

# Fill missing text values
X_imputed = X.copy()
for col in text_cols:
    X_imputed[col] = X_imputed[col].fillna("missing").astype(str)

# Preprocess text (TF-IDF Vectorization)
processed_parts = []
for col in text_cols:
    vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    X_text_encoded = vectorizer.fit_transform(X_imputed[col]).toarray()
    encoded_names = [f"{{col}}_tfidf_{{word}}" for word in vectorizer.get_feature_names_out()]
    df_text = pd.DataFrame(X_text_encoded, columns=encoded_names, index=X.index)
    processed_parts.append(df_text)

if processed_parts:
    X_processed = pd.concat(processed_parts, axis=1)
else:
    X_processed = X_imputed

# Ensure columns align with expectations
expected_features = {feature_names}
for col in expected_features:
    if col not in X_processed.columns:
        X_processed[col] = 0.0
X_processed = X_processed[expected_features]

# 3. Load Trained Model and Predict
MODEL_PATH = r"{model_file_name}"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Successfully loaded model from:", MODEL_PATH)
    preds = model.predict(X_processed)
    print("First 10 predictions:", preds[:10])
    
    if y is not None:
        if "{task_type}" == "classification":
            score = accuracy_score(y, preds)
            print("Reproduction Accuracy:", score)
        else:
            score = r2_score(y, preds)
            print("Reproduction R2:", score)
else:
    print(f"Model file not found at {{MODEL_PATH}}.")
'''
        return code

    elif dataset_type == "image":
        code = f'''# Aura - Image Classification Reproduction Pipeline
# Generated for target: '{target_col}' ({task_type})

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# 1. Load the dataset (NPZ or table)
DATA_PATH = r"{dataset_path}"
print(f"Loading image data from: {{DATA_PATH}}")

if DATA_PATH.endswith('.npz'):
    npz = np.load(DATA_PATH, allow_pickle=True)
    keys = list(npz.keys())
    x_keys = [k for k in keys if k.lower() in ["x", "x_train", "train_x", "data", "features", "images"]]
    y_keys = [k for k in keys if k.lower() in ["y", "y_train", "train_y", "labels", "target", "classes"]]
    X_arr = npz[x_keys[0]] if x_keys else npz[keys[0]]
    y_arr = npz[y_keys[0]] if y_keys else None
    
    # Flatten X if needed
    if len(X_arr.shape) > 2:
        X_arr = X_arr.reshape((X_arr.shape[0], -1))
        
    X_processed = pd.DataFrame(X_arr)
    y = y_arr.flatten() if y_arr is not None else None
else:
    df = pd.read_csv(DATA_PATH)
    target_col = "{target_col}"
    if target_col in df.columns:
        X_processed = df.drop(columns=[target_col])
        y = df[target_col].to_numpy()
    else:
        X_processed = df
        y = None

# Ensure columns align with expectations
expected_features = {feature_names}
for col in expected_features:
    if col not in X_processed.columns:
        X_processed[col] = 0.0
X_processed = X_processed[expected_features]

# Load Trained Model and Predict
MODEL_PATH = r"{model_file_name}"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_processed)
    print("First 10 predictions:", preds[:10])
    if y is not None:
        print("Reproduction Accuracy:", accuracy_score(y, preds))
else:
    print(f"Model file not found at {{MODEL_PATH}}.")
'''
        return code

    else:
        # Standard Tabular Template
        code = f'''# Aura - Tabular Reproduction Pipeline
# Generated for target: '{target_col}' ({task_type})

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score

# 1. Load the dataset
DATA_PATH = r"{dataset_path}"
if not os.path.exists(DATA_PATH):
    print(f"Warning: File not found at {{DATA_PATH}}. Please update DATA_PATH to your local file path.")

df = pd.read_csv(DATA_PATH) if not DATA_PATH.endswith('.parquet') else pd.read_parquet(DATA_PATH)

# Exclude ignored columns
exclude_cols = {exclude_list}
df = df.drop(columns=exclude_cols, errors='ignore')

# Drop target missing values
target_col = "{target_col}"
if target_col in df.columns:
    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)
    y_raw = df_clean[target_col]
    if "{task_type}" == "classification":
        y = y_raw.fillna(y_raw.mode()[0] if not y_raw.mode().empty else "missing").astype(str).to_numpy()
    else:
        y = y_raw.fillna(y_raw.median()).to_numpy()
    X = df_clean.drop(columns=[target_col])
else:
    X = df
    y = None

# 2. Impute and Preprocess Features
X_imputed = X.copy()
numeric_cols = {num_list}
categorical_cols = {cat_list}
text_cols = {txt_list}

# Impute numeric columns with median
for col in numeric_cols:
    if col in X_imputed.columns:
        X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())

# Impute categorical/text columns with mode
for col in (categorical_cols + text_cols):
    if col in X_imputed.columns:
        mode_val = X_imputed[col].mode()
        mode_choice = mode_val[0] if not mode_val.empty else "missing"
        X_imputed[col] = X_imputed[col].fillna(mode_choice).astype(str)

processed_parts = []
if numeric_cols:
    df_num = X_imputed[[c for c in numeric_cols if c in X_imputed.columns]]
    processed_parts.append(df_num)

if categorical_cols:
    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_imputed[[c for c in categorical_cols if c in X_imputed.columns]])
    encoded_names = encoder.get_feature_names_out([c for c in categorical_cols if c in X_imputed.columns])
    df_cat = pd.DataFrame(X_cat_encoded, columns=encoded_names, index=X.index)
    processed_parts.append(df_cat)

if text_cols:
    for col in text_cols:
        if col in X_imputed.columns:
            vectorizer = TfidfVectorizer(max_features=15)
            X_text_encoded = vectorizer.fit_transform(X_imputed[col]).toarray()
            encoded_names = [f"{{col}}_tfidf_{{word}}" for word in vectorizer.get_feature_names_out()]
            df_text = pd.DataFrame(X_text_encoded, columns=encoded_names, index=X.index)
            processed_parts.append(df_text)

X_processed = pd.concat(processed_parts, axis=1)
# Ensure columns match what the model expects
expected_features = {feature_names}
# Re-align features
for col in expected_features:
    if col not in X_processed.columns:
        X_processed[col] = 0.0
X_processed = X_processed[expected_features]

# 3. Load Trained Model and Predict
MODEL_PATH = r"{model_file_name}"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Successfully loaded model from:", MODEL_PATH)
    
    preds = model.predict(X_processed)
    print("First 10 predictions:", preds[:10])
    
    if y is not None:
        if "{task_type}" == "classification":
            score = accuracy_score(y, preds)
            print("Reproduction Accuracy:", score)
        else:
            score = r2_score(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            print("Reproduction R2:", score)
            print("Reproduction RMSE:", rmse)
else:
    print(f"Model file not found at {{MODEL_PATH}}. Please place it in the same directory as this script.")
'''
        return code

def _export_model_and_code(model_obj, model_path, code_path, dataset_path, dataset_type,
                           target_col, exclude_cols, task_type, feature_names,
                           model_name, numeric_cols=None, categorical_cols=None, text_cols=None, time_col=None):
    import joblib
    try:
        # Save model
        if model_path and model_obj is not None:
            joblib.dump(model_obj, model_path)
            sys.stderr.write(f"Model exported successfully to {model_path}\n")
            
        # Save reproduction code
        if code_path:
            code_content = _generate_reproduction_code(
                dataset_path, dataset_type, target_col, exclude_cols,
                task_type, feature_names, model_name, model_path,
                numeric_cols, categorical_cols, text_cols, time_col=time_col
            )
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            sys.stderr.write(f"Reproduction code exported successfully to {code_path}\n")
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to export model/code: {str(e)}\n")
