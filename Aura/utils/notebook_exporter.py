"""
notebook_exporter.py — Pure-stdlib Jupyter Notebook generator for Aura.

No `nbformat` dependency required. Writes a valid .ipynb JSON file that
reproduces the full pipeline: data loading → cleaning → training → evaluation.

Called by analyze.py when --notebook-export-path is provided.
"""

from __future__ import annotations

import json
import os
from typing import Any


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def _code_cell(source: str | list[str]) -> dict:
    if isinstance(source, list):
        source = "\n".join(source)
    lines = [l + "\n" for l in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def _md_cell(source: str) -> dict:
    lines = [l + "\n" for l in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def _notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_notebook(config: dict[str, Any],
                      result: dict[str, Any],
                      output_path: str) -> None:
    """
    Generate a reproducible Jupyter Notebook from an Aura analysis result.

    Args:
        config: Serialised AnalysisConfig dict (snakeCase keys from JSON).
        result: Serialised AnalysisResult dict returned by analyze.py.
        output_path: Destination .ipynb file path.
    """
    cells: list[dict] = []

    # ─── 0. Title ───────────────────────────────────────────────────────────
    file_path   = config.get("train_file_path") or config.get("trainFilePath") or "dataset.csv"
    target_col  = config.get("target_column")   or config.get("targetColumn")  or "target"
    task_type   = result.get("task_type", "classification")
    dataset_type = config.get("dataset_type") or config.get("datasetType") or "tabular"
    best_model  = result.get("metrics", {}).get("model", "Best Model")
    score_type  = result.get("metrics", {}).get("score_type", "score")
    score_val   = result.get("metrics", {}).get("score", 0.0)

    cells.append(_md_cell(
        f"# Aura — Reproducible Pipeline\n\n"
        f"**Task**: `{task_type}` · **Dataset type**: `{dataset_type}`  \n"
        f"**Target column**: `{target_col}`  \n"
        f"**Best model**: `{best_model}` ({score_type}: {score_val:.4f})  \n\n"
        f"> Generated automatically by **Aura**. Run each cell in order."
    ))

    # ─── 1. Dependencies ────────────────────────────────────────────────────
    cells.append(_md_cell("## 1. Dependencies"))
    cells.append(_code_cell(
        "# Install required packages (skip if already installed)\n"
        "# !pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn"
    ))
    cells.append(_code_cell([
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error",
        "import joblib",
        "",
        f"FILE_PATH   = {file_path!r}",
        f"TARGET_COL  = {target_col!r}",
        f"TASK_TYPE   = {task_type!r}",
        f"RANDOM_SEED = 42",
    ]))

    # ─── 2. Data Loading ────────────────────────────────────────────────────
    cells.append(_md_cell("## 2. Data Loading"))
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        load_code = "df = pd.read_parquet(FILE_PATH)"
    elif ext in (".xls", ".xlsx"):
        load_code = "df = pd.read_excel(FILE_PATH)"
    elif ext == ".tsv":
        load_code = "df = pd.read_csv(FILE_PATH, sep='\\t')"
    else:
        load_code = "df = pd.read_csv(FILE_PATH)"
    cells.append(_code_cell([
        load_code,
        "print(f'Shape: {df.shape}')",
        "df.info()",
        "df.head()"
    ]))

    # ─── 3. EDA Summary ─────────────────────────────────────────────────────
    cells.append(_md_cell("## 3. Exploratory Data Analysis"))
    cells.append(_code_cell([
        "# Descriptive statistics",
        "df.describe(include='all')"
    ]))
    cells.append(_code_cell([
        "# Missing values",
        "missing = df.isnull().sum().sort_values(ascending=False)",
        "missing[missing > 0]"
    ]))

    # Correlations only for tabular/numeric
    if dataset_type in ("tabular", "timeseries"):
        cells.append(_code_cell([
            "# Correlation heatmap",
            "fig, ax = plt.subplots(figsize=(10, 8))",
            "numeric_df = df.select_dtypes(include='number')",
            "sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', ax=ax, cmap='coolwarm')",
            "ax.set_title('Correlation Matrix')",
            "plt.tight_layout()",
            "plt.show()"
        ]))

    # ─── 4. Data Cleaning ───────────────────────────────────────────────────
    cells.append(_md_cell("## 4. Data Cleaning"))
    cleaning_actions = config.get("cleaning_actions") or config.get("cleaningActions") or []
    cleaning_lines = [
        "# ── Reproduce Aura cleaning steps ──",
        "df_clean = df.copy()",
    ]
    if cleaning_actions:
        for action in cleaning_actions:
            col    = action.get("column", "")
            atype  = action.get("actionType") or action.get("action_type", "")
            if atype == "drop":
                cleaning_lines.append(f"df_clean = df_clean.dropna(subset=[{col!r}])")
            elif atype == "impute_mean":
                cleaning_lines.append(
                    f"df_clean[{col!r}] = df_clean[{col!r}].fillna(df_clean[{col!r}].mean())"
                )
            elif atype == "impute_median":
                cleaning_lines.append(
                    f"df_clean[{col!r}] = df_clean[{col!r}].fillna(df_clean[{col!r}].median())"
                )
            elif atype == "impute_mode":
                cleaning_lines.append(
                    f"df_clean[{col!r}] = df_clean[{col!r}].fillna(df_clean[{col!r}].mode()[0])"
                )
            elif atype == "clip_outliers":
                cleaning_lines.append(
                    f"q1, q3 = df_clean[{col!r}].quantile(0.25), df_clean[{col!r}].quantile(0.75)\n"
                    f"iqr = q3 - q1\n"
                    f"df_clean[{col!r}] = df_clean[{col!r}].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)"
                )
    else:
        cleaning_lines.append(
            "# No explicit cleaning actions were applied — fill numeric NaNs with median as a safe default"
        )
        cleaning_lines.append(
            "for col in df_clean.select_dtypes(include='number').columns:\n"
            "    df_clean[col] = df_clean[col].fillna(df_clean[col].median())"
        )
    cleaning_lines.append("print(f'Rows after cleaning: {len(df_clean)}')")
    cells.append(_code_cell(cleaning_lines))

    # ─── 5. Feature Engineering ─────────────────────────────────────────────
    cells.append(_md_cell("## 5. Feature Engineering"))
    excl = list(config.get("excluded_columns") or config.get("excludedColumns") or [])
    cells.append(_code_cell([
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import OneHotEncoder",
        "",
        f"EXCLUDE_COLS = {excl!r}",
        "",
        "feature_cols = [c for c in df_clean.columns",
        f"                if c != TARGET_COL and c not in EXCLUDE_COLS]",
        "X = df_clean[feature_cols]",
        "y = df_clean[TARGET_COL]",
        "",
        "numeric_features   = X.select_dtypes(include='number').columns.tolist()",
        "categorical_features = X.select_dtypes(include='object').columns.tolist()",
        "",
        "numeric_transformer = Pipeline([",
        "    ('imputer', SimpleImputer(strategy='median')),",
        "    ('scaler',  StandardScaler()),",
        "])",
        "categorical_transformer = Pipeline([",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),",
        "    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),",
        "])",
        "preprocessor = ColumnTransformer([",
        "    ('num', numeric_transformer, numeric_features),",
        "    ('cat', categorical_transformer, categorical_features),",
        "], remainder='drop')",
        "",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=RANDOM_SEED",
        ")",
        "print(f'Train: {X_train.shape}, Test: {X_test.shape}')"
    ]))

    # ─── 6. Model Training ──────────────────────────────────────────────────
    cells.append(_md_cell(f"## 6. Model Training — {best_model}"))
    # Pick model import based on best_model string
    bm = best_model.lower()
    if "xgboost" in bm or "xgb" in bm:
        import_line  = "from xgboost import XGBClassifier, XGBRegressor"
        clf_line     = ("XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED)"
                        if task_type == "classification"
                        else "XGBRegressor(random_state=RANDOM_SEED)")
    elif "lightgbm" in bm or "lgbm" in bm:
        import_line  = "from lightgbm import LGBMClassifier, LGBMRegressor"
        clf_line     = ("LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)"
                        if task_type == "classification"
                        else "LGBMRegressor(random_state=RANDOM_SEED, verbose=-1)")
    elif "random forest" in bm:
        import_line  = "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor"
        clf_line     = ("RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)"
                        if task_type == "classification"
                        else "RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED)")
    elif "logistic" in bm:
        import_line  = "from sklearn.linear_model import LogisticRegression"
        clf_line     = "LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)"
    elif "ridge" in bm:
        import_line  = "from sklearn.linear_model import Ridge"
        clf_line     = "Ridge()"
    else:
        import_line  = "from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor"
        clf_line     = ("GradientBoostingClassifier(random_state=RANDOM_SEED)"
                        if task_type == "classification"
                        else "GradientBoostingRegressor(random_state=RANDOM_SEED)")

    cells.append(_code_cell([
        import_line,
        "",
        f"base_model = {clf_line}",
        "",
        "# Wrap in full pipeline with preprocessing",
        "model = Pipeline([",
        "    ('preprocessor', preprocessor),",
        "    ('classifier',   base_model),",
        "])",
        "",
        "model.fit(X_train, y_train)",
        "print('Training complete.')"
    ]))

    # ─── 7. Evaluation ──────────────────────────────────────────────────────
    cells.append(_md_cell("## 7. Evaluation"))
    if task_type == "classification":
        eval_code = [
            "y_pred = model.predict(X_test)",
            "print(classification_report(y_test, y_pred))",
            "",
            "# Confusion matrix",
            "cm = confusion_matrix(y_test, y_pred)",
            "fig, ax = plt.subplots(figsize=(6, 5))",
            "sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')",
            "ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')",
            f"ax.set_title('Confusion Matrix — {best_model}')",
            "plt.tight_layout(); plt.show()"
        ]
    else:
        eval_code = [
            "y_pred = model.predict(X_test)",
            "print(f'R²  : {r2_score(y_test, y_pred):.4f}')",
            "print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}')",
            "",
            "# Residuals plot",
            "residuals = y_test - y_pred",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
            "axes[0].scatter(y_pred, residuals, alpha=0.4)",
            "axes[0].axhline(0, color='red'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Residual')",
            "axes[0].set_title('Residuals vs Predicted')",
            "axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white')",
            "axes[1].set_title('Residual Distribution')",
            "plt.tight_layout(); plt.show()"
        ]

    # Feature importance (tree-based models)
    eval_code += [
        "",
        "# Feature importances (if available)",
        "try:",
        "    clf_step = model.named_steps['classifier']",
        "    if hasattr(clf_step, 'feature_importances_'):",
        "        feature_names = (numeric_features +",
        "            model.named_steps['preprocessor']",
        "            .named_transformers_['cat']",
        "            .named_steps['onehot']",
        "            .get_feature_names_out(categorical_features).tolist())",
        "        imp = pd.Series(clf_step.feature_importances_, index=feature_names).sort_values(ascending=False)[:20]",
        "        fig, ax = plt.subplots(figsize=(8, 5))",
        "        imp.plot(kind='barh', ax=ax, color='steelblue')",
        "        ax.invert_yaxis()",
        "        ax.set_title('Top 20 Feature Importances')",
        "        plt.tight_layout(); plt.show()",
        "except Exception as e:",
        "    print(f'Feature importance not available: {e}')"
    ]
    cells.append(_code_cell(eval_code))

    # ─── 8. Save Model ──────────────────────────────────────────────────────
    cells.append(_md_cell("## 8. Save & Load Model"))
    model_path = (config.get("model_export_path") or config.get("modelExportPath")
                  or "best_model.joblib")
    cells.append(_code_cell([
        f"MODEL_PATH = {model_path!r}",
        "joblib.dump(model, MODEL_PATH)",
        "print(f'Model saved to: {MODEL_PATH}')",
        "",
        "# To load later:",
        "# loaded_model = joblib.load(MODEL_PATH)",
        "# loaded_model.predict(new_data)"
    ]))

    # ─── 9. Prediction Template ─────────────────────────────────────────────
    cells.append(_md_cell("## 9. Prediction Template"))
    cells.append(_code_cell([
        "# Replace these sample values with your new data",
        "new_data = pd.DataFrame([{",
        f"    # col: value,  ... (add one key per feature column)",
        "}])",
        "",
        "prediction = model.predict(new_data)",
        "print(f'Prediction: {prediction[0]}')"
    ]))

    # ─── Write file ──────────────────────────────────────────────────────────
    nb = _notebook(cells)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
