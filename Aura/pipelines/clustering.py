import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from utils.helpers import print_progress, clean_nan
from utils.profiler import profile_dataset

def analyze_clustering(df, row_count, col_count, columns, full_preview, missing,
                       numeric_cols, categorical_cols, file_path=None):
    try:
        # We need to set OMP_NUM_THREADS = 1 in case it's not set
        os.environ["OMP_NUM_THREADS"] = "1"
        
        print_progress(0.35, "Preprocessing data for clustering...")
        
        # 1. Clean columns (exclude ID columns)
        # Identify identifier columns to exclude from clustering
        identifier_cols = []
        for col in columns:
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
                    
        active_numeric = [c for c in numeric_cols if c not in identifier_cols]
        active_categorical = [c for c in categorical_cols if c not in identifier_cols]
        
        if not active_numeric and not active_categorical:
            # Fallback to all columns if everything was excluded
            active_numeric = [c for c in numeric_cols]
            active_categorical = [c for c in categorical_cols]
            
        # 2. Build preprocessing pipeline
        transformers = []
        if active_numeric:
            transformers.append((
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]),
                active_numeric
            ))
            
        if active_categorical:
            transformers.append((
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                ]),
                active_categorical
            ))
            
        if not transformers:
            raise ValueError("No numeric or categorical columns available for clustering.")
            
        preprocessor = ColumnTransformer(transformers=transformers)
        X_processed = preprocessor.fit_transform(df)
        
        print_progress(0.50, "Finding optimal number of clusters for K-Means...")
        
        # 3. K-Means clustering with auto-K selection (silhouette score)
        best_k = 3
        best_score = -1
        sample_size = min(1000, X_processed.shape[0])
        
        if sample_size >= 10:
            rng = np.random.default_rng(42)
            indices = rng.choice(X_processed.shape[0], size=sample_size, replace=False)
            X_sample = X_processed[indices]
            
            # Test k from 2 to 5
            for k in [2, 3, 4, 5]:
                if k < X_processed.shape[0]:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5)
                    labels = km.fit_predict(X_processed)
                    score = silhouette_score(X_sample, labels[indices])
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
        print_progress(0.60, f"Fitting final K-Means with k={best_k}...")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_processed)
        
        # Compute final silhouette score for K-Means
        kmeans_silhouette = 0.0
        if sample_size >= 10:
            kmeans_silhouette = float(silhouette_score(X_processed[indices], kmeans_labels[indices]))
            
        print_progress(0.70, "Running DBSCAN clustering...")
        # 4. DBSCAN clustering
        # We standard scale X_processed to make default eps=0.5 reasonable
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_processed)
        
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise_points = int(np.sum(dbscan_labels == -1))
        
        dbscan_silhouette = 0.0
        if n_dbscan_clusters >= 2 and sample_size >= 10:
            try:
                dbscan_silhouette = float(silhouette_score(X_processed[indices], dbscan_labels[indices]))
            except Exception:
                pass
                
        # 5. Dimensionality Reduction (PCA & t-SNE)
        print_progress(0.80, "Generating dimensionality reduction coordinates...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_processed)
        
        has_tsne = X_processed.shape[0] <= 1500
        X_tsne = None
        if has_tsne:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, X_processed.shape[0] // 5)))
                X_tsne = tsne.fit_transform(X_processed)
            except Exception as tsne_err:
                sys.stderr.write(f"Warning: t-SNE projection failed: {tsne_err}\n")
                has_tsne = False
                
        # 6. Inject Cluster Columns to preview
        df_display = df.copy()
        df_display["K-Means Cluster"] = [f"Cluster {l}" for l in kmeans_labels]
        
        db_labels_str = []
        for l in dbscan_labels:
            if l == -1:
                db_labels_str.append("Noise")
            else:
                db_labels_str.append(f"Cluster {l}")
        df_display["DBSCAN Cluster"] = db_labels_str
        
        preview_df = df_display.head(500).fillna("").astype(str)
        updated_full_preview = {
            "columns": list(preview_df.columns),
            "rows": preview_df.values.tolist(),
            "total_rows": int(row_count)
        }
        
        # 7. Generate Charts
        print_progress(0.88, "Building cluster visualization charts...")
        charts = []
        
        # PCA colored by K-Means
        pca_km_data = []
        for i in range(len(X_pca)):
            pca_km_data.append({
                "x_val": None,
                "x_num": float(X_pca[i, 0]),
                "y": float(X_pca[i, 1]),
                "series": f"Cluster {kmeans_labels[i]}"
            })
        charts.append({
            "type": "scatter",
            "title": "PCA 2D Cluster Projection (K-Means)",
            "x_label": "Principal Component 1",
            "y_label": "Principal Component 2",
            "data": pca_km_data[:2000] # Cap points for performance
        })
        
        # PCA colored by DBSCAN
        pca_db_data = []
        for i in range(len(X_pca)):
            series_name = "Noise" if dbscan_labels[i] == -1 else f"Cluster {dbscan_labels[i]}"
            pca_db_data.append({
                "x_val": None,
                "x_num": float(X_pca[i, 0]),
                "y": float(X_pca[i, 1]),
                "series": series_name
            })
        charts.append({
            "type": "scatter",
            "title": "PCA 2D Cluster Projection (DBSCAN)",
            "x_label": "Principal Component 1",
            "y_label": "Principal Component 2",
            "data": pca_db_data[:2000]
        })
        
        # t-SNE Charts if computed
        if has_tsne and X_tsne is not None:
            tsne_km_data = []
            for i in range(len(X_tsne)):
                tsne_km_data.append({
                    "x_val": None,
                    "x_num": float(X_tsne[i, 0]),
                    "y": float(X_tsne[i, 1]),
                    "series": f"Cluster {kmeans_labels[i]}"
                })
            charts.append({
                "type": "scatter",
                "title": "t-SNE 2D Cluster Projection (K-Means)",
                "x_label": "t-SNE Component 1",
                "y_label": "t-SNE Component 2",
                "data": tsne_km_data[:2000]
            })
            
            tsne_db_data = []
            for i in range(len(X_tsne)):
                series_name = "Noise" if dbscan_labels[i] == -1 else f"Cluster {dbscan_labels[i]}"
                tsne_db_data.append({
                    "x_val": None,
                    "x_num": float(X_tsne[i, 0]),
                    "y": float(X_tsne[i, 1]),
                    "series": series_name
                })
            charts.append({
                "type": "scatter",
                "title": "t-SNE 2D Cluster Projection (DBSCAN)",
                "x_label": "t-SNE Component 1",
                "y_label": "t-SNE Component 2",
                "data": tsne_db_data[:2000]
            })
            
        # Target Distribution/Cluster Sizes Chart
        km_counts = pd.Series(kmeans_labels).value_counts().to_dict()
        charts.append({
            "type": "bar",
            "title": "K-Means Cluster Sizes",
            "x_label": "Cluster",
            "y_label": "Count",
            "data": [{"x_val": f"Cluster {k}", "x_num": None, "y": float(v)} for k, v in km_counts.items()]
        })
        
        # Profile original dataset columns
        print_progress(0.92, "Profiling columns & generating data statistics...")
        profiling = profile_dataset(df)
        
        # 8. Summary Report
        print_progress(0.95, "Compiling summary report & finalizing...")
        summary = f"### 🧩 Unsupervised Clustering Overview\n"
        summary += f"- **Rows:** {row_count:,} | **Columns:** {col_count:,}\n"
        summary += f"- **K-Means configuration:** Inferred optimal clusters `k = {best_k}`\n"
        summary += f"- **K-Means Silhouette Score:** `{kmeans_silhouette:.4f}`\n"
        summary += f"- **DBSCAN configuration:** Found `{n_dbscan_clusters}` clusters, `{n_noise_points}` noise points\n"
        if n_dbscan_clusters >= 2:
            summary += f"- **DBSCAN Silhouette Score:** `{dbscan_silhouette:.4f}`\n"
        else:
            summary += f"- **DBSCAN Silhouette Score:** N/A (fewer than 2 clusters found)\n"
            
        # Compile model metrics structure
        metrics = {
            "model": "K-Means + DBSCAN Clustering",
            "score_type": "Silhouette Score",
            "score": kmeans_silhouette,
            "additional_metrics": {
                "Optimal K": float(best_k),
                "DBSCAN Clusters": float(n_dbscan_clusters),
                "DBSCAN Noise": float(n_noise_points)
            }
        }
        
        models_compared = [
            {"name": f"K-Means (k={best_k})", "score": kmeans_silhouette, "metric": "Silhouette Score"},
            {"name": "DBSCAN", "score": dbscan_silhouette, "metric": "Silhouette Score"}
        ]
        
        return {
            "summary": summary,
            "columns": list(df_display.columns),
            "row_count": int(row_count),
            "col_count": int(df_display.shape[1]),
            "task_type": "clustering",
            "numeric_col_count": len(active_numeric),
            "categorical_col_count": len(active_categorical),
            "text_col_count": 0,
            "missing_values": {k: int(v) for k, v in missing.items()},
            "correlations": [],
            "charts": charts,
            "metrics": metrics,
            "val_metrics": None,
            "val_confusion_matrix": None,
            "models_compared": models_compared,
            "target_column": "",
            "full_preview": updated_full_preview,
            "dummy_baseline_score": 0.0,
            "cv_scores": [],
            "cv_mean": 0.0,
            "cv_std": 0.0,
            "confusion_matrix": None,
            "profiling": profiling,
            "data_leakage_warnings": [],
            "cleaning_recommendations": [],
            "error": None
        }
        
    except Exception as e:
        import traceback
        return {"error": f"An error occurred during Unsupervised Clustering execution: {str(e)}\n{traceback.format_exc()}"}
