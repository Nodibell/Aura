import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier, DummyRegressor
from utils.helpers import print_progress, _export_model_and_code
from utils.profiler import profile_dataset

def analyze_nlp(df, target_col, task_type_override,
                row_count, col_count, columns, full_preview, missing,
                numeric_cols, categorical_cols,
                file_path=None, model_export_path=None, code_export_path=None):
    try:
        print_progress(0.30, "Detecting primary text feature column...")
        
        # 1. Detect primary text column
        text_candidates = []
        for col in columns:
            if col != target_col:
                sample_series = df[col].dropna().astype(str)
                if not sample_series.empty:
                    avg_len = sample_series.str.len().mean()
                    text_candidates.append((col, avg_len))
        
        if text_candidates:
            text_candidates.sort(key=lambda x: x[1], reverse=True)
            text_col = text_candidates[0][0]
        else:
            text_col = [c for c in columns if c != target_col][0] if len(columns) > 1 else columns[0]
            
        print_progress(0.35, f"Computing text metrics for column '{text_col}'...")
        
        # 2. Compute text profiling statistics
        text_series = df[text_col].fillna("").astype(str)
        char_counts = text_series.str.len()
        word_counts = text_series.apply(lambda x: len(x.split()))
        
        def count_sentences(text):
            delims = ['.', '!', '?']
            count = sum(text.count(d) for d in delims)
            return max(1, count)
        sentence_counts = text_series.apply(count_sentences)
        
        text_stats = {
            "avg_chars": float(char_counts.mean()) if not np.isnan(char_counts.mean()) else 0.0,
            "max_chars": int(char_counts.max()) if not np.isnan(char_counts.max()) else 0,
            "avg_words": float(word_counts.mean()) if not np.isnan(word_counts.mean()) else 0.0,
            "max_words": int(word_counts.max()) if not np.isnan(word_counts.max()) else 0,
            "avg_sentences": float(sentence_counts.mean()) if not np.isnan(sentence_counts.mean()) else 0.0
        }
        
        # Target variable setup
        y_raw = df[target_col]
        is_numeric_target = pd.api.types.is_numeric_dtype(y_raw.dtype)
        
        is_classification = False
        if task_type_override == "classification":
            is_classification = True
        elif task_type_override == "regression":
            is_classification = False
        else:
            is_classification = not is_numeric_target
            
        if is_classification:
            y = y_raw.ffill().bfill().astype(str).to_numpy()
            unique_targets = len(np.unique(y))
        else:
            y = y_raw.interpolate(method='linear').ffill().bfill().to_numpy()
            
        print_progress(0.45, "Running TF-IDF vectorization (max_features=200)...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        X_processed = vectorizer.fit_transform(text_series).toarray()
        feature_names = vectorizer.get_feature_names_out()
        vocab_size = len(vectorizer.vocabulary_)
        
        print_progress(0.52, "Calculating word importances...")
        tfidf_sums = X_processed.sum(axis=0)
        word_importance = []
        for word, weight in zip(feature_names, tfidf_sums):
            word_importance.append({"word": word, "weight": float(weight)})
        word_importance.sort(key=lambda x: x["weight"], reverse=True)
        top_words = word_importance[:20]
        
        charts = []
        
        charts.append({
            "type": "bar",
            "title": f"Top Words by TF-IDF Importance in '{text_col}'",
            "x_label": "Word",
            "y_label": "Aggregate TF-IDF Weight",
            "data": [{"x_val": w["word"], "x_num": None, "y": w["weight"]} for w in top_words]
        })
        
        # Word Cloud chart config
        wordcloud_words = word_importance[:50]
        charts.append({
            "type": "wordcloud",
            "title": f"Word Cloud for Column '{text_col}'",
            "x_label": "Word",
            "y_label": "Aggregate TF-IDF Weight",
            "data": [{"x_val": w["word"], "x_num": None, "y": w["weight"]} for w in wordcloud_words]
        })
        
        counts, bin_edges = np.histogram(word_counts, bins=min(10, max(2, len(np.unique(word_counts)))))
        word_dist_data = []
        for i in range(len(counts)):
            bin_range = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
            word_dist_data.append({
                "x_val": bin_range,
                "x_num": None,
                "y": float(counts[i])
            })
        charts.append({
            "type": "bar",
            "title": "Word Count Distribution per Document",
            "x_label": "Words Range",
            "y_label": "Count of Documents",
            "data": word_dist_data
        })
        X_val, y_val = None, None
        val_metrics = None
        val_confusion_matrix_data = None

        if "__is_test" in df.columns:
            train_mask = (df["__is_test"] == 0).to_numpy()
            test_mask = (df["__is_test"] == 1).to_numpy()
            val_mask = (df["__is_test"] == 2).to_numpy()
            
            X_train_full, y_train_full = X_processed[train_mask], y[train_mask]
            
            if np.any(val_mask):
                X_val, y_val = X_processed[val_mask], y[val_mask]
                
            if np.any(test_mask):
                X_train, X_test = X_train_full, X_processed[test_mask]
                y_train, y_test = y_train_full, y[test_mask]
            else:
                use_stratify = y_train_full if is_classification else None
                if is_classification:
                    unique_classes, class_counts = np.unique(y_train_full, return_counts=True)
                    if min(class_counts) < 2:
                        use_stratify = None
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=use_stratify
                )
        else:
            use_stratify = y if is_classification else None
            if is_classification:
                unique_classes, class_counts = np.unique(y, return_counts=True)
                if min(class_counts) < 2:
                    use_stratify = None
                    
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=use_stratify
            )
            
        # Cap dataset size for model training to avoid slow execution/OOM
        if len(X_train) > 10000:
            np.random.seed(42)
            indices = np.random.choice(len(X_train), 10000, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
        if len(X_test) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(X_test), 5000, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
        if X_val is not None and len(X_val) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(X_val), 5000, replace=False)
            X_val = X_val[indices]
            y_val = y_val[indices]
        
        models_compared = []
        metrics = {}
        dummy_score = None
        confusion_matrix = None
        
        if is_classification:
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix as sklearn_cm
            
            nb = MultinomialNB()
            nb.fit(X_train, y_train)
            nb_preds = nb.predict(X_test)
            nb_f1 = f1_score(y_test, nb_preds, average='weighted', zero_division=0)
            
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_f1 = f1_score(y_test, lr_preds, average='weighted', zero_division=0)
            
            if lr_f1 >= nb_f1:
                best_model = "Logistic Regression"
                best_score = lr_f1
                best_preds = lr_preds
                best_clf = lr
            else:
                best_model = "Multinomial Naive Bayes"
                best_score = nb_f1
                best_preds = nb_preds
                best_clf = nb
                
            from sklearn.metrics import classification_report
            report = classification_report(y_test, best_preds, output_dict=True, zero_division=0)
            
            f1_breakdown = []
            for k, v in report.items():
                if k not in ["accuracy", "macro avg", "weighted avg"]:
                    f1_breakdown.append({
                        "x_val": str(k),
                        "x_num": None,
                        "y": float(v["f1-score"])
                    })
            
            charts.append({
                "type": "bar",
                "title": "Per-Class F1-Score Breakdown",
                "x_label": "Class Label",
                "y_label": "F1-Score",
                "data": f1_breakdown
            })
            
            unique_labels = sorted(list(np.unique(y)))
            cm_vals = sklearn_cm(y_test, best_preds, labels=unique_labels)
            confusion_matrix = {
                "labels": unique_labels,
                "values": cm_vals.tolist()
            }
            
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train, y_train)
            dummy_preds = dummy.predict(X_test)
            dummy_score = float(f1_score(y_test, dummy_preds, average='weighted', zero_division=0))
            
            models_compared = [
                {"name": "Multinomial Naive Bayes", "score": float(nb_f1), "metric": "Weighted F1"},
                {"name": "Logistic Regression", "score": float(lr_f1), "metric": "Weighted F1"}
            ]
            metrics = {
                "model": best_model,
                "score_type": "Weighted F1",
                "score": float(best_score),
                "additional_metrics": {
                    "Accuracy": float(accuracy_score(y_test, best_preds))
                }
            }
        else:
            from sklearn.linear_model import Ridge, LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error
            
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            ridge_preds = ridge.predict(X_test)
            ridge_r2 = r2_score(y_test, ridge_preds)
            
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_r2 = r2_score(y_test, lr_preds)
            
            if ridge_r2 >= lr_r2:
                best_model = "Ridge Regression"
                best_score = ridge_r2
                best_preds = ridge_preds
                best_reg = ridge
            else:
                best_model = "Linear Regression"
                best_score = lr_r2
                best_preds = lr_preds
                best_reg = lr
                
            test_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
            
            from sklearn.dummy import DummyRegressor
            dummy = DummyRegressor(strategy="mean")
            dummy.fit(X_train, y_train)
            dummy_preds = dummy.predict(X_test)
            dummy_score = float(r2_score(y_test, dummy_preds))
            
            models_compared = [
                {"name": "Linear Regression", "score": float(lr_r2), "metric": "R\u00b2 Score"},
                {"name": "Ridge Regression", "score": float(ridge_r2), "metric": "R\u00b2 Score"}
            ]
            metrics = {
                "model": best_model,
                "score_type": "R\u00b2 Score",
                "score": float(best_score),
                "additional_metrics": {
                    "RMSE": float(test_rmse)
                }
            }
            
            scatter_data = []
            for i in range(len(y_test)):
                scatter_data.append({
                    "x_val": None,
                    "x_num": float(y_test[i]),
                    "y": float(best_preds[i])
                })
            charts.append({
                "type": "scatter",
                "title": "Actual vs Predicted Values Scatter Plot",
                "x_label": "Actual Value",
                "y_label": "Predicted Value",
                "data": scatter_data
            })
            
        # Evaluate on Validation set if present
        if X_val is not None and len(X_val) > 0:
            try:
                if is_classification:
                    val_preds = best_clf.predict(X_val)
                    val_acc = accuracy_score(y_val, val_preds)
                    val_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
                    val_metrics = {
                        "model": best_model,
                        "score_type": "Weighted F1",
                        "score": float(val_f1),
                        "additional_metrics": {
                            "Accuracy": float(val_acc)
                        }
                    }
                    classes_list = sorted(list(np.unique(y_val)))
                    cm_val = sklearn_cm(y_val, val_preds, labels=classes_list)
                    val_confusion_matrix_data = {
                        "labels": [str(c) for c in classes_list],
                        "values": cm_val.tolist()
                    }
                else:
                    val_preds = best_reg.predict(X_val)
                    val_r2 = r2_score(y_val, val_preds)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                    val_metrics = {
                        "model": best_model,
                        "score_type": "R\u00b2 Score",
                        "score": float(val_r2),
                        "additional_metrics": {
                            "RMSE": float(val_rmse)
                        }
                    }
            except Exception as val_err:
                sys.stderr.write(f"Warning: Validation evaluation failed in NLP: {str(val_err)}\n")
            
        print_progress(0.90, "Profiling columns & generating data statistics...")
        profiling = profile_dataset(df)
            
        print_progress(0.95, "Compiling summary report & finalizing...")
        summary_sections = []
        overview = f"### 💬 Text / NLP Analysis Overview\n"
        overview += f"- **Rows:** {row_count:,} | **Columns:** {col_count:,}\n"
        overview += f"- **Primary Text Column:** `{text_col}`\n"
        overview += f"- **Vocabulary Size (TF-IDF):** {vocab_size:,} distinct tokens\n"
        overview += f"- **Missing Value Cells:** {sum(missing.values()):,} total."
        summary_sections.append(overview)
        
        stats_rep = f"### 📊 Text Profiling Metrics\n"
        stats_rep += f"- **Average Character Count:** `{text_stats['avg_chars']:.1f}` (Max: `{text_stats['max_chars']:,}`)\n"
        stats_rep += f"- **Average Word Count:** `{text_stats['avg_words']:.1f}` (Max: `{text_stats['max_words']:,}`)\n"
        stats_rep += f"- **Average Sentence Count (approx):** `{text_stats['avg_sentences']:.1f}`"
        summary_sections.append(stats_rep)
        
        target_info = f"### 🎯 Target Variable Analysis (`{target_col}`)\n"
        target_info += f"- **Task Type:** {('Classification' if is_classification else 'Regression')}\n"
        if is_classification:
            target_info += f"- **Unique Classes:** {unique_targets} categories"
        else:
            target_info += f"- **Range:** `{y_raw.min():.2f}` to `{y_raw.max():.2f}` (Mean: `{y_raw.mean():.2f}`)"
        summary_sections.append(target_info)
        
        model_perf = f"### 🤖 Machine Learning Model Performance\n"
        model_perf += f"- **Best Model:** `{best_model}`\n"
        model_perf += f"- **Primary Score ({metrics['score_type']}):** `{metrics['score']:.4f}`\n"
        if 'additional_metrics' in metrics and metrics['additional_metrics']:
            for am_name, am_val in metrics['additional_metrics'].items():
                model_perf += f"- **{am_name}:** `{am_val:.4f}`"
        summary_sections.append(model_perf)
        
        summary = "\n\n".join(summary_sections)

        # Phase 1: Model & Code Export
        if model_export_path or code_export_path:
            model_to_save = best_clf if is_classification else best_reg
            # NLP feature names are vocabulary words
            feats = list(vectorizer.get_feature_names_out()) if 'vectorizer' in locals() else []
            _export_model_and_code(
                model_to_save, model_export_path, code_export_path,
                file_path, "nlp", target_col, None,
                "classification" if is_classification else "regression",
                feats, best_model, numeric_cols, categorical_cols, [text_col] if 'text_col' in locals() else []
            )
        
        return {
            "summary": summary,
            "columns": columns,
            "row_count": int(row_count),
            "col_count": int(col_count),
            "task_type": "classification" if is_classification else "regression",
            "numeric_col_count": len(numeric_cols),
            "categorical_col_count": len(categorical_cols),
            "text_col_count": 1,
            "missing_values": {k: int(v) for k, v in missing.items()},
            "correlations": [],
            "charts": charts,
            "metrics": metrics,
            "val_metrics": val_metrics,
            "val_confusion_matrix": val_confusion_matrix_data,
            "models_compared": models_compared,
            "target_column": target_col,
            "full_preview": full_preview,
            "dummy_baseline_score": dummy_score,
            "cv_scores": [],
            "cv_mean": None,
            "cv_std": None,
            "confusion_matrix": confusion_matrix,
            "profiling": profiling,
            "error": None
        }
    except Exception as e:
        import traceback
        return {"error": f"An error occurred during Text/NLP execution: {str(e)}\n{traceback.format_exc()}"}
