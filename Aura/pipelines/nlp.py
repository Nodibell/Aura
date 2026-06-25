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
        from pipelines.cv_nlp_engine import extract_text_features, calculate_lexicon_sentiment_and_diversity
        X_processed, feature_names, vocab_size, vectorizer = extract_text_features(text_series, max_features=200)
        
        print_progress(0.52, "Calculating word importances...")
        tfidf_sums = X_processed.sum(axis=0)
        word_importance = []
        for word, weight in zip(feature_names, tfidf_sums):
            word_importance.append({"word": word, "weight": float(weight)})
        word_importance.sort(key=lambda x: x["weight"], reverse=True)
        top_words = word_importance[:20]
        
        # 2. Advanced NLP Analysis: N-grams, Sentiment, and Lexical Diversity
        polarities, lexical_diversities, avg_word_lengths = calculate_lexicon_sentiment_and_diversity(text_series)
        
        mean_polarity = float(np.mean(polarities)) if polarities else 0.0
        mean_lex_div = float(np.mean(lexical_diversities)) if lexical_diversities else 0.0
        mean_word_len = float(np.mean(avg_word_lengths)) if avg_word_lengths else 0.0

        charts = []
        
        # 1. Unigrams Chart
        charts.append({
            "type": "bar",
            "title": f"Top Words by TF-IDF Importance in '{text_col}'",
            "x_label": "Word",
            "y_label": "Aggregate TF-IDF Weight",
            "data": [{"x_val": w["word"], "x_num": None, "y": w["weight"]} for w in top_words]
        })
        
        # 2. Word Cloud
        wordcloud_words = word_importance[:50]
        charts.append({
            "type": "wordcloud",
            "title": f"Word Cloud for Column '{text_col}'",
            "x_label": "Word",
            "y_label": "Aggregate TF-IDF Weight",
            "data": [{"x_val": w["word"], "x_num": None, "y": w["weight"]} for w in wordcloud_words]
        })

        # 3. Bigrams Chart
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer_2 = CountVectorizer(ngram_range=(2, 2), max_features=10, stop_words='english')
            X_2 = vectorizer_2.fit_transform(text_series)
            sums_2 = X_2.sum(axis=0).A1
            names_2 = vectorizer_2.get_feature_names_out()
            bigrams_data = [{"x_val": name, "x_num": None, "y": float(s)} for name, s in zip(names_2, sums_2)]
            bigrams_data.sort(key=lambda x: x["y"], reverse=True)
            charts.append({
                "type": "bar",
                "title": f"Top Bigrams (2-word phrases) in '{text_col}'",
                "x_label": "Bigram",
                "y_label": "Frequency",
                "data": bigrams_data
            })
        except Exception as e_bg:
            sys.stderr.write(f"Warning: Bigrams extraction failed: {e_bg}\n")

        # 4. Trigrams Chart
        try:
            vectorizer_3 = CountVectorizer(ngram_range=(3, 3), max_features=10, stop_words='english')
            X_3 = vectorizer_3.fit_transform(text_series)
            sums_3 = X_3.sum(axis=0).A1
            names_3 = vectorizer_3.get_feature_names_out()
            trigrams_data = [{"x_val": name, "x_num": None, "y": float(s)} for name, s in zip(names_3, sums_3)]
            trigrams_data.sort(key=lambda x: x["y"], reverse=True)
            charts.append({
                "type": "bar",
                "title": f"Top Trigrams (3-word phrases) in '{text_col}'",
                "x_label": "Trigram",
                "y_label": "Frequency",
                "data": trigrams_data
            })
        except Exception as e_tg:
            sys.stderr.write(f"Warning: Trigrams extraction failed: {e_tg}\n")
        
        # 5. Document Word Count Distribution
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

        # 6. Sentiment Distribution Chart
        pos_docs = sum(1 for p in polarities if p > 0.1)
        neu_docs = sum(1 for p in polarities if -0.1 <= p <= 0.1)
        neg_docs = sum(1 for p in polarities if p < -0.1)
        charts.append({
            "type": "bar",
            "title": "Estimated Document Sentiment Tone Distribution",
            "x_label": "Sentiment Tone",
            "y_label": "Count of Documents",
            "data": [
                {"x_val": "Positive", "x_num": None, "y": float(pos_docs)},
                {"x_val": "Neutral", "x_num": None, "y": float(neu_docs)},
                {"x_val": "Negative", "x_num": None, "y": float(neg_docs)}
            ]
        })

        # 7. Sentiment / Word Count vs Target Class Relationship
        if is_classification:
            avg_pol_by_class = {}
            avg_wc_by_class = {}
            for cls in np.unique(y):
                mask = (y == cls)
                cls_pols = [polarities[i] for i, m in enumerate(mask) if m]
                cls_wcs = [word_counts.iloc[i] for i, m in enumerate(mask) if m]
                avg_pol_by_class[cls] = float(np.mean(cls_pols)) if cls_pols else 0.0
                avg_wc_by_class[cls] = float(np.mean(cls_wcs)) if cls_wcs else 0.0
                
            charts.append({
                "type": "bar",
                "title": f"Average Sentiment Polarity by Class in '{target_col}'",
                "x_label": "Class",
                "y_label": "Mean Sentiment Polarity",
                "data": [{"x_val": str(cls), "x_num": None, "y": val} for cls, val in avg_pol_by_class.items()]
            })
            charts.append({
                "type": "bar",
                "title": f"Average Word Count by Class in '{target_col}'",
                "x_label": "Class",
                "y_label": "Mean Word Count",
                "data": [{"x_val": str(cls), "x_num": None, "y": val} for cls, val in avg_wc_by_class.items()]
            })
        else:
            scatter_pol_target = []
            for i in range(len(y)):
                scatter_pol_target.append({
                    "x_val": None,
                    "x_num": float(y[i]),
                    "y": float(polarities[i])
                })
            charts.append({
                "type": "scatter",
                "title": f"Document Sentiment Polarity vs Target '{target_col}'",
                "x_label": f"Target Value ({target_col})",
                "y_label": "Sentiment Polarity",
                "data": scatter_pol_target[:1000]
            })

        # Note: Chart 8 (Model Coefficients) moved to post-training phase below

        # 9. Document Embedding 2D Projection using TruncatedSVD (max 1000 points)
        try:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=2, random_state=42)
            X_svd = svd.fit_transform(X_processed)
            
            n_samples = len(df)
            sample_size = min(1000, n_samples)
            if n_samples > 1000:
                np.random.seed(42)
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            else:
                sample_indices = range(n_samples)
                
            projection_data = []
            for idx in sample_indices:
                x_val = float(X_svd[idx, 0])
                y_val = float(X_svd[idx, 1])
                class_label = str(y[idx]) if is_classification else "Document"
                projection_data.append({
                    "x_val": None,
                    "x_num": x_val,
                    "y": y_val,
                    "series": class_label
                })
                
            charts.append({
                "type": "scatter",
                "title": "Document Semantic Space (2D SVD Projection)",
                "x_label": "SVD Component 1",
                "y_label": "SVD Component 2",
                "data": projection_data
            })
        except Exception as svd_err:
            sys.stderr.write(f"Warning: Failed to generate document SVD projection: {str(svd_err)}\n")

        # 10. Lexical Diversity Boxplots by Class
        if is_classification:
            try:
                # We calculate stats of word counts grouped by target class
                for cls in np.unique(y):
                    mask = (y == cls)
                    cls_wcs = word_counts[mask].values
                    if len(cls_wcs) >= 5:
                        sorted_wcs = np.sort(cls_wcs)
                        q1 = float(np.percentile(sorted_wcs, 25))
                        median = float(np.percentile(sorted_wcs, 50))
                        q3 = float(np.percentile(sorted_wcs, 75))
                        iqr = q3 - q1
                        
                        if iqr <= 0.0:
                            lower_whisker = float(sorted_wcs.min())
                            upper_whisker = float(sorted_wcs.max())
                            outliers_list = []
                        else:
                            lower_fence = q1 - 1.5 * iqr
                            upper_fence = q3 + 1.5 * iqr
                            non_outliers = sorted_wcs[(sorted_wcs >= lower_fence) & (sorted_wcs <= upper_fence)]
                            lower_whisker = float(non_outliers.min()) if len(non_outliers) > 0 else q1
                            upper_whisker = float(non_outliers.max()) if len(non_outliers) > 0 else q3
                            outliers = sorted_wcs[(sorted_wcs < lower_whisker) | (sorted_wcs > upper_whisker)]
                            outliers_list = [float(x) for x in outliers[:100]]
                            
                        charts.append({
                            "type": "boxplot",
                            "title": f"Lexical Complexity Boxplot: Class {cls}",
                            "x_label": "",
                            "y_label": "Word Count",
                            "data": [],
                            "box_stats": {
                                "min": lower_whisker,
                                "q1": q1,
                                "median": median,
                                "q3": q3,
                                "max": upper_whisker,
                                "outliers": outliers_list
                            }
                        })
            except Exception as box_err:
                sys.stderr.write(f"Warning: Failed to generate lexical diversity boxplots: {str(box_err)}\n")

        # 11. Class-Specific Top TF-IDF Terms Grouped Bar Chart
        if is_classification:
            try:
                class_words_data = []
                for cls in np.unique(y):
                    class_mask = (y == cls)
                    class_tfidf_sum = X_processed[class_mask].sum(axis=0)
                    
                    # Sort words for this class
                    top_word_indices = np.argsort(class_tfidf_sum)[-5:]
                    for idx in top_word_indices:
                        class_words_data.append({
                            "x_val": str(feature_names[idx]),
                            "x_num": None,
                            "y": float(class_tfidf_sum[idx]),
                            "series": f"Class {cls}"
                        })
                charts.append({
                    "type": "bar",
                    "title": "Class-Specific Top TF-IDF Words",
                    "x_label": "Word",
                    "y_label": "Cumulative TF-IDF Score",
                    "data": class_words_data
                })
            except Exception as tfidf_cls_err:
                sys.stderr.write(f"Warning: Failed to generate class-specific top words: {str(tfidf_cls_err)}\n")

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
            from sklearn.naive_bayes import MultinomialNB, ComplementNB
            from sklearn.linear_model import LogisticRegression, SGDClassifier
            from sklearn.svm import LinearSVC
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix as sklearn_cm
            from sklearn.preprocessing import LabelEncoder
            import optuna
            from xgboost import XGBClassifier
            
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            nb = MultinomialNB()
            nb.fit(X_train, y_train)
            nb_preds = nb.predict(X_test)
            nb_f1 = f1_score(y_test, nb_preds, average='weighted', zero_division=0)
            
            cnb = ComplementNB()
            cnb.fit(X_train, y_train)
            cnb_preds = cnb.predict(X_test)
            cnb_f1 = f1_score(y_test, cnb_preds, average='weighted', zero_division=0)
            
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_f1 = f1_score(y_test, lr_preds, average='weighted', zero_division=0)
            
            svc = LinearSVC(random_state=42, max_iter=2000)
            svc.fit(X_train, y_train)
            svc_preds = svc.predict(X_test)
            svc_f1 = f1_score(y_test, svc_preds, average='weighted', zero_division=0)
            
            sgd = SGDClassifier(random_state=42, max_iter=2000)
            sgd.fit(X_train, y_train)
            sgd_preds = sgd.predict(X_test)
            sgd_f1 = f1_score(y_test, sgd_preds, average='weighted', zero_division=0)
            
            print_progress(0.66, "Tuning XGBoost Text Classifier with Optuna...")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            tuning_split_idx = int(len(X_train) * 0.8)
            if tuning_split_idx >= 2:
                tuning_X_tr, tuning_X_val = X_train[:tuning_split_idx], X_train[tuning_split_idx:]
                tuning_y_tr_encoded = y_train_encoded[:tuning_split_idx]
                tuning_y_val_encoded = y_train_encoded[tuning_split_idx:]
                
                def objective(trial):
                    n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
                    max_depth = trial.suggest_int("xgb_max_depth", 3, 8)
                    learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
                    clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, n_jobs=-1, eval_metric="mlogloss")
                    clf.fit(tuning_X_tr, tuning_y_tr_encoded)
                    preds = clf.predict(tuning_X_val)
                    return f1_score(tuning_y_val_encoded, preds, average='weighted', zero_division=0)
                
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=30, timeout=8.0)
                xgb_best_n = study.best_params.get("xgb_n_estimators", 50)
                xgb_best_d = study.best_params.get("xgb_max_depth", 5)
                xgb_best_lr = study.best_params.get("xgb_learning_rate", 0.1)
            else:
                xgb_best_n, xgb_best_d, xgb_best_lr = 50, 5, 0.1
                
            xgb = XGBClassifier(n_estimators=xgb_best_n, max_depth=xgb_best_d, learning_rate=xgb_best_lr, random_state=42, n_jobs=-1, eval_metric="mlogloss")
            xgb.fit(X_train, y_train_encoded)
            xgb_preds_encoded = xgb.predict(X_test)
            xgb_preds = le.inverse_transform(xgb_preds_encoded)
            xgb_f1 = f1_score(y_test, xgb_preds, average='weighted', zero_division=0)
            
            best_model = "Logistic Regression"
            best_score = lr_f1
            best_preds = lr_preds
            best_clf = lr
            
            if nb_f1 >= best_score:
                best_model = "Multinomial Naive Bayes"
                best_score = nb_f1
                best_preds = nb_preds
                best_clf = nb
                
            if cnb_f1 >= best_score:
                best_model = "Complement Naive Bayes"
                best_score = cnb_f1
                best_preds = cnb_preds
                best_clf = cnb
                
            if svc_f1 >= best_score:
                best_model = "Linear SVC"
                best_score = svc_f1
                best_preds = svc_preds
                best_clf = svc
                
            if sgd_f1 >= best_score:
                best_model = "SGD Classifier"
                best_score = sgd_f1
                best_preds = sgd_preds
                best_clf = sgd
                
            if xgb_f1 >= best_score:
                best_model = "Tuned XGBoost Classifier"
                best_score = xgb_f1
                best_preds = xgb_preds
                best_clf = xgb
                
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
                {"name": "Complement Naive Bayes", "score": float(cnb_f1), "metric": "Weighted F1"},
                {"name": "Logistic Regression", "score": float(lr_f1), "metric": "Weighted F1"},
                {"name": "Linear SVC", "score": float(svc_f1), "metric": "Weighted F1"},
                {"name": "SGD Classifier", "score": float(sgd_f1), "metric": "Weighted F1"},
                {"name": f"Tuned XGBoost Classifier (n={xgb_best_n}, d={xgb_best_d})", "score": float(xgb_f1), "metric": "Weighted F1"}
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
            
        # 8. Most Informative Features (Model Coefficients) Diverging Bar Chart (Post-Training)
        if is_classification:
            try:
                # We extract coefficients if best_model is Logistic Regression or Linear SVC
                if best_model in ["Logistic Regression", "Linear SVC"]:
                    coefs = best_clf.coef_
                    # Take first class for multiclass, or the single vector for binary
                    if len(coefs.shape) > 1 and coefs.shape[0] > 0:
                        flat_coefs = coefs[0]
                    else:
                        flat_coefs = coefs
                    
                    # Sort indices by absolute weight
                    sorted_indices = np.argsort(np.abs(flat_coefs))
                    # Take top 15 highest weight features
                    top_indices = sorted_indices[-15:] if len(sorted_indices) >= 15 else sorted_indices
                    
                    coef_chart_data = []
                    for idx in top_indices:
                        coef_chart_data.append({
                            "x_val": str(feature_names[idx]),
                            "x_num": None,
                            "y": float(flat_coefs[idx])
                        })
                    # Sort top indices by value so the bar chart looks clean
                    coef_chart_data.sort(key=lambda x: x["y"])
                    
                    charts.append({
                        "type": "bar",
                        "title": "Model Coefficients: Most Informative Words",
                        "x_label": "Word",
                        "y_label": "Coefficient Impact",
                        "data": coef_chart_data
                    })
            except Exception as coef_err:
                sys.stderr.write(f"Warning: Failed to compile model coefficients chart: {str(coef_err)}\n")

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
        stats_rep += f"- **Average Sentence Count (approx):** `{text_stats['avg_sentences']:.1f}`\n"
        stats_rep += f"- **Average Word Length:** `{mean_word_len:.2f}` characters\n"
        stats_rep += f"- **Lexical Diversity (Type-Token Ratio):** `{mean_lex_div:.2%}`\n"
        stats_rep += f"- **Mean Sentiment Polarity:** `{mean_polarity:+.2f}` (range -1.0 to +1.0)"
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
            raw_clf = best_clf if is_classification else best_reg
            # Build a full inference pipeline: TF-IDF vectorizer → classifier.
            # This ensures run_predict can accept raw text without a separate
            # vectorization step and the full chain is serialized together.
            from sklearn.pipeline import Pipeline as SKPipeline
            model_to_save = SKPipeline([
                ('tfidf', vectorizer),
                ('clf', raw_clf)
            ])
            export_le = None
            if is_classification and best_model == "Tuned XGBoost Classifier":
                export_le = le
            _export_model_and_code(
                model_to_save, model_export_path, code_export_path,
                file_path, "nlp", target_col, None,
                "classification" if is_classification else "regression",
                None, best_model, numeric_cols, categorical_cols,
                [text_col] if 'text_col' in locals() else [],
                cleaner=None, preprocessor=None, label_encoder=export_le
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
