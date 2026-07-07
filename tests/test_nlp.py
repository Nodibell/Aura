try:
    import torch  # noqa: F401
except ImportError:
    pass

import pytest
import numpy as np
import pandas as pd
from Aura.pipelines.nlp import analyze_nlp

def test_nlp_pipeline():
    # Make a synthetic NLP classification dataset
    np.random.seed(42)
    n_samples = 60
    
    # We want text containing positive or negative keywords
    positive_texts = [
        "This is an amazing and excellent movie! I loved it, it was perfect.",
        "A beautiful masterpiece, highly recommend. Best film of the year.",
        "Very good and pleasant experience, we enjoyed it a lot.",
        "Great acting, wonderful story, outstanding and brilliant.",
        "A heartwarming delight. Perfect and fantastic!"
    ]
    negative_texts = [
        "This was a terrible and awful experience. Hate it.",
        "A boring and disappointing waste of time. Worst movie ever.",
        "Very bad acting, horrible story, useless and lifeless.",
        "An embarrassing mess. Annoying, stupid and a disaster.",
        "I disliked this. Painful to watch, worst show."
    ]
    
    texts = []
    labels = []
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(positive_texts))
        labels.append("positive")
        texts.append(np.random.choice(negative_texts))
        labels.append("negative")
        
    df = pd.DataFrame({
        "review": texts,
        "sentiment": labels
    })
    
    columns = list(df.columns)
    row_count = len(df)
    col_count = len(df.columns)
    missing = df.isna().sum().to_dict()
    
    # Run the NLP analysis
    result = analyze_nlp(
        df=df,
        target_col="sentiment",
        task_type_override="classification",
        row_count=row_count,
        col_count=col_count,
        columns=columns,
        full_preview=None,
        missing=missing,
        numeric_cols=[],
        categorical_cols=[]
    )
    
    # Check output
    assert result is not None
    assert result["error"] is None
    assert result["metrics"]["score_type"] == "Accuracy"
    assert len(result["charts"]) > 0
    
    # Let's inspect some of the specific charts that were added in Phase 10
    titles = [c["title"] for c in result["charts"]]
    
    # Verify presence of SVD projection, lexical complexity, etc.
    assert any("Document Semantic Space" in t for t in titles)
    assert any("Lexical Complexity Boxplot" in t for t in titles)
    assert any("Class-Specific Top TF-IDF" in t for t in titles)

    # Test model export
    import tempfile
    import os
    import joblib
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "nlp_model.joblib")
        result_export = analyze_nlp(
            df=df,
            target_col="sentiment",
            task_type_override="classification",
            row_count=row_count,
            col_count=col_count,
            columns=columns,
            full_preview=None,
            missing=missing,
            numeric_cols=[],
            categorical_cols=[],
            model_export_path=model_path
        )
        assert result_export is not None
        assert result_export["error"] is None
        assert os.path.exists(model_path)
        
        # Test loading and prediction
        loaded = joblib.load(model_path)
        assert "model" in loaded
        assert "feature_names" in loaded
        
        # Predict on raw text
        test_sentences = ["This movie is outstanding and beautiful", "Terrible, boring waste of time"]
        preds = loaded["model"].predict(test_sentences)
        if "label_encoder" in loaded and loaded["label_encoder"] is not None:
            preds = loaded["label_encoder"].inverse_transform(preds)
        assert len(preds) == 2
        assert preds[0] in ["positive", "negative"]
