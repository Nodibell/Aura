import pytest
import numpy as np
import pandas as pd
from Aura.pipelines.cv_nlp_engine import (
    pca_compress_images,
    extract_text_features,
    calculate_lexicon_sentiment_and_diversity
)

def test_pca_compress_images():
    # Shape: (10 images, 32 height, 32 width, 3 channels)
    X = np.random.rand(10, 32, 32, 3)
    X_comp, pca = pca_compress_images(X, n_comps=5)
    assert X_comp.shape == (10, 5)

def test_extract_text_features():
    texts = pd.Series([
        "Aura is a beautiful and amazing application for machine learning.",
        "The model engine works flawlessly and with high performance.",
        "Simple sentiment analysis based on lexicons is quick and helpful."
    ])
    X_processed, feature_names, vocab_size, vectorizer = extract_text_features(texts, max_features=10)
    assert X_processed.shape == (3, 10)
    assert len(feature_names) == 10
    assert vocab_size >= 10

def test_calculate_lexicon_sentiment_and_diversity():
    texts = pd.Series([
        "Aura is an amazing and awesome masterpiece of software engineering.",
        "Hate terrible bad awful experience with this broken tool.",
        "Neutral statement without any emotional polarity words here."
    ])
    polarities, diversities, lengths = calculate_lexicon_sentiment_and_diversity(texts)
    assert len(polarities) == 3
    assert polarities[0] > 0.0 # positive
    assert polarities[1] < 0.0 # negative
    assert polarities[2] == 0.0 # neutral
    assert len(diversities) == 3
    assert len(lengths) == 3
