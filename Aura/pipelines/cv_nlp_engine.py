import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------------------------------
# Computer Vision Helpers
# -------------------------------------------------------------------------

def pca_compress_images(X, n_comps=50):
    """
    Compresses image pixel arrays using PCA after flattening them.
    X: numpy array of shape (N, H, W, C) or (N, H, W)
    """
    N = X.shape[0]
    flat_dim = np.prod(X.shape[1:])
    X_flat = X.reshape(N, flat_dim)
    
    actual_comps = min(n_comps, N, flat_dim)
    pca = PCA(n_components=actual_comps, random_state=42)
    X_compressed = pca.fit_transform(X_flat)
    return X_compressed, pca

# -------------------------------------------------------------------------
# Natural Language Processing Helpers
# -------------------------------------------------------------------------

def extract_text_features(text_series, max_features=200):
    """
    Extracts TF-IDF features from text.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_processed = vectorizer.fit_transform(text_series).toarray()
    feature_names = vectorizer.get_feature_names_out()
    vocab_size = len(vectorizer.vocabulary_)
    return X_processed, feature_names, vocab_size, vectorizer

def calculate_lexicon_sentiment_and_diversity(text_series):
    """
    Calculates lexicon-based sentiment polarity, lexical diversity, and word lengths.
    """
    positive_lex = {
        "love", "loved", "loving", "likes", "like", "liked", "awesome", "amazing", "great", "excellent", "good",
        "wonderful", "fantastic", "beautiful", "perfect", "enjoy", "enjoyed", "happy", "pleasant", "glad",
        "satisfactory", "satisfied", "recommend", "best", "superb", "masterpiece", "outstanding", "brilliant",
        "witty", "smart", "touching", "heartwarming", "delight", "delightful", "incredible", "feast"
    }
    negative_lex = {
        "hate", "hated", "hating", "dislike", "disliked", "bad", "terrible", "awful", "horrible", "worst",
        "poor", "boring", "bored", "disappoint", "disappointed", "disappointing", "waste", "wasteful",
        "annoy", "annoyed", "annoying", "painful", "lifeless", "useless", "broken", "fail", "failed", "disaster",
        "embarrassment", "flat", "stupid", "mess", "shame", "mediocre"
    }
    
    polarities = []
    lexical_diversities = []
    avg_word_lengths = []
    
    for doc in text_series:
        words = [w.lower().strip(".,!?\"'()[]{}") for w in str(doc).split() if w]
        if not words:
            polarities.append(0.0)
            lexical_diversities.append(0.0)
            avg_word_lengths.append(0.0)
            continue
        
        # Polarity
        pos = sum(1 for w in words if w in positive_lex)
        neg = sum(1 for w in words if w in negative_lex)
        pol = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
        polarities.append(pol)
        
        # Lexical diversity
        lexical_diversities.append(len(set(words)) / len(words))
        
        # Average word length
        avg_word_lengths.append(float(np.mean([len(w) for w in words])))
        
    return polarities, lexical_diversities, avg_word_lengths
