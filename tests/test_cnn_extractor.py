import numpy as np
import pytest
from Aura.pipelines.cnn_extractor import extract_cnn_features

def test_extract_cnn_features_resnet():
    # Construct synthetic images batch: shape (4 images, 64 height, 64 width, 3 channels)
    np.random.seed(42)
    X_images = np.random.randint(0, 256, size=(4, 64, 64, 3), dtype=np.uint8)

    embeddings, extractor_name = extract_cnn_features(X_images)
    
    assert embeddings is not None
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 4
    
    if "ResNet" in extractor_name:
        assert embeddings.shape[1] == 512
    else:
        assert extractor_name == "PCA"
        assert embeddings.shape[1] <= 100

def test_extract_cnn_features_pca_fallback():
    # Construct synthetic images batch: shape (2 images, 16 height, 16 width, 3 channels)
    X_images = np.random.randint(0, 256, size=(2, 16, 16, 3), dtype=np.uint8)

    import Aura.pipelines.cnn_extractor as cnn
    original_device = cnn.DEVICE
    cnn.DEVICE = None
    try:
        embeddings, extractor_name = extract_cnn_features(X_images)
        assert extractor_name == "PCA"
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 2
    finally:
        cnn.DEVICE = original_device
