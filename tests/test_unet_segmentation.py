import numpy as np
import pytest
from Aura.pipelines.unet_segmentation import train_unet, predict_unet

def test_unet_segmentation_pipeline():
    # Construct synthetic images batch: shape (4 images, 128 height, 128 width, 3 channels)
    np.random.seed(42)
    X_imgs = np.random.randint(0, 256, size=(4, 128, 128, 3), dtype=np.uint8)
    # Binary masks: shape (4 images, 128 height, 128 width), values 0 or 1
    y_masks = np.random.randint(0, 2, size=(4, 128, 128), dtype=np.uint8)

    # Train U-Net for 2 epochs
    model = train_unet(X_imgs, y_masks, epochs=2, batch_size=2)
    assert model is not None

    # Predict U-Net (with train/test split at index 2)
    iou, dice, accuracy, overlays = predict_unet(model, X_imgs, y_masks, split_idx=2)
    
    assert isinstance(iou, float)
    assert isinstance(dice, float)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    
    assert isinstance(overlays, list)
    assert len(overlays) > 0
    for overlay in overlays:
        assert "label" in overlay
        assert "base64" in overlay
        assert isinstance(overlay["label"], str)
        assert isinstance(overlay["base64"], str)
