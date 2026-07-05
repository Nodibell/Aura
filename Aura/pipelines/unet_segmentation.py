"""
pipelines/unet_segmentation.py
──────────────────────────────
Compact U-Net for binary semantic segmentation trained entirely from scratch
on the provided image–mask pairs. Replaces the Random Forest pixel classifier
in pipelines/image.py's analyze_image_segmentation().

Architecture
------------
    Encoder  : 3→32→64→128  (MaxPool ×3)
    Bottleneck: 128→256
    Decoder  : 256→128→64→32  (bilinear upsample ×3)
    Head     : Conv(32→1) + sigmoid

Loss
----
    0.5 · BCEWithLogitsLoss  +  0.5 · DiceLoss
    All scalar weights created on DEVICE to avoid the CPU↔MPS scalar bug.

Public API
----------
    train_unet(X_imgs, y_masks, epochs=15, batch_size=4, progress_cb=None)
        -> trained UNet nn.Module

    predict_unet(model, X_imgs, y_masks, split_idx)
        -> iou: float, dice: float, accuracy: float, overlay_images: list[dict]
"""

import os
import sys
import io
import base64

import numpy as np

os.environ.setdefault("MTL_DEBUG_LAYER", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    from utils.helpers import print_progress as _print_progress
except ImportError:
    def _print_progress(frac, msg):
        sys.stderr.write(f"[{int(frac * 100)}%] {msg}\n")

# Re-use the MPS-safe device from cnn_extractor when possible
try:
    from pipelines.cnn_extractor import DEVICE as _EXT_DEVICE
    DEVICE = _EXT_DEVICE
except ImportError:
    try:
        import torch
        from pipelines.cnn_extractor import _select_device
        DEVICE = _select_device()
    except ImportError:
        DEVICE = None

TARGET_SIZE = (128, 128)   # H × W used during segmentation


# ── Model Definition ───────────────────────────────────────────────────────

def _double_conv(in_ch: int, out_ch: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _UNet(object):
    """Pure Python wrapper; actual nn.Module built lazily."""
    pass


def _build_unet():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = _double_conv(3, 32)
            self.enc2 = _double_conv(32, 64)
            self.enc3 = _double_conv(64, 128)
            self.pool = nn.MaxPool2d(2)
            # Bottleneck
            self.bottleneck = _double_conv(128, 256)
            # Decoder
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec3 = _double_conv(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = _double_conv(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = _double_conv(64, 32)
            # Head
            self.head = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            # Encode
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            # Bottleneck
            b = self.bottleneck(self.pool(e3))
            # Decode with skip connections
            d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)   # (B, 1, H, W) — raw logits

    return UNet().to(DEVICE)


# ── Loss ───────────────────────────────────────────────────────────────────

def _dice_loss(logits, targets, eps=1e-6):
    """
    Differentiable Dice loss from logits.
    All intermediate tensors stay on DEVICE — no CPU↔MPS scalar mixing.
    """
    import torch
    probs = torch.sigmoid(logits)
    B = probs.shape[0]
    probs_flat   = probs.view(B, -1)
    targets_flat = targets.view(B, -1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    union        = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    eps_t        = torch.tensor(eps, device=DEVICE)
    dice         = (torch.tensor(2.0, device=DEVICE) * intersection + eps_t) / (union + eps_t)
    return torch.tensor(1.0, device=DEVICE) - dice.mean()


def _combined_loss(logits, targets):
    import torch.nn as nn
    bce_fn  = nn.BCEWithLogitsLoss()
    w_bce   = 0.5
    w_dice  = 0.5
    bce_val  = bce_fn(logits, targets)
    dice_val = _dice_loss(logits, targets)
    return w_bce * bce_val + w_dice * dice_val


# ── Training ───────────────────────────────────────────────────────────────

def train_unet(
    X_imgs: np.ndarray,
    y_masks: np.ndarray,
    epochs: int = 15,
    batch_size: int = 4,
    progress_cb=None,
    progress_start: float = 0.45,
    progress_end: float = 0.75,
):
    """
    Train a U-Net on (N, H, W, 3) images and (N, H, W) binary masks.
    Returns the trained nn.Module.
    """
    import torch
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    pb = progress_cb or _print_progress

    if DEVICE is None:
        raise RuntimeError("torch not available — cannot train U-Net.")

    pb(progress_start, f"Building U-Net (device={DEVICE})...")
    model = _build_unet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    N = len(X_imgs)
    # Prepare tensors
    X_norm = X_imgs.astype(np.float32) / 255.0
    X_t = torch.from_numpy(X_norm).permute(0, 3, 1, 2).to(DEVICE)   # (N, 3, H, W)
    y_t = torch.from_numpy(y_masks.astype(np.float32)).unsqueeze(1).to(DEVICE)   # (N, 1, H, W)

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=min(batch_size, N), shuffle=True, drop_last=False)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss   = _combined_loss(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        frac = progress_start + ((epoch + 1) / epochs) * (progress_end - progress_start)
        pb(frac, f"U-Net training epoch {epoch + 1}/{epochs} — loss {epoch_loss / max(1, len(loader)):.4f}")

    model.eval()
    return model


# ── Inference & Metrics ────────────────────────────────────────────────────

def _to_base64_png(img_arr: np.ndarray) -> str:
    from PIL import Image as _Image
    if img_arr.max() <= 1.01:
        img_arr = img_arr * 255.0
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = _Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_unet(
    model,
    X_imgs: np.ndarray,
    y_masks: np.ndarray,
    split_idx: int,
    progress_cb=None,
    progress_start: float = 0.75,
    progress_end: float = 0.90,
):
    """
    Run inference on test split, compute metrics and overlay images.

    Returns
    -------
    iou : float
    dice : float
    accuracy : float
    overlay_images : list[dict]   — up to 4 base64 comparison strips
    """
    import torch

    pb = progress_cb or _print_progress
    N = len(X_imgs)
    H, W = X_imgs.shape[1], X_imgs.shape[2]

    test_indices = list(range(split_idx, N))

    # ── Metrics on test set ───────────────────────────────────────────────
    pb(progress_start, "Evaluating U-Net on test split...")
    all_preds  = []
    all_labels = []
    model.eval()

    batch_size = 4
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_x = X_imgs[i : i + batch_size].astype(np.float32) / 255.0
            t = torch.from_numpy(batch_x).permute(0, 3, 1, 2).to(DEVICE)
            logits = model(t)                                   # (B, 1, H, W)
            preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
            all_preds.append(preds[:, 0])                       # (B, H, W)
            all_labels.append(y_masks[i : i + batch_size])

    all_preds  = np.concatenate(all_preds, axis=0)              # (N, H, W)
    all_labels = np.concatenate(all_labels, axis=0)

    # Restrict metrics to test split
    test_preds  = all_preds[split_idx:]
    test_labels = all_labels[split_idx:]

    if len(test_preds) == 0:
        test_preds  = all_preds
        test_labels = all_labels

    tp = float(np.logical_and(test_preds == 1, test_labels == 1).sum())
    fp = float(np.logical_and(test_preds == 1, test_labels == 0).sum())
    fn = float(np.logical_and(test_preds == 0, test_labels == 1).sum())
    tn = float(np.logical_and(test_preds == 0, test_labels == 0).sum())

    iou      = tp / (tp + fp + fn + 1e-6)
    dice     = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    # ── Overlay images (up to 4 from test split) ──────────────────────────
    pb(progress_start + 0.5 * (progress_end - progress_start),
       "Generating U-Net segmentation overlays...")

    overlay_images = []
    vis_indices = test_indices[:4] if test_indices else list(range(min(4, N)))

    for t_idx in vis_indices:
        orig = X_imgs[t_idx]                                    # (H, W, 3) uint8
        gt   = y_masks[t_idx]                                  # (H, W) binary
        pred = all_preds[t_idx]                                 # (H, W) binary

        # Overlay: original + red highlight for predicted positives
        overlay = orig.copy()
        overlay[pred == 1] = [255, 50, 50]

        gt_vis  = np.stack([gt * 255, gt * 255, gt * 255], axis=-1).astype(np.uint8)
        strip   = np.concatenate([orig, gt_vis, overlay], axis=1)
        b64_str = _to_base64_png(strip)
        overlay_images.append({
            "label": f"Image {t_idx} — Original | Ground Truth | U-Net Prediction",
            "base64": b64_str,
        })

    pb(progress_end, "U-Net evaluation complete.")
    return iou, dice, accuracy, overlay_images
