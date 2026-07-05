"""
pipelines/cnn_extractor.py
──────────────────────────
Extracts 512-dimensional embeddings from image arrays using a pretrained
ResNet-18 backbone (torchvision). Runs on MPS (Apple Silicon GPU) when
available, falling back to CPU.

If torchvision is not installed, falls back to PCA(100) on flattened pixels.

Public API
----------
    DEVICE : torch.device
        Selected compute device (mps / cuda / cpu).

    extract_cnn_features(X_images, progress_cb=None)
        -> (embeddings: np.ndarray shape (N, 512|100), extractor_name: str)
"""

import os
import sys

# ── Metal / MPS safety ─────────────────────────────────────────────────────
# Disable Metal validation layer to prevent assertion aborts in Xcode subprocess.
# Enable op-level fallback so any MPS-unsupported op silently uses CPU.
os.environ.setdefault("MTL_DEBUG_LAYER", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Pin the torch hub cache to a stable user-level dir so the app sandbox
# and the venv both share the same downloaded weights.
os.environ.setdefault(
    "TORCH_HOME",
    os.path.expanduser("~/.cache/torch")
)

import numpy as np

try:
    from utils.helpers import print_progress as _print_progress
except ImportError:
    def _print_progress(frac, msg):
        sys.stderr.write(f"[{int(frac * 100)}%] {msg}\n")


# ── Device selection ───────────────────────────────────────────────────────

def _select_device():
    """
    Returns the best available torch.device for CV inference/training.

    Strategy (per-component — see implementation plan):
      1. MPS  — Apple Silicon GPU. Smoke-tested with an explicit on-device
                 scalar divide to confirm the Metal binding is functional.
      2. CUDA — NVIDIA (future-proofing).
      3. CPU  — universal fallback.

    The smoke-test uses `torch.tensor(scalar, device="mps")` rather than a
    plain Python float to avoid the CPU↔MPS scalar-binding Metal assertion
    that affects deep_learning.py's TabularNN trainer.
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            try:
                _t = torch.zeros(2, 2, device="mps")
                _s = torch.tensor(2.0, device="mps")   # scalar ON device
                _r = _t / _s
                del _t, _s, _r
                return torch.device("mps")
            except Exception as e:
                sys.stderr.write(
                    f"[cnn_extractor] MPS smoke-test failed ({e}), falling back to CPU.\n"
                )
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None   # torch not available → PCA path


DEVICE = _select_device()


# ── ResNet-18 extractor ────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Batch size for inference — keeps GPU memory usage < ~500 MB
_BATCH_SIZE = 64


def _build_resnet18_extractor():
    """
    Returns a ResNet-18 model with the final FC layer removed,
    ready for 512-d feature extraction.
    """
    import torch
    import torch.nn as nn
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except (ImportError, TypeError):
        # Older torchvision API
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)

    # Strip the classification head; keep the 512-d avgpool output
    model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


def _preprocess_batch(batch_np: np.ndarray):
    """
    Converts a uint8/float32 (N, H, W, 3) numpy batch to a normalised
    (N, 3, 224, 224) float32 torch.Tensor on DEVICE.
    """
    import torch
    import torch.nn.functional as F

    # Normalise pixel values to [0, 1]
    if batch_np.dtype != np.float32:
        batch_np = batch_np.astype(np.float32)
    if batch_np.max() > 1.01:
        batch_np = batch_np / 255.0

    # (N, H, W, 3) → (N, 3, H, W)
    t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(DEVICE)

    # Resize to 224×224 if needed
    if t.shape[-2:] != (224, 224):
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)

    # ImageNet normalisation — use tensors on DEVICE to avoid scalar bug
    mean = torch.tensor(_IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=DEVICE).view(1, 3, 1, 1)
    t = (t - mean) / std

    return t


def _extract_with_resnet(X_images: np.ndarray, progress_start=0.50, progress_end=0.70, progress_cb=None) -> np.ndarray:
    """
    Extracts (N, 512) float32 embeddings using ResNet-18 on DEVICE.
    """
    import torch
    model = _build_resnet18_extractor()
    N = len(X_images)
    embeddings = []

    pb = progress_cb or _print_progress

    with torch.no_grad():
        for i in range(0, N, _BATCH_SIZE):
            batch = X_images[i : i + _BATCH_SIZE]
            # Ensure 4-channel → 3-channel
            if batch.shape[-1] == 1:
                batch = np.repeat(batch, 3, axis=-1)
            elif batch.shape[-1] == 4:
                batch = batch[..., :3]

            t = _preprocess_batch(batch)
            feats = model(t)                   # (B, 512)
            embeddings.append(feats.cpu().numpy())

            frac = progress_start + (i / N) * (progress_end - progress_start)
            pb(frac, f"Extracting CNN features {min(i + _BATCH_SIZE, N)}/{N}...")

    return np.concatenate(embeddings, axis=0).astype(np.float32)


# ── PCA fallback ───────────────────────────────────────────────────────────

def _extract_with_pca(X_images: np.ndarray, n_components: int = 100) -> np.ndarray:
    """
    Fallback: PCA(n_components) on flattened pixels.
    Returns (N, n_components) float32 array.
    """
    from sklearn.decomposition import PCA
    N = len(X_images)
    flat_dim = int(np.prod(X_images.shape[1:]))
    X_flat = X_images.reshape(N, flat_dim).astype(np.float32)
    n_comps = min(n_components, N, flat_dim)
    pca = PCA(n_components=n_comps, random_state=42)
    return pca.fit_transform(X_flat).astype(np.float32)


# ── Public API ─────────────────────────────────────────────────────────────

def extract_cnn_features(
    X_images: np.ndarray,
    progress_cb=None,
    progress_start: float = 0.50,
    progress_end: float = 0.70,
) -> tuple:
    """
    Extract feature vectors from a batch of images.

    Parameters
    ----------
    X_images : np.ndarray
        Shape (N, H, W, C), uint8 or float32, values 0–255 or 0–1.
    progress_cb : callable(frac, msg) | None
        Optional progress reporter. Defaults to stderr.
    progress_start, progress_end : float
        Progress range to report within.

    Returns
    -------
    embeddings : np.ndarray  shape (N, 512) or (N, 100)
    extractor_name : str     "ResNet-18 (MPS)" / "ResNet-18 (CPU)" / "PCA"
    """
    pb = progress_cb or _print_progress

    # Try ResNet-18
    if DEVICE is not None:
        try:
            import torch          # noqa: F401 (confirm torch present)
            import torchvision    # noqa: F401 (confirm torchvision present)
            pb(progress_start, f"Extracting features via ResNet-18 on {DEVICE}...")
            embeddings = _extract_with_resnet(
                X_images,
                progress_start=progress_start,
                progress_end=progress_end,
                progress_cb=pb,
            )
            device_label = str(DEVICE).upper()
            return embeddings, f"ResNet-18 ({device_label})"
        except Exception as e:
            sys.stderr.write(
                f"[cnn_extractor] ResNet-18 extraction failed ({e}), "
                f"falling back to PCA.\n"
            )

    # PCA fallback
    pb(progress_start, "torchvision unavailable — using PCA feature reduction...")
    embeddings = _extract_with_pca(X_images)
    return embeddings, "PCA"
