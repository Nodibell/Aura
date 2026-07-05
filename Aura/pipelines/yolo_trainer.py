"""
pipelines/yolo_trainer.py
──────────────────────────
Fine-tunes a YOLOv8-nano model on a local YOLO-format dataset and returns
an evaluation result dict compatible with the existing object_detection
pipeline result structure.

Dataset layout expected (same as object_detection.py):
    <root>/
        dataset.yaml  (or data.yaml)
        images/train/ val/ test/
        labels/train/ val/ test/

Public API
----------
    train_yolo_and_evaluate(root_dir, model_export_path=None) -> dict
"""

import os
import sys
import io
import base64
import json
import tempfile

import numpy as np

os.environ.setdefault("MTL_DEBUG_LAYER", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    from utils.helpers import print_progress
except ImportError:
    def print_progress(frac, msg):
        sys.stderr.write(f"[{int(frac * 100)}%] {msg}\n")


# ── YAML helper (re-used from object_detection.py) ─────────────────────────

def _find_yaml(root):
    if not root or not os.path.isdir(root):
        return None
    for name in ("dataset.yaml", "data.yaml", "yolo.yaml"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    try:
        for entry in os.scandir(root):
            if entry.is_dir():
                for name in ("dataset.yaml", "data.yaml", "yolo.yaml"):
                    p = os.path.join(entry.path, name)
                    if os.path.exists(p):
                        return p
    except OSError:
        pass
    return None


def _to_base64_png(img_arr: np.ndarray) -> str:
    from PIL import Image as _PILImage
    if img_arr.dtype != np.uint8:
        if img_arr.max() <= 1.01:
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        else:
            img_arr = img_arr.clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    _PILImage.fromarray(img_arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _draw_boxes_on_image(img_path: str, boxes, class_names: list) -> np.ndarray:
    """
    Draw YOLO prediction bounding boxes on an image.
    boxes: list of (class_id, conf, x1, y1, x2, y2) in pixel coordinates.
    """
    from PIL import Image as _PILImage, ImageDraw, ImageFont

    img = _PILImage.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    colours = [
        "#FF4136", "#2ECC40", "#0074D9", "#FF851B",
        "#B10DC9", "#FFDC00", "#7FDBFF", "#F012BE",
    ]
    for (cls_id, conf, x1, y1, x2, y2) in boxes:
        colour = colours[int(cls_id) % len(colours)]
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        label = f"{class_names[int(cls_id)] if int(cls_id) < len(class_names) else cls_id}: {conf:.2f}"
        draw.text((x1 + 2, y1 + 2), label, fill=colour)
    return np.array(img)


# ── Main training function ─────────────────────────────────────────────────

def train_yolo_and_evaluate(
    root_dir: str,
    model_export_path: str = None,
    epochs: int = 10,
    imgsz: int = 640,
) -> dict:
    """
    Fine-tune YOLOv8n on `root_dir` and return a result dict.
    Uses MPS on Apple Silicon (ultralytics supports device="mps" natively).
    `workers=0` avoids fork() sandbox issues in Xcode subprocess context.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {
            "error": (
                "ultralytics package is required for YOLO training. "
                "Run: pip install ultralytics>=8.3.0"
            )
        }

    # ── Detect device ──────────────────────────────────────────────────────
    try:
        from pipelines.cnn_extractor import DEVICE
        device_str = str(DEVICE) if DEVICE is not None else "cpu"
    except ImportError:
        try:
            import torch
            device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            device_str = "cpu"

    # ── Locate YAML ────────────────────────────────────────────────────────
    print_progress(0.05, "Locating YOLO dataset YAML...")
    yaml_path = _find_yaml(root_dir)
    if not yaml_path:
        return {"error": f"Could not find dataset.yaml / data.yaml in '{root_dir}'."}

    # ── Parse class names from YAML ────────────────────────────────────────
    class_names = []
    try:
        import yaml as _yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, dict):
            class_names = [names[k] for k in sorted(names.keys())]
        elif isinstance(names, list):
            class_names = [str(n) for n in names]
    except Exception:
        pass

    # ── Load base model ────────────────────────────────────────────────────
    print_progress(0.10, "Loading YOLOv8n base model...")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        return {"error": f"Failed to load YOLOv8n weights: {e}"}

    # ── Training ───────────────────────────────────────────────────────────
    print_progress(0.15, f"Fine-tuning YOLOv8n for {epochs} epochs on {device_str}...")

    train_results = None
    try:
        # workers=0: avoids multiprocessing fork issues in Xcode sandbox
        train_results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            device=device_str,
            verbose=False,
            workers=0,
            exist_ok=True,
        )
    except Exception as e:
        return {"error": f"YOLO training failed: {e}"}

    print_progress(0.70, "Running validation (mAP evaluation)...")

    # ── Validation metrics ─────────────────────────────────────────────────
    map50    = 0.0
    map50_95 = 0.0
    try:
        val_metrics = model.val(device=device_str, verbose=False, workers=0)
        map50    = float(val_metrics.box.map50)
        map50_95 = float(val_metrics.box.map)
    except Exception as e:
        sys.stderr.write(f"[yolo_trainer] val() failed: {e}\n")

    # ── Export model ───────────────────────────────────────────────────────
    if model_export_path:
        try:
            model.export(format="torchscript") if model_export_path.endswith(".pt") else None
            # Save the ultralytics best.pt to the requested path
            import shutil
            best_pt = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")
            if os.path.exists(best_pt):
                shutil.copy2(best_pt, model_export_path)
        except Exception as e:
            sys.stderr.write(f"[yolo_trainer] model export failed: {e}\n")

    # ── Generate bbox overlay images ───────────────────────────────────────
    print_progress(0.80, "Generating bounding-box overlay previews...")
    overlay_images = []

    try:
        # Find up to 16 validation images
        val_img_dirs = []
        for sub in ("val", "valid", "test"):
            candidate = os.path.join(root_dir, "images", sub)
            if os.path.isdir(candidate):
                val_img_dirs.append(candidate)
                break
        if not val_img_dirs:
            candidate = os.path.join(root_dir, "images")
            if os.path.isdir(candidate):
                val_img_dirs.append(candidate)

        img_paths = []
        for d in val_img_dirs:
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        img_paths.append(os.path.join(root, f))
                if len(img_paths) >= 16:
                    break

        img_paths = img_paths[:16]
        for img_path in img_paths:
            try:
                results = model.predict(
                    img_path,
                    device=device_str,
                    verbose=False,
                    conf=0.25,
                )
                r = results[0]
                boxes_raw = []
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy  = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss  = r.boxes.cls.cpu().numpy()
                    for j in range(len(xyxy)):
                        boxes_raw.append((
                            int(clss[j]), float(confs[j]),
                            float(xyxy[j][0]), float(xyxy[j][1]),
                            float(xyxy[j][2]), float(xyxy[j][3]),
                        ))
                drawn = _draw_boxes_on_image(img_path, boxes_raw, class_names)
                overlay_images.append({
                    "label": os.path.basename(img_path),
                    "base64": _to_base64_png(drawn),
                })
            except Exception as e:
                sys.stderr.write(f"[yolo_trainer] Prediction failed for {img_path}: {e}\n")
    except Exception as e:
        sys.stderr.write(f"[yolo_trainer] Overlay generation failed: {e}\n")

    print_progress(0.95, "Assembling training results...")

    # ── Build result dict (compatible with object_detection result structure) ──
    summary = (
        f"### 🎯 YOLOv8n Fine-Tuning Complete\n"
        f"- **Epochs:** {epochs}\n"
        f"- **Image size:** {imgsz}×{imgsz}\n"
        f"- **Device:** {device_str.upper()}\n"
        f"- **Classes:** {', '.join(class_names) if class_names else 'Unknown'}\n\n"
        f"### 📊 Validation Metrics\n"
        f"- **mAP@0.5:** `{map50:.4f}`\n"
        f"- **mAP@0.5:0.95:** `{map50_95:.4f}`\n"
    )

    charts = []
    if overlay_images:
        charts.append({
            "type": "image_grid",
            "title": "YOLOv8n Validation — Bounding Box Detections",
            "x_label": "",
            "y_label": "",
            "data": [],
            "images": overlay_images,
        })

    # Class-count bar if class names available
    if class_names:
        charts.append({
            "type": "bar",
            "title": "Dataset Classes",
            "x_label": "Class",
            "y_label": "Class ID",
            "data": [{"x_val": n, "y": float(i)} for i, n in enumerate(class_names)],
        })

    columns = ["metric", "value"]
    rows = [
        ["mAP@0.5", f"{map50:.4f}"],
        ["mAP@0.5:0.95", f"{map50_95:.4f}"],
        ["Epochs", str(epochs)],
        ["Device", device_str.upper()],
        ["Classes", str(len(class_names))],
    ]

    return {
        "summary": summary,
        "columns": columns,
        "row_count": len(rows),
        "col_count": 2,
        "task_type": "object_detection_train",
        "numeric_col_count": 2,
        "categorical_col_count": 0,
        "text_col_count": 0,
        "missing_values": {},
        "correlations": [],
        "charts": charts,
        "metrics": {
            "model": "YOLOv8n (fine-tuned)",
            "score_type": "mAP@0.5",
            "score": map50,
            "additional_metrics": {"mAP@0.5:0.95": map50_95},
        },
        "models_compared": [
            {"name": "YOLOv8n (fine-tuned)", "score": map50, "metric": "mAP@0.5"},
        ],
        "target_column": "class",
        "dummy_baseline_score": 0.0,
        "cv_scores": [map50],
        "cv_mean": map50,
        "cv_std": 0.0,
        "confusion_matrix": None,
        "profiling": {"duplicate_rows": 0, "columns": {}},
        "full_preview": {"columns": columns, "rows": rows, "total_rows": len(rows)},
        "test_row_count": None,
        "test_col_count": None,
        "test_missing_values": None,
        "test_correlations": None,
        "test_profiling": None,
        "test_full_preview": None,
        "error": None,
    }
