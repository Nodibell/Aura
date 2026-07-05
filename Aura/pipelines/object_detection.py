"""
Object Detection pipeline for Aura — supports YOLO-format datasets.

Layout expected:
    <root>/
        dataset.yaml  (or data.yaml)   ← class names + split paths
        images/
            train/  *.jpg / *.png
            val/
            test/
        labels/
            train/  *.txt  (one per image, YOLO format)
            val/
            test/

Each label .txt has zero or more lines:
    <class_id>  <cx>  <cy>  <width>  <height>   (all normalised 0..1)

Variant A — EDA:
    • Class distribution bar chart
    • Bounding-box size distribution (w×h scatter)
    • Spatial heatmap of object centres (cx, cy)
    • Per-split statistics table

Variant B — Crop Classifier:
    • Extract every labelled bounding-box crop (resized to 64×64)
    • Train Logistic Regression + Random Forest on flattened pixel features
    • Report Accuracy, F1, Confusion Matrix per class
"""

import os
import sys
import json
import math
import base64
import random
import numpy as np
from io import BytesIO
from collections import defaultdict, Counter

from utils.helpers import print_progress, clean_nan


# ─── YAML loader (no PyYAML dependency) ─────────────────────────────────────

def _parse_yaml_simple(path):
    """Minimal YAML parser for dataset.yaml.
    Handles:  key: value,  key: [a, b, c],  indented list items (- item)
    Does NOT handle nested dicts or anchors.
    """
    result = {}
    current_key = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            # Indented list item
            stripped = line.lstrip()
            if stripped.startswith("- ") and current_key:
                val = stripped[2:].strip().strip("'\"")
                if isinstance(result.get(current_key), list):
                    result[current_key].append(val)
                else:
                    result[current_key] = [val]
                continue
            if ":" in line and not line.startswith(" "):
                key, _, rest = line.partition(":")
                key = key.strip()
                rest = rest.strip()
                current_key = key
                if not rest:
                    result[key] = []
                elif rest.startswith("["):
                    # Inline list: [a, b, c]
                    inner = rest.strip("[]")
                    result[key] = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                else:
                    # Try int, then keep string
                    try:
                        result[key] = int(rest)
                    except ValueError:
                        result[key] = rest.strip("'\"")
    return result


# ─── Dataset discovery ───────────────────────────────────────────────────────

def _find_yaml(root):
    if not root or not os.path.isdir(root):
        return None
    for name in ("dataset.yaml", "data.yaml", "yolo.yaml"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    # Search one level down
    try:
        for entry in os.scandir(root):
            if entry.is_dir():
                for name in ("dataset.yaml", "data.yaml", "yolo.yaml"):
                    p = os.path.join(entry.path, name)
                    if os.path.exists(p):
                        return p
    except Exception:
        pass
    return None


def _find_splits(root, yaml_info):
    """Return dict: split_name -> (images_dir, labels_dir, format_type)  for existing splits."""
    splits = {}
    # Prefer paths declared in yaml
    for split in ("train", "val", "test"):
        rel = yaml_info.get(split)
        if rel:
            img_dir = os.path.join(root, rel) if not os.path.isabs(rel) else rel
            # labels live alongside or in sibling 'labels' folder
            lab_dir = img_dir.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
            if os.path.isdir(img_dir):
                splits[split] = (img_dir, lab_dir if os.path.isdir(lab_dir) else None, "yolo")
    # Fall back: look for images/ + labels/ (YOLO) or images/ + annotations/ (VOC)
    if not splits and root and os.path.isdir(root):
        bases = [root]
        try:
            for entry in os.scandir(root):
                if entry.is_dir():
                    bases.append(entry.path)
        except Exception:
            pass

        for base in bases:
            img_root = os.path.join(base, "images")
            lab_root = os.path.join(base, "labels")
            ann_root = os.path.join(base, "annotations")
            if os.path.isdir(img_root):
                # Try VOC format first
                if os.path.isdir(ann_root):
                    has_sub = False
                    for split in ("train", "val", "test"):
                        img_dir = os.path.join(img_root, split)
                        ann_dir = os.path.join(ann_root, split)
                        if os.path.isdir(img_dir) and os.path.isdir(ann_dir):
                            splits[split] = (img_dir, ann_dir, "voc")
                            has_sub = True
                    if not has_sub:
                        splits["train"] = (img_root, ann_root, "voc")
                    break
                # Try YOLO format next
                elif os.path.isdir(lab_root):
                    has_sub = False
                    for split in ("train", "val", "test"):
                        img_dir = os.path.join(img_root, split)
                        lab_dir = os.path.join(lab_root, split)
                        if os.path.isdir(img_dir) and os.path.isdir(lab_dir):
                            splits[split] = (img_dir, lab_dir, "yolo")
                            has_sub = True
                    if not has_sub:
                        splits["train"] = (img_root, lab_root, "yolo")
                    break
    return splits


def _list_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for fname in os.listdir(img_dir):
        if os.path.splitext(fname)[1].lower() in exts:
            paths.append(os.path.join(img_dir, fname))
    return sorted(paths)


def _discover_voc_classes(ann_dir):
    classes = set()
    if not os.path.isdir(ann_dir):
        return []
    try:
        import xml.etree.ElementTree as ET
        for fname in os.listdir(ann_dir):
            if fname.lower().endswith(".xml"):
                p = os.path.join(ann_dir, fname)
                try:
                    tree = ET.parse(p)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        name_node = obj.find("name")
                        if name_node is not None and name_node.text:
                            classes.add(name_node.text.strip())
                except Exception:
                    pass
    except Exception:
        pass
    return sorted(list(classes))


def _parse_voc_label_file(path, class_names):
    """Parse a Pascal VOC XML annotation file.
    Returns list of (class_id, cx, cy, w, h) in normalised YOLO format.
    """
    boxes = []
    if not os.path.exists(path):
        return boxes
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        size_node = root.find("size")
        if size_node is not None:
            width = float(size_node.find("width").text)
            height = float(size_node.find("height").text)
        else:
            width, height = 0.0, 0.0

        if width <= 0 or height <= 0:
            return boxes

        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is None or not name_node.text:
                continue
            class_name = name_node.text.strip()

            # Map name to ID
            if class_name in class_names:
                cid = class_names.index(class_name)
            else:
                cid = len(class_names)
                class_names.append(class_name)

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            w_box = xmax - xmin
            h_box = ymax - ymin
            cx = xmin + w_box / 2.0
            cy = ymin + h_box / 2.0

            # Normalise
            cx_norm = cx / width
            cy_norm = cy / height
            w_norm = w_box / width
            h_norm = h_box / height

            boxes.append((cid, cx_norm, cy_norm, w_norm, h_norm))
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to parse XML {path}: {e}\n")
    return boxes


def _parse_label_file(path):
    """Return list of (class_id, cx, cy, w, h) for one YOLO .txt file."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    boxes.append((int(parts[0]),
                                  float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
                except ValueError:
                    pass
    return boxes


# ─── Chart helpers ───────────────────────────────────────────────────────────

def _encode_pil(img, quality=70):
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bar_chart(title, x_label, y_label, names, counts):
    data = [{"x_val": n, "x_num": None, "y": float(c), "series": None}
            for n, c in zip(names, counts)]
    return {"type": "bar", "title": title,
            "x_label": x_label, "y_label": y_label,
            "data": data, "images": None, "box_stats": None}


def _scatter_chart(title, x_label, y_label, xs, ys):
    data = [{"x_val": None, "x_num": float(x), "y": float(y), "series": None}
            for x, y in zip(xs, ys)]
    return {"type": "scatter", "title": title,
            "x_label": x_label, "y_label": y_label,
            "data": data, "images": None, "box_stats": None}


def _image_grid_chart(title, images_b64, labels):
    items = [{"label": lbl, "base64": b64} for b64, lbl in zip(images_b64, labels)]
    return {"type": "image_grid", "title": title,
            "x_label": "", "y_label": "",
            "data": [], "images": items, "box_stats": None}


# ─── Variant A — EDA ─────────────────────────────────────────────────────────

def _eda_charts(all_boxes, class_names, splits_info, max_scatter=2000):
    """Build EDA charts from collected box data.

    all_boxes: list of (class_id, cx, cy, w, h)
    """
    charts = []

    # 1. Class distribution
    class_counts = Counter(b[0] for b in all_boxes)
    sorted_ids = sorted(class_counts.keys())
    names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in sorted_ids]
    counts = [class_counts[i] for i in sorted_ids]
    charts.append(_bar_chart("Class Distribution", "Class", "Annotation Count", names, counts))

    # 2. Per-split image / annotation count
    split_names = list(splits_info.keys())
    split_img_counts = [splits_info[s]["n_images"] for s in split_names]
    split_ann_counts = [splits_info[s]["n_annotations"] for s in split_names]
    charts.append(_bar_chart("Images per Split", "Split", "Image Count",
                             split_names, split_img_counts))
    charts.append(_bar_chart("Annotations per Split", "Split", "Annotation Count",
                             split_names, split_ann_counts))

    # 3. Box area distribution (w * h, expressed as % of image)
    areas = [b[3] * b[4] * 100 for b in all_boxes]  # percentage
    # Bucket into 20 bins
    if areas:
        max_area = max(areas)
        n_bins = 20
        bin_size = max(max_area / n_bins, 0.001)
        bin_counts = defaultdict(int)
        for a in areas:
            bucket = round(math.floor(a / bin_size) * bin_size, 4)
            bin_counts[bucket] += 1
        sorted_bins = sorted(bin_counts.keys())
        charts.append({
            "type": "bar",
            "title": "Bounding Box Area Distribution (% of image)",
            "x_label": "Box Area (% image area)",
            "y_label": "Count",
            "data": [{"x_val": f"{k:.2f}%", "x_num": float(k), "y": float(bin_counts[k]), "series": None}
                     for k in sorted_bins],
            "images": None,
            "box_stats": None
        })

    # 4. Spatial heatmap (cx, cy scatter — subsample for performance)
    cxs = [b[1] for b in all_boxes]
    cys = [b[2] for b in all_boxes]
    if len(cxs) > max_scatter:
        idx = random.sample(range(len(cxs)), max_scatter)
        cxs = [cxs[i] for i in idx]
        cys = [cys[i] for i in idx]
    charts.append(_scatter_chart(
        "Object Centre Spatial Distribution",
        "Centre X (normalised)", "Centre Y (normalised)",
        cxs, cys
    ))

    # 5. Box aspect ratio distribution
    ratios = [b[3] / b[4] if b[4] > 0 else 0 for b in all_boxes]
    if ratios:
        ratio_bins = defaultdict(int)
        bin_size = 0.25
        for r in ratios:
            bucket = round(math.floor(r / bin_size) * bin_size, 2)
            ratio_bins[bucket] += 1
        sorted_rbins = sorted(ratio_bins.keys())[:40]  # cap display
        charts.append({
            "type": "bar",
            "title": "Bounding Box Aspect Ratio Distribution (width / height)",
            "x_label": "Aspect Ratio",
            "y_label": "Count",
            "data": [{"x_val": f"{k:.2f}", "x_num": float(k), "y": float(ratio_bins[k]), "series": None}
                     for k in sorted_rbins],
            "images": None,
            "box_stats": None
        })

    return charts


# ─── Variant B — Crop Classifier ─────────────────────────────────────────────

def _crop_and_classify(splits, class_names, max_crops_per_class=200, crop_size=64):
    """
    For each labelled box in train split, cut the crop from the image and
    build an (X, y) array.  Then train LR + RF and report metrics.
    Returns (metrics_dict, confusion_matrix_dict, models_compared_list, sample_images_per_class)
    """
    try:
        from PIL import Image
    except ImportError:
        return None, None, None, None, None

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.dummy import DummyClassifier

    print_progress(0.55, "Extracting bounding-box crops for classifier…")

    crops_by_class = defaultdict(list)
    sample_imgs_by_class = {}   # one sample crop per class (b64)

    train_info = splits.get("train", (None, None, "yolo"))
    train_img_dir, train_lab_dir, fmt = train_info
    if train_img_dir is None or train_lab_dir is None:
        return None, None, None, None, None

    img_paths = _list_images(train_img_dir)
    random.shuffle(img_paths)

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if fmt == "voc":
            xml_path = os.path.join(train_lab_dir, stem + ".xml")
            boxes = _parse_voc_label_file(xml_path, class_names)
        else:
            lab_path = os.path.join(train_lab_dir, stem + ".txt")
            boxes = _parse_label_file(lab_path)
        if not boxes:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
        except Exception:
            continue

        for cid, cx, cy, bw, bh in boxes:
            if all(len(v) >= max_crops_per_class for v in crops_by_class.values() if v):
                # Check if this class still needs more
                if len(crops_by_class[cid]) >= max_crops_per_class:
                    continue

            # Convert normalised YOLO → pixel coords
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            try:
                crop = img.crop((x1, y1, x2, y2)).resize((crop_size, crop_size))
                arr = np.array(crop).flatten().astype(np.float32) / 255.0
                crops_by_class[cid].append(arr)

                if cid not in sample_imgs_by_class:
                    sample_imgs_by_class[cid] = _encode_pil(crop)
            except Exception:
                continue

        # Early stop once all classes have enough
        if all(len(crops_by_class[c]) >= max_crops_per_class
               for c in range(len(class_names))):
            break

    if not crops_by_class or len(crops_by_class) < 2:
        return None, None, None, None, None

    print_progress(0.62, "Building feature matrix…")
    X_list, y_list = [], []
    for cid, arrs in crops_by_class.items():
        X_list.extend(arrs)
        y_list.extend([cid] * len(arrs))

    X = np.array(X_list)
    y = np.array(y_list)

    # Shuffle + split 80/20
    idx = np.random.permutation(len(X))
    split_at = int(len(X) * 0.8)
    X_train, X_test = X[idx[:split_at]], X[idx[split_at:]]
    y_train, y_test = y[idx[:split_at]], y[idx[split_at:]]

    print_progress(0.68, "Training Logistic Regression on crops…")
    lr = LogisticRegression(max_iter=500, random_state=42, C=0.1)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)

    print_progress(0.76, "Training Random Forest on crops…")
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=2)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    # Dummy baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

    # Pick best
    if rf_acc >= lr_acc:
        best_preds = rf_preds
        best_acc = rf_acc
        best_name = "Random Forest (crops)"
    else:
        best_preds = lr_preds
        best_acc = lr_acc
        best_name = "Logistic Regression (crops)"

    f1 = f1_score(y_test, best_preds, average="weighted", zero_division=0)

    # Confusion matrix
    unique_classes = sorted(crops_by_class.keys())
    cm = confusion_matrix(y_test, best_preds, labels=unique_classes)
    cm_labels = [class_names[c] if c < len(class_names) else f"class_{c}"
                 for c in unique_classes]

    metrics = {
        "model": best_name,
        "score_type": "Accuracy",
        "score": float(best_acc),
        "additional_metrics": {"F1 (weighted)": float(f1)}
    }
    models_compared = [
        {"name": "Dummy Baseline", "score": float(dummy_acc), "metric": "Accuracy"},
        {"name": "Logistic Regression (crops)", "score": float(lr_acc), "metric": "Accuracy"},
        {"name": "Random Forest (crops)", "score": float(rf_acc), "metric": "Accuracy"},
    ]
    cm_data = {"labels": cm_labels, "values": cm.tolist()}

    # Sample image grid (one per class)
    sample_images = []
    for cid in unique_classes:
        if cid in sample_imgs_by_class:
            lbl = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            sample_images.append({"label": lbl, "base64": sample_imgs_by_class[cid]})

    importance_chart = None
    if hasattr(rf, 'feature_importances_'):
        imp = rf.feature_importances_
        # Channel averages:
        r_imp = float(np.mean(imp[0::3]))
        g_imp = float(np.mean(imp[1::3]))
        b_imp = float(np.mean(imp[2::3]))
        
        # Spatial averages:
        center_imps = []
        border_imps = []
        for idx_pixel in range(crop_size * crop_size):
            py = idx_pixel // crop_size
            px = idx_pixel % crop_size
            pixel_imp = sum(imp[3 * idx_pixel + c] for c in range(3))
            if 16 <= px < 48 and 16 <= py < 48:
                center_imps.append(pixel_imp)
            else:
                border_imps.append(pixel_imp)
        
        center_avg = float(np.mean(center_imps)) if center_imps else 0.0
        border_avg = float(np.mean(border_imps)) if border_imps else 0.0

        # Normalise so that the sum of these values is 1.0
        total_sum = r_imp + g_imp + b_imp + center_avg + border_avg
        if total_sum > 0:
            r_imp /= total_sum
            g_imp /= total_sum
            b_imp /= total_sum
            center_avg /= total_sum
            border_avg /= total_sum

        importance_chart = {
            "type": "bar",
            "title": "Random Forest Crop Feature Importances",
            "x_label": "Feature Group",
            "y_label": "Relative Importance",
            "data": [
                {"x_val": "Red Channel", "x_num": None, "y": r_imp, "series": None},
                {"x_val": "Green Channel", "x_num": None, "y": g_imp, "series": None},
                {"x_val": "Blue Channel", "x_num": None, "y": b_imp, "series": None},
                {"x_val": "Center Pixels (32x32)", "x_num": None, "y": center_avg, "series": None},
                {"x_val": "Border Pixels", "x_num": None, "y": border_avg, "series": None}
            ],
            "images": None,
            "box_stats": None
        }

    return metrics, cm_data, models_compared, sample_images, importance_chart


# ─── Preview helpers for preview.py ─────────────────────────────────────────

def preview_yolo(root):
    """Return a DatasetPreview-compatible dict for a YOLO or Pascal VOC dataset folder."""
    if root and os.path.isfile(root):
        root = os.path.dirname(root)
    yaml_path = _find_yaml(root)
    yaml_info = _parse_yaml_simple(yaml_path) if yaml_path else {}
    class_names = yaml_info.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names.keys())]

    splits = _find_splits(root, yaml_info)

    # Discover class names if not in YAML (Pascal VOC format)
    if not class_names:
        voc_ann_dir = None
        for split, (img_dir, lab_dir, fmt) in splits.items():
            if fmt == "voc" and lab_dir:
                voc_ann_dir = lab_dir
                break
        if voc_ann_dir:
            class_names = _discover_voc_classes(voc_ann_dir)

    row_count = 0
    ann_count = 0
    for split, (img_dir, lab_dir, fmt) in splits.items():
        imgs = _list_images(img_dir)
        row_count += len(imgs)
        if lab_dir:
            use_estimation = len(imgs) > 1000
            sampled_imgs = imgs
            if use_estimation:
                import random
                # Use a fixed seed for reproducible preview counts
                random.seed(42)
                sampled_imgs = random.sample(imgs, 1000)
            
            sampled_ann_count = 0
            for p in sampled_imgs:
                stem = os.path.splitext(os.path.basename(p))[0]
                if fmt == "voc":
                    sampled_ann_count += len(_parse_voc_label_file(os.path.join(lab_dir, stem + ".xml"), class_names))
                else:
                    sampled_ann_count += len(_parse_label_file(os.path.join(lab_dir, stem + ".txt")))
            
            if use_estimation:
                ann_count += int((sampled_ann_count / 1000.0) * len(imgs))
            else:
                ann_count += sampled_ann_count

    # Build a mini preview table: filename, split, n_boxes, classes_present
    preview_rows = []
    for split, (img_dir, lab_dir, fmt) in splits.items():
        img_paths = _list_images(img_dir)[:5]
        for p in img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            boxes = []
            if lab_dir:
                if fmt == "voc":
                    boxes = _parse_voc_label_file(os.path.join(lab_dir, stem + ".xml"), class_names)
                else:
                    boxes = _parse_label_file(os.path.join(lab_dir, stem + ".txt"))
            classes_present = ", ".join(sorted(set(
                class_names[b[0]] if b[0] < len(class_names) else f"class_{b[0]}"
                for b in boxes
            ))) or "—"
            preview_rows.append([
                os.path.basename(p),
                split,
                str(len(boxes)),
                classes_present
            ])

    cols = ["filename", "split", "n_boxes", "classes_present"]
    n_classes = len(class_names)

    return {
        "columns": cols,
        "preview_rows": preview_rows,
        "inferred_dataset_type": "object_detection",
        "local_path": root,
        "total_rows": row_count,
        "available_files": [root],
        # Extra metadata shown as a hint in the UI
        "_od_meta": {
            "n_classes": n_classes,
            "class_names": class_names,
            "n_annotations": ann_count,
            "splits": {s: len(_list_images(d[0])) for s, d in splits.items()},
            "yaml_path": yaml_path or ""
        }
    }


# ─── Main entry point ────────────────────────────────────────────────────────

def analyze_object_detection(file_path, task_type_override="auto",
                              target_col=None, test_file_path=None,
                              model_export_path=None, code_export_path=None):
    try:
        root = file_path  # the directory dropped / entered
        if root and os.path.isfile(root):
            root = os.path.dirname(root)

        # ── 0. YOLO training mode ──────────────────────────────────────────
        if task_type_override == "train":
            from pipelines.yolo_trainer import train_yolo_and_evaluate
            return train_yolo_and_evaluate(root, model_export_path=model_export_path)

        # ── 1. Parse YAML ──────────────────────────────────────────────────
        print_progress(0.05, "Parsing YOLO dataset.yaml…")
        yaml_path = _find_yaml(root)
        if yaml_path:
            yaml_info = _parse_yaml_simple(yaml_path)
        else:
            yaml_info = {}
            sys.stderr.write("Warning: No dataset.yaml / data.yaml found. "
                             "Proceeding with auto-discovery.\n")

        class_names = yaml_info.get("names", [])
        if isinstance(class_names, dict):
            class_names = [class_names[k] for k in sorted(class_names.keys())]
        n_classes = int(yaml_info.get("nc", len(class_names)))

        # ── 2. Discover splits ─────────────────────────────────────────────
        print_progress(0.10, "Discovering dataset splits…")
        splits = _find_splits(root, yaml_info)
        if not splits:
            return {"error": "Could not find any images/ split folders. "
                             "Expected structure: images/train/, images/val/, images/test/."}

        # Discover class names if not in YAML (Pascal VOC format)
        if not class_names:
            voc_ann_dir = None
            for split, (img_dir, lab_dir, fmt) in splits.items():
                if fmt == "voc" and lab_dir:
                    voc_ann_dir = lab_dir
                    break
            if voc_ann_dir:
                class_names = _discover_voc_classes(voc_ann_dir)
            n_classes = max(n_classes, len(class_names))

        # ── 3. Collect all annotations ─────────────────────────────────────
        print_progress(0.20, "Parsing label files…")
        all_boxes = []          # (class_id, cx, cy, w, h)
        splits_info = {}        # split → {n_images, n_annotations}

        for split, (img_dir, lab_dir, fmt) in splits.items():
            img_paths = _list_images(img_dir)
            split_boxes = []
            for p in img_paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                if fmt == "voc":
                    xml_path = os.path.join(lab_dir, stem + ".xml")
                    boxes = _parse_voc_label_file(xml_path, class_names)
                else:
                    lab_path = os.path.join(lab_dir, stem + ".txt") if lab_dir else ""
                    boxes = _parse_label_file(lab_path)
                split_boxes.extend(boxes)
            all_boxes.extend(split_boxes)
            splits_info[split] = {
                "n_images": len(img_paths),
                "n_annotations": len(split_boxes)
            }

        total_images = sum(v["n_images"] for v in splits_info.values())
        total_annotations = sum(v["n_annotations"] for v in splits_info.values())

        # Compute correlations of bounding box features
        correlations = []
        if all_boxes:
            try:
                import pandas as pd
                box_df = pd.DataFrame(all_boxes, columns=["class_id", "x_center", "y_center", "width", "height"])
                box_df["area"] = box_df["width"] * box_df["height"]
                box_df["aspect_ratio"] = box_df["width"] / box_df["height"].replace(0, 1e-6)
                
                numeric_cols = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]
                corr_matrix = box_df[numeric_cols].corr()
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        val = corr_matrix.loc[col1, col2]
                        if not pd.isna(val):
                            correlations.append({
                                "x": col1,
                                "y": col2,
                                "value": float(val)
                             })
            except Exception as e:
                sys.stderr.write(f"Warning: Bounding box correlations failed: {e}\n")

        # Auto-extend class_names if labels use higher IDs
        if all_boxes:
            max_id = max(b[0] for b in all_boxes)
            while len(class_names) <= max_id:
                class_names.append(f"class_{len(class_names)}")

        # ── 4. Variant A — EDA charts ──────────────────────────────────────
        print_progress(0.35, "Generating EDA charts…")
        charts = _eda_charts(all_boxes, class_names, splits_info)

        # ── 5. Variant B — Crop Classifier ────────────────────────────────
        metrics = None
        cm_data = None
        models_compared = []
        dummy_baseline_score = None
        sample_images_chart = None

        train_info = splits.get("train", (None, None, "yolo"))
        train_img_dir = train_info[0]
        if train_img_dir is not None:
            try:
                print_progress(0.50, "Running crop-based classifier (Variant B)…")
                metrics, cm_data, models_compared, sample_imgs, importance_chart = \
                    _crop_and_classify(splits, class_names)

                if metrics is not None:
                    dummy_baseline_score = models_compared[0]["score"] if models_compared else None

                    # Add sample grid chart
                    if sample_imgs:
                        charts.append(_image_grid_chart(
                            "Sample Bounding-Box Crops (one per class)",
                            [s["base64"] for s in sample_imgs],
                            [s["label"] for s in sample_imgs]
                        ))
                        
                    # Add Random Forest importance chart
                    if importance_chart:
                        charts.append(importance_chart)
            except Exception as e:
                sys.stderr.write(f"Warning: Crop classifier failed: {e}\n")

        # ── 6. Fallback metrics if classifier not available ────────────────
        if metrics is None:
            # No ML result — still show EDA
            class_counts = Counter(b[0] for b in all_boxes)
            most_common_id = class_counts.most_common(1)[0][0] if class_counts else 0
            most_common_name = class_names[most_common_id] if most_common_id < len(class_names) else "unknown"
            metrics = {
                "model": "EDA only (no classifier)",
                "score_type": "Annotation Count",
                "score": float(total_annotations),
                "additional_metrics": {
                    "Total Images": float(total_images),
                    "Num Classes": float(n_classes),
                    "Avg Boxes/Image": round(total_annotations / max(total_images, 1), 2)
                }
            }
            models_compared = []

        # ── 7. Summary text ────────────────────────────────────────────────
        print_progress(0.92, "Assembling results…")
        # ── 7. Summary text (rich markdown format) ─────────────────────────
        print_progress(0.92, "Assembling results…")
        split_summary_list = [
            f"- **{s.capitalize()} Split:** {v['n_images']:,} images, {v['n_annotations']:,} annotations"
            for s, v in splits_info.items()
        ]
        split_info_text = "\n".join(split_summary_list)
        
        class_list = ", ".join(f"`{c}`" for c in class_names[:15])
        if len(class_names) > 15:
            class_list += f" … (+{len(class_names) - 15} more)"

        avg_boxes = round(total_annotations / max(total_images, 1), 2)
        
        summary_sections = []
        
        # 1. Dataset Overview
        overview = (
            f"### 📊 Dataset Overview\n"
            f"- **Images:** {total_images:,} | **Annotations:** {total_annotations:,}\n"
            f"- **Object Classes:** {n_classes:,} unique categories\n"
            f"- **Average Density:** {avg_boxes:.2f} objects per image"
        )
        summary_sections.append(overview)
        
        # 2. Split Details
        splits_sec = (
            f"### 📁 Dataset Splits\n"
            f"{split_info_text}"
        )
        summary_sections.append(splits_sec)
        
        # 3. Target Variable / Classes
        target_sec = (
            f"### 🎯 Target Categories\n"
            f"- **Detected Classes:** {class_list}\n"
        )
        # Class counts
        class_counts = Counter(b[0] for b in all_boxes)
        if class_counts:
            sorted_counts = class_counts.most_common(3)
            majority_class_info = []
            for cid, cnt in sorted_counts:
                cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                pct = (cnt / total_annotations) * 100
                majority_class_info.append(f"`{cname}` ({pct:.1f}%)")
            target_sec += f"- **Top Categories:** {', '.join(majority_class_info)}"
        summary_sections.append(target_sec)
        
        # 4. ML Model Performance
        if metrics is not None:
            model_perf = (
                f"### 🤖 Machine Learning Model Performance\n"
                f"- **Best Model:** `{metrics['model']}`\n"
                f"- **Primary Score ({metrics['score_type']}):** `{metrics['score']:.4f}`\n"
            )
            if 'additional_metrics' in metrics and metrics['additional_metrics']:
                for am_name, am_val in metrics['additional_metrics'].items():
                    model_perf += f"- **{am_name}:** `{am_val:.4f}`\n"
            summary_sections.append(model_perf)
            
        summary = "\n\n".join(summary_sections)

        # Build the column / row info for the Data tab
        # Represent each annotation as a row in the preview table
        preview_rows = []
        for split, (img_dir, lab_dir, fmt) in splits.items():
            for p in _list_images(img_dir)[:100]:
                stem = os.path.splitext(os.path.basename(p))[0]
                if fmt == "voc":
                    boxes = _parse_voc_label_file(
                        os.path.join(lab_dir, stem + ".xml") if lab_dir else "", class_names
                    )
                else:
                    boxes = _parse_label_file(
                        os.path.join(lab_dir, stem + ".txt") if lab_dir else ""
                    )
                for cid, cx, cy, bw, bh in boxes:
                    cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                    preview_rows.append([
                        stem, split, cname,
                        f"{cx:.4f}", f"{cy:.4f}", f"{bw:.4f}", f"{bh:.4f}"
                    ])

        preview_cols = ["image_stem", "split", "class", "cx", "cy", "width", "height"]
        full_preview = {
            "columns": preview_cols,
            "rows": preview_rows[:500],
            "total_rows": total_annotations
        }

        # Profiling for the Summary tab
        class_counts_dict = Counter(b[0] for b in all_boxes)
        profiling_cols = {}
        for cid, cnt in class_counts_dict.items():
            cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            profiling_cols[cname] = {
                "nunique": 1,
                "missing": 0,
                "type": "categorical",
                "stats": None,
                "top_categories": [{"value": cname, "count": cnt}]
            }
        profiling = {
            "duplicate_rows": 0,
            "columns": profiling_cols
        }

        missing_values = {c: 0 for c in preview_cols}

        result = {
            "summary": summary,
            "columns": preview_cols,
            "row_count": total_annotations,
            "col_count": len(preview_cols),
            "task_type": "object_detection",
            "numeric_col_count": 4,
            "categorical_col_count": 3,
            "text_col_count": 0,
            "missing_values": missing_values,
            "correlations": correlations,
            "charts": charts,
            "metrics": metrics,
            "models_compared": models_compared,
            "target_column": "class",
            "dummy_baseline_score": dummy_baseline_score,
            "cv_scores": None,
            "cv_mean": None,
            "cv_std": None,
            "confusion_matrix": cm_data,
            "profiling": profiling,
            "full_preview": full_preview,
            "test_row_count": None,
            "test_col_count": None,
            "test_missing_values": None,
            "test_correlations": None,
            "test_profiling": None,
            "test_full_preview": None,
            "warning": None,
            "error": None
        }
        return clean_nan(result)

    except Exception as e:
        import traceback
        return {"error": f"Object detection pipeline failed: {e}\n{traceback.format_exc()}"}
