import os
import sys
import uuid
import zipfile
import shutil
import tempfile
import numpy as np
import pandas as pd
from PIL import Image

try:
    from utils.helpers import print_progress
except ImportError:
    def print_progress(fraction, message):
        sys.stderr.write(f"[{int(fraction * 100)}%] {message}\n")

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif')

def get_cache_dir():
    cache_dir = os.path.join(tempfile.gettempdir(), "aura_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def detect_dataset_format(dir_path):
    """
    Detects the dataset layout in the given directory.
    Returns one of: 'yolo', 'segmentation', 'class_hierarchy', 'flat', 'unknown'.
    """
    if not os.path.isdir(dir_path):
        return 'unknown'

    # Check for YOLO (YAML files or images/labels structure with .txt files)
    has_yaml = False
    for f in os.listdir(dir_path):
        if f.lower().endswith(('.yaml', '.yml')):
            yaml_path = os.path.join(dir_path, f)
            try:
                import yaml
                with open(yaml_path, 'r', encoding='utf-8') as yf:
                    data = yaml.safe_load(yf)
                    if isinstance(data, dict) and ('names' in data or 'nc' in data or 'train' in data):
                        has_yaml = True
                        break
            except Exception:
                pass

    images_subdirs = []
    labels_subdirs = []
    masks_subdirs = []
    other_subdirs = []

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            name_lower = item.lower()
            if name_lower in ['images', 'image', 'img']:
                images_subdirs.append(item_path)
            elif name_lower in ['labels']:
                labels_subdirs.append(item_path)
            elif name_lower in ['masks', 'mask']:
                masks_subdirs.append(item_path)
            elif name_lower not in ['__pycache__', 'temp', 'cache']:
                other_subdirs.append(item_path)

    # 1. YOLO Detection
    if has_yaml:
        return 'yolo'
        
    # Check if images/ and labels/ exist with label txt files
    if images_subdirs and labels_subdirs:
        # Check if labels contains any .txt files
        has_txt_labels = False
        for root, _, files in os.walk(labels_subdirs[0]):
            if any(f.lower().endswith('.txt') for f in files):
                has_txt_labels = True
                break
        if has_txt_labels:
            return 'yolo'

    # 2. Semantic Segmentation Detection
    # If we have parallel images/ and masks/ (or labels/ with image files inside)
    if images_subdirs and (masks_subdirs or labels_subdirs):
        target_dir = masks_subdirs[0] if masks_subdirs else labels_subdirs[0]
        # Check if the target directory contains images
        has_mask_images = False
        for root, _, files in os.walk(target_dir):
            if any(f.lower().endswith(IMAGE_EXTENSIONS) for f in files):
                has_mask_images = True
                break
        if has_mask_images:
            return 'segmentation'

    # 3. Class Hierarchy Detection
    # If there are subdirectories that contain images, we assume it's standard class_hierarchy
    has_class_folders = False
    for sub in other_subdirs:
        for root, _, files in os.walk(sub):
            if any(f.lower().endswith(IMAGE_EXTENSIONS) for f in files):
                has_class_folders = True
                break
        if has_class_folders:
            break
            
    if has_class_folders:
        return 'class_hierarchy'

    # 4. Flat Directory Detection
    # Check if root contains images directly
    has_flat_images = any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))
    if has_flat_images:
        return 'flat'

    return 'unknown'

def parse_metadata_file(dir_path):
    """
    Look for a CSV, TSV or JSON file that could contain metadata for a flat image directory.
    Returns a dictionary mapping image filename -> label.
    """
    metadata_extensions = ('.csv', '.tsv', '.json')
    for f in os.listdir(dir_path):
        if f.lower().endswith(metadata_extensions) and not f.startswith('.'):
            file_path = os.path.join(dir_path, f)
            try:
                if f.lower().endswith('.json'):
                    import json
                    with open(file_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                    # Support dictionary or list of dicts
                    if isinstance(data, dict):
                        return {str(k): str(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        mapping = {}
                        for item in data:
                            if isinstance(item, dict):
                                keys = list(item.keys())
                                # Try to find image and label key
                                file_key = next((k for k in keys if k.lower() in ['file', 'filename', 'image', 'path', 'id']), None)
                                label_key = next((k for k in keys if k.lower() in ['label', 'class', 'target', 'category']), None)
                                if file_key and label_key:
                                    mapping[str(item[file_key])] = str(item[label_key])
                        if mapping:
                            return mapping
                else:
                    # CSV or TSV
                    sep = '\t' if f.lower().endswith('.tsv') else ','
                    df = pd.read_csv(file_path, sep=sep)
                    file_col = next((c for c in df.columns if c.lower() in ['file', 'filename', 'image', 'path', 'id']), None)
                    label_col = next((c for c in df.columns if c.lower() in ['label', 'class', 'target', 'category']), None)
                    if file_col and label_col:
                        return dict(zip(df[file_col].astype(str), df[label_col].astype(str)))
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to parse metadata file {f}: {str(e)}\n")
    return {}

def convert_to_npz(dir_path, detected_type):
    """
    Scans the directory for image classification data and compiles it into an optimized NPZ.
    Returns the path to the compiled NPZ file.
    """
    print_progress(0.10, "Scanning files for standardization...")
    image_label_pairs = []
    
    if detected_type == 'class_hierarchy':
        for item in sorted(os.listdir(dir_path)):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                for f in sorted(os.listdir(item_path)):
                    if f.lower().endswith(IMAGE_EXTENSIONS):
                        image_label_pairs.append((os.path.join(item_path, f), item))
    elif detected_type == 'flat':
        metadata_map = parse_metadata_file(dir_path)
        for f in sorted(os.listdir(dir_path)):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(dir_path, f)
                # Look up label in metadata
                label = metadata_map.get(f, None)
                if not label:
                    # Try lookup without extension
                    name_no_ext = os.path.splitext(f)[0]
                    label = metadata_map.get(name_no_ext, "default")
                image_label_pairs.append((full_path, label))
    else:
        # Fallback recursive search
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, f)
                    parent = os.path.basename(os.path.dirname(full_path))
                    label = parent if parent and parent != os.path.basename(dir_path) else "default"
                    image_label_pairs.append((full_path, label))

    if not image_label_pairs:
        raise ValueError("No valid image files found in the directory.")

    print_progress(0.30, f"Loading and resizing {len(image_label_pairs)} images...")
    pixel_arrays = []
    labels = []
    
    # Cap total images to load/convert to stay within safe memory limits
    max_images = 1500
    if len(image_label_pairs) > max_images:
        np.random.seed(42)
        selected_indices = np.random.choice(len(image_label_pairs), max_images, replace=False)
        selected_pairs = [image_label_pairs[i] for i in sorted(selected_indices)]
    else:
        selected_pairs = image_label_pairs

    total_pairs = len(selected_pairs)
    for idx, (img_path, label) in enumerate(selected_pairs):
        if idx % max(1, total_pairs // 10) == 0:
            prog = 0.30 + (idx / total_pairs) * 0.50
            print_progress(prog, f"Processing image {idx + 1}/{total_pairs}...")
        try:
            with Image.open(img_path) as img:
                # Resize to standard 32x32 RGB as expected by pipelines.image
                img_rgb = img.convert("RGB").resize((32, 32))
                pixel_arrays.append(np.array(img_rgb))
                labels.append(label)
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to load image {img_path}: {str(e)}\n")
            continue

    if not pixel_arrays:
        raise ValueError("Failed to successfully load any images from the directory.")

    print_progress(0.85, "Saving standardized NPZ archive...")
    X = np.array(pixel_arrays)
    y = np.array(labels)
    
    cache_dir = get_cache_dir()
    npz_filename = f"ingested_{uuid.uuid4().hex[:10]}.npz"
    npz_path = os.path.join(cache_dir, npz_filename)
    
    # Save with keys compatible with loader.py / image.py
    np.savez(npz_path, X=X, y=y)
    
    print_progress(0.95, "Dataset standardization complete.")
    return npz_path

def ingest_dataset(file_path, dataset_type="auto"):
    """
    Main entry point for smart ingestion.
    Processes the input path, detects format, and standardizes it if needed.
    Returns: (resolved_path, resolved_dataset_type)
    """
    if not file_path or not os.path.exists(file_path):
        return file_path, dataset_type

    temp_dir = None
    working_path = file_path

    # If it is a zip archive, extract it first
    if file_path.endswith('.zip'):
        print_progress(0.05, "Extracting dataset archive...")
        cache_dir = get_cache_dir()
        temp_dir = os.path.join(cache_dir, f"extracted_{uuid.uuid4().hex[:10]}")
        os.makedirs(temp_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        working_path = temp_dir

    if os.path.isdir(working_path):
        detected = detect_dataset_format(working_path)
        
        if detected == 'yolo':
            # Keep as is, it's object detection
            return working_path, "object_detection"
        elif detected == 'segmentation':
            # Keep as is, but it's image segmentation
            return working_path, "image"
        elif detected in ['class_hierarchy', 'flat']:
            # Return the raw directory — image.py loads natively at 224×224
            # using the CNN extractor path, avoiding NPZ compilation overhead.
            return working_path, "image"
        else:
            # Unknown directory, default to image classification directory loading
            return working_path, "image"
    else:
        # It's a file (.csv, .npz, .parquet, etc.)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.npz':
            return file_path, "image"
        elif ext in ['.csv', '.tsv', '.parquet']:
            # Respect user chosen type if it's a valid pipeline type
            if dataset_type in ["nlp", "timeseries", "image", "tabular", "object_detection"]:
                resolved_type = dataset_type
            else:
                resolved_type = "tabular"
            return file_path, resolved_type

    return file_path, dataset_type
