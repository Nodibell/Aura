import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.dummy import DummyClassifier
from utils.helpers import print_progress, _export_model_and_code
from utils.loader import load_dataset
from utils.profiler import profile_dataset
from utils.charts import load_images_from_tabular, get_image_preview

def load_image_dataset_from_dir(dir_path):
    import os
    from PIL import Image
    import numpy as np
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    
    metadata_rows = []
    pixel_arrays = []
    labels = []
    
    all_image_paths = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(image_extensions):
                all_image_paths.append(os.path.join(root, f))
                
    all_image_paths.sort()
    
    max_images = 1500
    truncation_warning = None
    if len(all_image_paths) > max_images:
        truncation_warning = (
            f"Dataset contains {len(all_image_paths)} images. "
            f"Only {max_images} were randomly sampled for analysis to stay within memory limits."
        )
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_image_paths), max_images, replace=False)
        selected_paths = [all_image_paths[i] for i in sorted(selected_indices)]
    else:
        selected_paths = all_image_paths
        
    for idx, path in enumerate(selected_paths):
        try:
            with Image.open(path) as img:
                w, h = img.size
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
                arr = np.array(img)
                pixel_mean = float(arr.mean())
                pixel_std = float(arr.std())
                
                parent_dir = os.path.basename(os.path.dirname(path))
                if not parent_dir or parent_dir == os.path.basename(dir_path):
                    class_label = "default"
                else:
                    class_label = parent_dir
                    
                metadata_rows.append([
                    str(idx),
                    str(class_label),
                    str(w),
                    str(h),
                    str(channels),
                    f"{pixel_mean:.2f}",
                    f"{pixel_std:.2f}"
                ])
                
                img_rgb = img.convert("RGB").resize((32, 32))
                pixel_arrays.append(np.array(img_rgb))
                labels.append(class_label)
        except Exception:
            continue
            
    if len(pixel_arrays) == 0:
        raise ValueError("No valid images found in the directory or zip archive.")
        
    X = np.array(pixel_arrays)
    y = np.array(labels)
    return X, y, metadata_rows, truncation_warning

def analyze_image_segmentation(images_dir, masks_dir, file_path, model_export_path=None, code_export_path=None):
    from PIL import Image
    import numpy as np
    import os
    import io
    import base64
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    # Helper to convert to base64
    def to_base64_png(img_arr):
        if img_arr.max() <= 1.01:
            img_arr = img_arr * 255.0
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    print_progress(0.28, "Loading paired images and segmentation masks...")
    
    img_exts = ('.tif', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(img_exts)])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(img_exts)])
    
    if not img_files or not mask_files:
        raise ValueError("Could not find matching images and masks.")
        
    # Pair 1-to-1 by index
    num_pairs = min(len(img_files), len(mask_files))
    
    X_imgs = []
    y_masks = []
    metadata_rows = []
    
    # Standard size for segmentation analysis
    target_size = (128, 128)
    
    for idx in range(num_pairs):
        img_path = os.path.join(images_dir, img_files[idx])
        mask_path = os.path.join(masks_dir, mask_files[idx])
        
        try:
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                orig_w, orig_h = img.size
                
                # Resize original image
                img_resized = img.convert("RGB").resize(target_size)
                img_arr = np.array(img_resized)
                
                # Resize mask
                mask_resized = mask.convert("L").resize(target_size)
                mask_arr = np.array(mask_resized)
                # Convert mask to binary 0 or 1
                mask_binary = (mask_arr > 127).astype(np.uint8)
                
                X_imgs.append(img_arr)
                y_masks.append(mask_binary)
                
                pixel_mean = float(img_arr.mean())
                pixel_std = float(img_arr.std())
                vessel_density = float(mask_binary.mean())
                
                metadata_rows.append([
                    str(idx),
                    f"Vessel Density: {vessel_density:.2%}",
                    str(orig_w),
                    str(orig_h),
                    "3",
                    f"{pixel_mean:.2f}",
                    f"{pixel_std:.2f}"
                ])
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to load pair {idx}: {str(e)}\n")
            continue
            
    if not X_imgs:
        raise ValueError("No valid image-mask pairs loaded successfully.")
        
    X_imgs = np.array(X_imgs)
    y_masks = np.array(y_masks)
    N = len(X_imgs)
    
    # 2. Extract pixel-level features for training
    print_progress(0.45, "Extracting pixel-level features...")
    
    # Features matrix X_pixels, labels y_pixels
    # We sample 500 pixels per image
    pixels_per_img = 500
    X_feats = []
    y_labels = []
    
    for idx in range(N):
        img = X_imgs[idx] / 255.0  # normalize
        mask = y_masks[idx]
        
        # Sample coordinates
        y_coords = np.random.randint(2, target_size[1] - 2, size=pixels_per_img)
        x_coords = np.random.randint(2, target_size[0] - 2, size=pixels_per_img)
        
        for y, x in zip(y_coords, x_coords):
            # RGB color values of center pixel
            rgb = img[y, x]
            
            # Local 3x3 patch average
            patch = img[y-1:y+2, x-1:x+2]
            patch_mean = patch.mean(axis=(0, 1))
            patch_std = patch.std(axis=(0, 1))
            
            # Position normalized
            pos = [y / target_size[1], x / target_size[0]]
            
            feat = np.concatenate([rgb, patch_mean, patch_std, pos])
            X_feats.append(feat)
            y_labels.append(mask[y, x])
            
    X_feats = np.array(X_feats)
    y_labels = np.array(y_labels)
    
    # Split train/test (by image index to avoid data leakage)
    print_progress(0.60, "Training pixel classifier for vessel segmentation...")
    split_idx = int(N * 0.8)
    if split_idx < 1:
        split_idx = 1
        
    train_pixels_mask = np.repeat(np.arange(N) < split_idx, pixels_per_img)
    X_train, X_test = X_feats[train_pixels_mask], X_feats[~train_pixels_mask]
    y_train, y_test = y_labels[train_pixels_mask], y_labels[~train_pixels_mask]
    
    # Fit Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=2)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    dice = float(f1_score(y_test, y_pred, zero_division=0))
    
    # Intersection over Union (IoU)
    intersection = np.logical_and(y_test, y_pred).sum()
    union = np.logical_or(y_test, y_pred).sum()
    iou = float(intersection / union) if union > 0 else 0.0
    
    # 3. Create comparison grids for sample images (max 4 images)
    print_progress(0.80, "Generating segmentation overlay grids...")
    overlay_images = []
    test_indices = list(range(split_idx, N))[:4]
    if not test_indices:
        test_indices = list(range(N))[:4]
        
    for t_idx in test_indices:
        img = X_imgs[t_idx]
        mask = y_masks[t_idx]
        
        # Predict mask for the entire image
        # Extract features for all pixels in this image
        img_norm = img / 255.0
        all_feats = []
        for y in range(target_size[1]):
            for x in range(target_size[0]):
                # Safe padding
                py = max(1, min(target_size[1] - 2, y))
                px = max(1, min(target_size[0] - 2, x))
                
                rgb = img_norm[py, px]
                patch = img_norm[py-1:py+2, px-1:px+2]
                patch_mean = patch.mean(axis=(0, 1))
                patch_std = patch.std(axis=(0, 1))
                pos = [py / target_size[1], px / target_size[0]]
                
                all_feats.append(np.concatenate([rgb, patch_mean, patch_std, pos]))
                
        pred_mask = rf.predict(np.array(all_feats)).reshape(target_size)
        
        # Create side-by-side comparison: [Original, Ground Truth, Prediction Overlay]
        # Prediction Overlay: original image with red highlights where pred_mask is 1
        overlay = img.copy()
        overlay[pred_mask == 1] = [255, 0, 0]  # Color vessel predictions red
        
        gt_visual = np.stack([mask*255, mask*255, mask*255], axis=-1)
        
        # Concat horizontally
        comparison = np.concatenate([img, gt_visual, overlay], axis=1)
        b64_str = to_base64_png(comparison)
        
        overlay_images.append({
            "label": f"Image {t_idx} (Left: Original | Middle: Ground Truth | Right: Overlay)",
            "base64": b64_str
        })
        
    charts = [
        {
            "type": "image_grid",
            "title": "Segmented Retinal Vessel Predictions (Red Overlay)",
            "x_label": "",
            "y_label": "",
            "data": [],
            "images": overlay_images
        }
    ]
    
    # Class distribution (Vessel vs Background pixels in test set)
    bg_count = float(np.sum(y_test == 0))
    fg_count = float(np.sum(y_test == 1))
    charts.append({
        "type": "bar",
        "title": "Pixel Class Distribution (Vessels vs Background)",
        "x_label": "Pixel Class",
        "y_label": "Pixel Count",
        "data": [
            {"x_val": "Background", "y": bg_count},
            {"x_val": "Vessel", "y": fg_count}
        ]
    })
    
    summary = (
        f"### 👁️ Semantic Image Segmentation Overview\n"
        f"- **Image-Mask Pairs loaded:** {N}\n"
        f"- **Resolved Resolution:** {target_size[0]}x{target_size[1]} pixels\n"
        f"- **Vessel Pixel Density:** {float(y_masks.mean()):.2%}\n\n"
        f"### 🤖 Pixel Classifier Performance (Random Forest)\n"
        f"- **Mean Intersection over Union (IoU):** `{iou:.4f}`\n"
        f"- **Dice Coefficient (F1-Score):** `{dice:.4f}`\n"
        f"- **Pixel Accuracy:** `{accuracy:.4f}`\n"
    )
    
    columns = ["image_index", "class_label", "width", "height", "channels", "mean_intensity", "std_intensity"]
    preview_rows = metadata_rows[:500]
    full_preview = {
        "columns": columns,
        "rows": preview_rows,
        "total_rows": N
    }
    
    models_compared = [
        {"name": "Dummy Baseline", "score": float(1.0 - y_test.mean()), "metric": "Pixel Accuracy"},
        {"name": "Random Forest Pixel Classifier", "score": float(accuracy), "metric": "Pixel Accuracy"}
    ]
    
    # Export code
    if model_export_path or code_export_path:
        import joblib
        if model_export_path:
            # compress=3 reduces peak RAM during serialization (prevents -9 SIGKILL on Apple Silicon)
            joblib.dump(rf, model_export_path, compress=3)
        if code_export_path:
            with open(code_export_path, "w") as f:
                f.write("# Segmentation Model Reproduction Script\n")
                f.write("import joblib\n")
                f.write("model = joblib.load('model.joblib')\n")
                f.write("print('Loaded Random Forest Pixel Classifier successfully!')\n")

    return {
        "summary": summary,
        "columns": columns,
        "row_count": N,
        "col_count": len(columns),
        "task_type": "segmentation",
        "numeric_col_count": 5,
        "categorical_col_count": 2,
        "text_col_count": 0,
        "missing_values": {c: 0 for c in columns},
        "correlations": [],
        "charts": charts,
        "metrics": {
            "model": "Random Forest Pixel Classifier",
            "score_type": "Mean IoU",
            "score": iou,
            "additional_metrics": {"Dice Coefficient": dice, "Pixel Accuracy": accuracy}
        },
        "models_compared": models_compared,
        "target_column": "class_label",
        "dummy_baseline_score": float(1.0 - y_test.mean()),
        "cv_scores": [accuracy],
        "cv_mean": accuracy,
        "cv_std": 0.0,
        "confusion_matrix": None,
        "profiling": {"duplicate_rows": 0, "columns": {}},
        "full_preview": full_preview,
        "test_row_count": None,
        "test_col_count": None,
        "test_missing_values": None,
        "test_correlations": None,
        "test_profiling": None,
        "test_full_preview": None,
        "error": None
    }

def analyze_image(file_path, task_type_override="auto", target_col=None, test_file_path=None, model_export_path=None, code_export_path=None):
    import io
    import base64
    import zipfile
    import tempfile
    import os
    import shutil
    from PIL import Image
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    # Helper for base64 conversion
    def to_base64_png(img_arr):
        if img_arr.max() <= 1.01:
            img_arr = img_arr * 255.0
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        
        if len(img_arr.shape) == 2:
            img = Image.fromarray(img_arr, mode='L')
        elif img_arr.shape[2] == 1:
            img = Image.fromarray(img_arr[:, :, 0], mode='L')
        elif img_arr.shape[2] == 3:
            img = Image.fromarray(img_arr, mode='RGB')
        elif img_arr.shape[2] == 4:
            img = Image.fromarray(img_arr, mode='RGBA')
        else:
            img = Image.fromarray(img_arr[:, :, :3], mode='RGB')
            
        if img.width < 128 or img.height < 128:
            img = img.resize((128, 128), resample=Image.NEAREST)
            
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
 
    temp_dir = None
    truncation_warning = None  # Set if images are silently subsampled
    try:
        if file_path.endswith((".csv", ".tsv", ".parquet")):
            print_progress(0.25, "Loading tabular dataset for image analysis...")
            df = load_dataset(file_path)
            X_images, y_arr = load_images_from_tabular(df, target_col)
            N = len(X_images)
            H, W, C = X_images.shape[1], X_images.shape[2], X_images.shape[3]
            
            # Subsample if too large
            max_images = 1500
            if N > max_images:
                truncation_warning = (
                    f"Dataset contains {N} images. "
                    f"Only {max_images} were randomly sampled for analysis to stay within memory limits."
                )
                np.random.seed(42)
                selected_indices = np.random.choice(N, max_images, replace=False)
                X_images = X_images[selected_indices]
                y_arr = y_arr[selected_indices]
                N = max_images
                
            metadata_rows = []
            for idx in range(N):
                img_slice = X_images[idx]
                pixel_min = float(img_slice.min())
                pixel_max = float(img_slice.max())
                pixel_mean = float(img_slice.mean())
                pixel_std = float(img_slice.std())
                label = str(y_arr[idx])
                metadata_rows.append([
                    str(idx),
                    label,
                    str(W),
                    str(H),
                    str(C),
                    f"{pixel_mean:.2f}",
                    f"{pixel_std:.2f}"
                ])
                
        elif file_path.endswith(".npz"):
            print_progress(0.25, "Loading NPZ image archive...")
            npz = np.load(file_path, allow_pickle=True)
            keys = list(npz.keys())
            
            x_keys = [k for k in keys if k.lower() in ["x", "x_train", "train_x", "data", "features", "images"]]
            y_keys = [k for k in keys if k.lower() in ["y", "y_train", "train_y", "labels", "target", "classes"]]
            
            if x_keys:
                X_arr = npz[x_keys[0]]
            else:
                sorted_keys = sorted(keys, key=lambda k: len(npz[k].shape), reverse=True)
                if sorted_keys:
                    X_arr = npz[sorted_keys[0]]
                else:
                    raise ValueError("No arrays found in NPZ.")
                    
            if y_keys:
                y_arr = npz[y_keys[0]]
            else:
                sorted_keys = sorted(keys, key=lambda k: len(npz[k].shape), reverse=True)
                if len(sorted_keys) >= 2:
                    y_arr = npz[sorted_keys[1]]
                else:
                    y_arr = np.zeros(len(X_arr), dtype=int)
            
            orig_shape = X_arr.shape
            N = orig_shape[0]
            
            if len(orig_shape) == 4:
                H, W, C = orig_shape[1], orig_shape[2], orig_shape[3]
                X_images = X_arr
            elif len(orig_shape) == 3:
                H, W = orig_shape[1], orig_shape[2]
                C = 1
                X_images = X_arr.reshape((N, H, W, 1))
            elif len(orig_shape) == 2:
                total_pixels = orig_shape[1]
                if total_pixels % 3 == 0:
                    pixels_per_channel = total_pixels // 3
                    side = int(np.sqrt(pixels_per_channel))
                    if side * side == pixels_per_channel:
                        H, W, C = side, side, 3
                        X_images = X_arr.reshape((N, H, W, 3))
                    else:
                        side = int(np.sqrt(total_pixels))
                        H, W, C = side, side, 1
                        X_images = X_arr.reshape((N, H, W, 1))
                else:
                    side = int(np.sqrt(total_pixels))
                    H, W, C = side, side, 1
                    X_images = X_arr.reshape((N, H, W, 1))
            else:
                raise ValueError(f"Unsupported NPZ shape: {orig_shape}")
                
            max_images = 1500
            if N > max_images:
                np.random.seed(42)
                selected_indices = np.random.choice(N, max_images, replace=False)
                X_images = X_images[selected_indices]
                y_arr = y_arr[selected_indices]
                N = max_images
                
            metadata_rows = []
            for idx in range(N):
                img_slice = X_images[idx]
                pixel_min = float(img_slice.min())
                pixel_max = float(img_slice.max())
                pixel_mean = float(img_slice.mean())
                pixel_std = float(img_slice.std())
                label = str(y_arr[idx])
                metadata_rows.append([
                    str(idx),
                    label,
                    str(W),
                    str(H),
                    str(C),
                    f"{pixel_mean:.2f}",
                    f"{pixel_std:.2f}"
                ])
                
        else:
            working_path = file_path
            if file_path.endswith(".zip"):
                print_progress(0.20, "Extracting zip image archive...")
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                working_path = temp_dir
                
            # Auto-detect if this is a semantic segmentation dataset
            images_dir = None
            masks_dir = None
            if task_type_override in ["auto", "segmentation"]:
                pairs = []
                for root, dirs, files in os.walk(working_path):
                    img_dirs = [d for d in dirs if d.lower() in ["images", "image", "img"]]
                    mask_dirs = [d for d in dirs if d.lower() in ["masks", "mask", "1st_manual", "labels", "2nd_manual"]]
                    if img_dirs and mask_dirs:
                        pairs.append((os.path.join(root, img_dirs[0]), os.path.join(root, mask_dirs[0])))
                
                if pairs:
                    train_pairs = [p for p in pairs if "train" in p[0].lower() or "training" in p[0].lower()]
                    if train_pairs:
                        images_dir, masks_dir = train_pairs[0]
                    else:
                        images_dir, masks_dir = pairs[0]
            
            if images_dir and masks_dir:
                return analyze_image_segmentation(images_dir, masks_dir, file_path, model_export_path, code_export_path)
                
            print_progress(0.25, "Loading image dataset from directory...")
            X_images, y_arr, metadata_rows, dir_warning = load_image_dataset_from_dir(working_path)
            if dir_warning:
                truncation_warning = dir_warning
            N = len(X_images)
            H, W, C = X_images.shape[1], X_images.shape[2], X_images.shape[3]
            
        # Load separate test set if provided
        X_test_images = None
        y_test_arr = None
        test_metadata_rows = []
        if test_file_path and os.path.exists(test_file_path):
            try:
                print_progress(0.28, "Loading separate test images...")
                if test_file_path.endswith((".csv", ".tsv", ".parquet")):
                    test_df = load_dataset(test_file_path)
                    X_test_images, y_test_arr = load_images_from_tabular(test_df, target_col)
                elif test_file_path.endswith(".npz"):
                    test_npz = np.load(test_file_path, allow_pickle=True)
                    t_keys = list(test_npz.keys())
                    t_x_keys = [k for k in t_keys if k.lower() in ["x", "x_test", "test_x", "data", "features", "images"]]
                    t_y_keys = [k for k in t_keys if k.lower() in ["y", "y_test", "test_y", "labels", "target", "classes"]]
                    
                    if t_x_keys:
                        X_test_raw = test_npz[t_x_keys[0]]
                    else:
                        X_test_raw = test_npz[sorted(t_keys, key=lambda k: len(test_npz[k].shape), reverse=True)[0]]
                    if t_y_keys:
                        y_test_arr = test_npz[t_y_keys[0]]
                    else:
                        y_test_arr = np.zeros(len(X_test_raw), dtype=int)
                        
                    t_shape = X_test_raw.shape
                    if len(t_shape) == 4:
                        X_test_images = X_test_raw
                    elif len(t_shape) == 3:
                        X_test_images = X_test_raw.reshape((t_shape[0], t_shape[1], t_shape[2], 1))
                    elif len(t_shape) == 2:
                        X_test_images = X_test_raw.reshape((t_shape[0], H, W, C))
                else:
                    t_working_path = test_file_path
                    if test_file_path.endswith(".zip"):
                        t_temp_dir = tempfile.mkdtemp()
                        with zipfile.ZipFile(test_file_path, 'r') as zip_ref:
                            zip_ref.extractall(t_temp_dir)
                        t_working_path = t_temp_dir
                    X_test_images, y_test_arr, _ = load_image_dataset_from_dir(t_working_path)
                
                if X_test_images is not None:
                    if len(X_test_images) > 1500:
                        np.random.seed(42)
                        sel = np.random.choice(len(X_test_images), 1500, replace=False)
                        X_test_images = X_test_images[sel]
                        y_test_arr = y_test_arr[sel]
                    
                    for idx in range(len(X_test_images)):
                        img_slice = X_test_images[idx]
                        p_min = float(img_slice.min())
                        p_max = float(img_slice.max())
                        p_mean = float(img_slice.mean())
                        p_std = float(img_slice.std())
                        lbl = str(y_test_arr[idx])
                        test_metadata_rows.append([
                            str(idx),
                            lbl,
                            str(img_slice.shape[2] if len(img_slice.shape) > 2 else img_slice.shape[1]),
                            str(img_slice.shape[1] if len(img_slice.shape) > 2 else img_slice.shape[0]),
                            str(img_slice.shape[3] if len(img_slice.shape) > 3 else (img_slice.shape[2] if len(img_slice.shape) > 2 else 1)),
                            f"{p_mean:.2f}",
                            f"{p_std:.2f}"
                        ])
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to load separate test images: {str(e)}\n")
                X_test_images = None

        print_progress(0.40, "Profiling image shapes and intensity stats...")
        columns = ["image_index", "class_label", "width", "height", "channels", "mean_intensity", "std_intensity"]
        
        unique_classes = sorted(list(set(y_arr)))
        num_classes = len(unique_classes)
        
        print_progress(0.50, "Generating sample grid...")
        sample_indices = []
        class_to_indices = {c: np.where(y_arr == c)[0] for c in unique_classes}
        idx_pointers = {c: 0 for c in unique_classes}
        target_sample_count = min(24, max(num_classes, 12))
        target_sample_count = min(target_sample_count, N)
        while len(sample_indices) < target_sample_count:
            added = False
            for c in unique_classes:
                ptr = idx_pointers[c]
                if ptr < len(class_to_indices[c]):
                    sample_indices.append(int(class_to_indices[c][ptr]))
                    idx_pointers[c] += 1
                    added = True
                if len(sample_indices) >= target_sample_count:
                    break
            if not added:
                break
                
        sample_images = []
        for s_idx in sample_indices:
            label = str(y_arr[s_idx])
            b64_str = to_base64_png(X_images[s_idx])
            sample_images.append({
                "label": f"Sample (Class: {label})",
                "base64": b64_str
            })
            
        print_progress(0.60, "Generating class average images...")
        mean_images = []
        for c in unique_classes[:12]:
            c_indices = class_to_indices[c]
            c_mean = np.mean(X_images[c_indices], axis=0)
            b64_str = to_base64_png(c_mean)
            mean_images.append({
                "label": f"Class: {c} (Mean)",
                "base64": b64_str
            })
            
        class_counts = {}
        for c in y_arr:
            class_counts[c] = class_counts.get(c, 0) + 1
        class_dist_data = [{"x_val": str(c), "y": float(class_counts[c])} for c in unique_classes]
        
        all_intensities = X_images.flatten()
        if all_intensities.max() <= 1.01:
            all_intensities = all_intensities * 255.0
        hist_counts, bin_edges = np.histogram(all_intensities, bins=10, range=(0, 255))
        pct_counts = (hist_counts / len(all_intensities)) * 100
        
        hist_data = []
        for i in range(10):
            bin_label = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
            hist_data.append({"x_val": bin_label, "y": float(pct_counts[i])})
            
        charts = [
            {
                "type": "image_grid",
                "title": "Representative Class Mean Images",
                "x_label": "",
                "y_label": "",
                "data": [],
                "images": mean_images
            },
            {
                "type": "image_grid",
                "title": "Sample Image Grid (Subset)",
                "x_label": "",
                "y_label": "",
                "data": [],
                "images": sample_images
            },
            {
                "type": "bar",
                "title": "Class Distribution",
                "x_label": "Class Label",
                "y_label": "Number of Samples",
                "data": class_dist_data
            },
            {
                "type": "bar",
                "title": "Pixel Intensity Distribution (%)",
                "x_label": "Intensity Range",
                "y_label": "Percentage of Pixels",
                "data": hist_data
            }
        ]
        
        print_progress(0.70, "Training fast classifier...")
        X_flat = X_images.reshape((N, int(np.prod(X_images.shape[1:]))))
        
        label_to_code = {label: i for i, label in enumerate(unique_classes)}
        y_encoded = np.array([label_to_code[lbl] for lbl in y_arr])
        
        if N >= 10 and num_classes > 1:
            min_class_size = min(class_counts.values())
            n_splits = min(5, min_class_size)
            if n_splits >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                n_features = X_flat.shape[1]
                if n_features > 100:
                    from sklearn.pipeline import Pipeline
                    from pipelines.cv_nlp_engine import PCA
                    n_comps = min(100, N, n_features)
                    model = Pipeline([
                        ('pca', PCA(n_components=n_comps, random_state=42)),
                        ('lr', LogisticRegression(max_iter=500, random_state=42, solver='lbfgs'))
                    ])
                else:
                    model = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
                
                dummy = DummyClassifier(strategy="most_frequent")
                dummy.fit(X_flat, y_encoded)
                dummy_score = accuracy_score(y_encoded, dummy.predict(X_flat))
                
                cv_scores = cross_val_score(model, X_flat, y_encoded, cv=cv, scoring='accuracy')
                cv_scores_list = [float(s) for s in cv_scores]
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
                
                if X_test_images is not None:
                    X_test_flat = X_test_images.reshape((len(X_test_images), int(np.prod(X_test_images.shape[1:]))))
                    y_test = np.array([label_to_code[lbl] if lbl in label_to_code else 0 for lbl in y_test_arr])
                    model.fit(X_flat, y_encoded)
                    y_pred = model.predict(X_test_flat)
                else:
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                overall_accuracy = float(accuracy_score(y_test, y_pred))
                overall_f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                overall_precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                overall_recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                
                raw_cm = confusion_matrix(y_test, y_pred)
                cm_data = {
                    "labels": [str(c) for c in unique_classes],
                    "values": [[int(val) for val in row] for row in raw_cm]
                }
            else:
                n_features = X_flat.shape[1]
                if n_features > 100:
                    from sklearn.pipeline import Pipeline
                    from pipelines.cv_nlp_engine import PCA
                    n_comps = min(100, N, n_features)
                    model = Pipeline([
                        ('pca', PCA(n_components=n_comps, random_state=42)),
                        ('lr', LogisticRegression(max_iter=500, random_state=42, solver='lbfgs'))
                    ])
                else:
                    model = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
                model.fit(X_flat, y_encoded)
                if X_test_images is not None:
                    X_test_flat = X_test_images.reshape((len(X_test_images), int(np.prod(X_test_images.shape[1:]))))
                    y_test = np.array([label_to_code[lbl] if lbl in label_to_code else 0 for lbl in y_test_arr])
                    y_pred = model.predict(X_test_flat)
                    overall_accuracy = float(accuracy_score(y_test, y_pred))
                    overall_f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    overall_precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                    overall_recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                    raw_cm = confusion_matrix(y_test, y_pred)
                else:
                    y_pred = model.predict(X_flat)
                    overall_accuracy = float(accuracy_score(y_encoded, y_pred))
                    overall_f1 = float(f1_score(y_encoded, y_pred, average='weighted', zero_division=0))
                    overall_precision = float(precision_score(y_encoded, y_pred, average='weighted', zero_division=0))
                    overall_recall = float(recall_score(y_encoded, y_pred, average='weighted', zero_division=0))
                    raw_cm = confusion_matrix(y_encoded, y_pred)
                
                dummy_score = 1.0 / num_classes
                cv_scores_list = [overall_accuracy]
                cv_mean = overall_accuracy
                cv_std = 0.0
                
                cm_data = {
                    "labels": [str(c) for c in unique_classes],
                    "values": [[int(val) for val in row] for row in raw_cm]
                }
            
            # Compute actual dummy baseline metrics
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy="most_frequent")
            if cv_is_possible and X_test_images is None:
                # train/test split was used
                dummy.fit(X_train, y_train)
                dummy_preds = dummy.predict(X_test)
                dummy_acc = float(accuracy_score(y_test, dummy_preds))
                dummy_f1 = float(f1_score(y_test, dummy_preds, average='weighted', zero_division=0))
                dummy_prec = float(precision_score(y_test, dummy_preds, average='weighted', zero_division=0))
                dummy_rec = float(recall_score(y_test, dummy_preds, average='weighted', zero_division=0))
            elif X_test_images is not None:
                dummy.fit(X_flat, y_encoded)
                dummy_preds = dummy.predict(X_test_flat)
                dummy_acc = float(accuracy_score(y_test, dummy_preds))
                dummy_f1 = float(f1_score(y_test, dummy_preds, average='weighted', zero_division=0))
                dummy_prec = float(precision_score(y_test, dummy_preds, average='weighted', zero_division=0))
                dummy_rec = float(recall_score(y_test, dummy_preds, average='weighted', zero_division=0))
            else:
                dummy.fit(X_flat, y_encoded)
                dummy_preds = dummy.predict(X_flat)
                dummy_acc = float(accuracy_score(y_encoded, dummy_preds))
                dummy_f1 = float(f1_score(y_encoded, dummy_preds, average='weighted', zero_division=0))
                dummy_prec = float(precision_score(y_encoded, dummy_preds, average='weighted', zero_division=0))
                dummy_rec = float(recall_score(y_encoded, dummy_preds, average='weighted', zero_division=0))
            dummy_score = dummy_acc
        else:
            overall_accuracy = 1.0
            overall_f1 = 1.0
            overall_precision = 1.0
            overall_recall = 1.0
            dummy_acc = 1.0
            dummy_f1 = 1.0
            dummy_prec = 1.0
            dummy_rec = 1.0
            dummy_score = 1.0
            cv_scores_list = [1.0]
            cv_mean = 1.0
            cv_std = 0.0
            cm_data = {
                "labels": [str(c) for c in unique_classes],
                "values": [[N]]
            }
            
        print_progress(0.90, "Assembling results...")
        summary_text = (
            f"Analyzed image dataset with {N} samples across {num_classes} unique classes. "
            f"Image dimensions are {W}x{H} pixels with {C} color channel(s). "
            f"Fitted a Logistic Regression classifier which achieved {overall_accuracy*100:.1f}% accuracy "
            f"compared to a dummy baseline of {dummy_score*100:.1f}%."
        )
        
        # Compute profiling using standard profile_dataset on metadata DataFrame
        import pandas as pd
        meta_df = pd.DataFrame(metadata_rows, columns=columns)
        for c in ["width", "height", "channels", "mean_intensity", "std_intensity"]:
            meta_df[c] = pd.to_numeric(meta_df[c], errors='coerce')
        profiling = profile_dataset(meta_df)
        
        preview_rows = metadata_rows[:500]
        full_preview = {
            "columns": columns,
            "rows": preview_rows,
            "total_rows": N
        }
        
        # Compute test set profiling & preview if loaded
        test_row_count = None
        test_col_count = None
        test_missing = None
        test_correlations = None
        test_profiling = None
        test_full_preview = None
        if X_test_images is not None:
            t_meta_df = pd.DataFrame(test_metadata_rows, columns=columns)
            for c in ["width", "height", "channels", "mean_intensity", "std_intensity"]:
                t_meta_df[c] = pd.to_numeric(t_meta_df[c], errors='coerce')
            test_profiling = profile_dataset(t_meta_df)
            test_row_count = len(X_test_images)
            test_col_count = len(columns)
            test_missing = {c: 0 for c in columns}
            test_correlations = []
            test_full_preview = {
                "columns": columns,
                "rows": test_metadata_rows[:500],
                "total_rows": len(X_test_images)
            }
        
        models_compared = [
            {"name": "Dummy Baseline", "score": float(dummy_acc), "metric": "Accuracy", "f1": float(dummy_f1), "precision": float(dummy_prec), "recall": float(dummy_rec)},
            {"name": "Logistic Regression", "score": float(overall_accuracy), "metric": "Accuracy", "f1": float(overall_f1), "precision": float(overall_precision), "recall": float(overall_recall)}
        ]
        
        # Phase 1: Model & Code Export
        if model_export_path or code_export_path:
            feats = [f"pixel_{i}" for i in range(X_flat.shape[1])] if 'X_flat' in locals() else []
            _export_model_and_code(
                model, model_export_path, code_export_path,
                file_path, "image", "class_label", None,
                "classification", feats, "Logistic Regression", None, None, None
            )
            
        return {
            "summary": summary_text,
            "columns": columns,
            "row_count": N,
            "col_count": len(columns),
            "task_type": "classification",
            "numeric_col_count": 5,
            "categorical_col_count": 2,
            "text_col_count": 0,
            "missing_values": {c: 0 for c in columns},
            "correlations": [],
            "charts": charts,
            "metrics": {
                "model": "Logistic Regression",
                "score_type": "Accuracy",
                "score": overall_accuracy,
                "additional_metrics": {"F1-Score": overall_f1}
            },
            "models_compared": models_compared,
            "target_column": "class_label",
            "dummy_baseline_score": dummy_score,
            "cv_scores": cv_scores_list,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "confusion_matrix": cm_data,
            "profiling": profiling,
            "full_preview": full_preview,
            "test_row_count": test_row_count,
            "test_col_count": test_col_count,
            "test_missing_values": test_missing,
            "test_correlations": test_correlations,
            "test_profiling": test_profiling,
            "test_full_preview": test_full_preview,
            "warning": truncation_warning,
            "error": None
        }
    except Exception as e:
        import traceback
        return {"error": f"An error occurred during Image dataset execution: {str(e)}\n{traceback.format_exc()}"}
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
