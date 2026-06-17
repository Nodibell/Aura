import json
import numpy as np
import pandas as pd
import base64
from io import BytesIO

def load_images_from_tabular(df, target_col=None):
    import numpy as np
    import pandas as pd
    
    # Auto-detect target column if not provided
    y_arr = None
    if target_col and target_col in df.columns:
        y_arr = df[target_col].to_numpy()
        X_df = df.drop(columns=[target_col])
    else:
        # Search for common label column names
        label_cols = [c for c in df.columns if c.lower() in ["label", "class", "target", "y", "digit", "price"]]
        if label_cols:
            y_arr = df[label_cols[0]].to_numpy()
            X_df = df.drop(columns=[label_cols[0]])
        else:
            # Assume first column is target if named something suspicious, or last column
            y_arr = df.iloc[:, 0].to_numpy()
            X_df = df.iloc[:, 1:]
            
    X_arr = X_df.to_numpy()
    N, P = X_arr.shape
    
    # Try to find a square resolution
    # Check grayscale first
    side = int(np.sqrt(P))
    if side * side == P:
        H, W, C = side, side, 1
        X_images = X_arr.reshape((N, H, W, 1))
    else:
        # Check RGB (3 channels)
        if P % 3 == 0:
            pixels_per_channel = P // 3
            side = int(np.sqrt(pixels_per_channel))
            if side * side == pixels_per_channel:
                H, W, C = side, side, 3
                X_images = X_arr.reshape((N, H, W, 3))
            else:
                raise ValueError(f"Tabular features count {P} cannot be reshaped into a square image.")
        else:
            raise ValueError(f"Tabular features count {P} cannot be reshaped into a square image.")
            
    # Normalize pixel values if they are 0-255
    if X_images.max() > 1.01:
        X_images = X_images / 255.0
        
    return X_images, y_arr

def get_image_preview(file_path, nrows=15):
    # This reads up to nrows images and returns metadata
    import zipfile
    import tempfile
    import os
    import shutil
    from PIL import Image
    import numpy as np
    
    columns = ["image_index", "class_label", "width", "height", "channels", "mean_intensity", "std_intensity"]
    preview_rows = []
    total_rows = 0
    
    temp_dir = None
    try:
        if file_path.endswith((".csv", ".tsv", ".parquet")):
            try:
                df_full = load_dataset(file_path)
                total_rows = len(df_full)
            except Exception:
                total_rows = nrows
            df = load_dataset(file_path, nrows=nrows)
            X_images, y_arr = load_images_from_tabular(df)
            for idx in range(len(X_images)):
                img_slice = X_images[idx]
                pixel_mean = float(img_slice.mean())
                pixel_std = float(img_slice.std())
                preview_rows.append([
                    str(idx),
                    str(y_arr[idx]),
                    str(X_images.shape[2]),
                    str(X_images.shape[1]),
                    str(X_images.shape[3]),
                    f"{pixel_mean:.2f}",
                    f"{pixel_std:.2f}"
                ])
            return {
                "columns": columns,
                "preview_rows": preview_rows,
                "local_path": file_path,
                "inferred_dataset_type": "image",
                "total_rows": total_rows,
                "error": None
            }

        if file_path.endswith(".npz"):
            npz = np.load(file_path, allow_pickle=True)
            keys = list(npz.keys())
            
            x_keys = [k for k in keys if k.lower() in ["x", "x_train", "train_x", "data", "features", "images"]]
            y_keys = [k for k in keys if k.lower() in ["y", "y_train", "train_y", "labels", "target", "classes"]]
            
            if x_keys:
                X_arr = npz[x_keys[0]]
            else:
                sorted_keys = sorted(keys, key=lambda k: len(npz[k].shape), reverse=True)
                X_arr = npz[sorted_keys[0]] if sorted_keys else None
                
            if y_keys:
                y_arr = npz[y_keys[0]]
            else:
                sorted_keys = sorted(keys, key=lambda k: len(npz[k].shape), reverse=True)
                if len(sorted_keys) >= 2:
                    y_arr = npz[sorted_keys[1]]
                else:
                    y_arr = np.zeros(len(X_arr), dtype=int)
                    
            if X_arr is None:
                raise ValueError("Could not find suitable data array in .npz archive.")
                
            orig_shape = X_arr.shape
            total_rows = int(orig_shape[0])
            N = min(nrows, orig_shape[0])
            
            # Deduce shape details
            if len(orig_shape) == 4:
                H, W, C = orig_shape[1], orig_shape[2], orig_shape[3]
                X_sub = X_arr[:N]
            elif len(orig_shape) == 3:
                H, W = orig_shape[1], orig_shape[2]
                C = 1
                X_sub = X_arr[:N].reshape((N, H, W, 1))
            elif len(orig_shape) == 2:
                total_pixels = orig_shape[1]
                if total_pixels % 3 == 0:
                    pixels_per_channel = total_pixels // 3
                    side = int(np.sqrt(pixels_per_channel))
                    if side * side == pixels_per_channel:
                        H, W, C = side, side, 3
                        X_sub = X_arr[:N].reshape((N, H, W, 3))
                    else:
                        side = int(np.sqrt(total_pixels))
                        H, W, C = side, side, 1
                        X_sub = X_arr[:N].reshape((N, H, W, 1))
                else:
                    side = int(np.sqrt(total_pixels))
                    H, W, C = side, side, 1
                    X_sub = X_arr[:N].reshape((N, H, W, 1))
            else:
                raise ValueError(f"Unsupported NPZ shape: {orig_shape}")
                
            for idx in range(N):
                img_slice = X_sub[idx]
                pixel_mean = float(img_slice.mean())
                pixel_std = float(img_slice.std())
                label = str(y_arr[idx])
                preview_rows.append([
                    str(idx),
                    label,
                    str(W),
                    str(H),
                    str(C),
                    f"{pixel_mean:.2f}",
                    f"{pixel_std:.2f}"
                ])
        else:
            # zip or folder
            working_path = file_path
            if file_path.endswith(".zip"):
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
                    img_members = [m for m in zip_ref.namelist() if m.lower().endswith(image_extensions)]
                    img_members.sort()
                    for m in img_members[:nrows]:
                        zip_ref.extract(m, temp_dir)
                working_path = temp_dir
                
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
            all_image_paths = []
            for root, dirs, files in os.walk(working_path):
                for f in files:
                    if f.lower().endswith(image_extensions):
                        all_image_paths.append(os.path.join(root, f))
            all_image_paths.sort()
            total_rows = len(all_image_paths)
            
            for idx, path in enumerate(all_image_paths[:nrows]):
                with Image.open(path) as img:
                    w, h = img.size
                    channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
                    arr = np.array(img)
                    pixel_mean = float(arr.mean())
                    pixel_std = float(arr.std())
                    parent_dir = os.path.basename(os.path.dirname(path))
                    if not parent_dir or parent_dir == os.path.basename(working_path):
                        class_label = "default"
                    else:
                        class_label = parent_dir
                    preview_rows.append([
                        str(idx),
                        str(class_label),
                        str(w),
                        str(h),
                        str(channels),
                        f"{pixel_mean:.2f}",
                        f"{pixel_std:.2f}"
                    ])
                    
        return {
            "columns": columns,
            "preview_rows": preview_rows,
            "local_path": file_path,
            "inferred_dataset_type": "image",
            "total_rows": total_rows,
            "error": None
        }
    except Exception as e:
        import traceback
        return {"error": f"Failed to generate image preview: {str(e)}\n{traceback.format_exc()}"}
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def generate_boxplots(df, numeric_cols, target_col=None):
    boxplots = []
    scored_boxplots = []
    
    for col in numeric_cols:
        if target_col and col == target_col:
            continue
        try:
            series = df[col].dropna()
            if len(series) < 5:
                continue
            sorted_vals = np.sort(series.values)
            q1 = float(np.percentile(sorted_vals, 25))
            median = float(np.percentile(sorted_vals, 50))
            q3 = float(np.percentile(sorted_vals, 75))
            iqr = q3 - q1
            if iqr <= 0.0:
                lower_whisker = float(sorted_vals.min())
                upper_whisker = float(sorted_vals.max())
                outliers_list = []
                outlier_pct = 0.0
            else:
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                
                non_outliers = sorted_vals[(sorted_vals >= lower_fence) & (sorted_vals <= upper_fence)]
                if len(non_outliers) > 0:
                    lower_whisker = float(non_outliers.min())
                    upper_whisker = float(non_outliers.max())
                else:
                    lower_whisker = q1
                    upper_whisker = q3
                    
                outliers = sorted_vals[(sorted_vals < lower_whisker) | (sorted_vals > upper_whisker)]
                outliers_list = [float(x) for x in outliers[:100]]
                outlier_pct = len(outliers) / len(sorted_vals)
                
            boxplot_obj = {
                "type": "boxplot",
                "title": f"Outlier Diagnostics: {col}",
                "x_label": "",
                "y_label": col,
                "data": [],
                "box_stats": {
                    "min": lower_whisker,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "max": upper_whisker,
                    "outliers": outliers_list
                }
            }
            scored_boxplots.append((boxplot_obj, outlier_pct))
        except Exception as box_err:
            sys.stderr.write(f"Boxplot error for {col}: {str(box_err)}\n")
            
    scored_boxplots.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored_boxplots[:5]]
