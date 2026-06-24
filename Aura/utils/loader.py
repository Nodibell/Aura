import os
import sys
import json
import urllib.request
import urllib.parse
import tempfile
import contextlib
import shutil
import numpy as np
import pandas as pd
from utils.helpers import print_progress

def download_dataset(url):
    """
    Parses url and downloads dataset from Hugging Face, Kaggle, or generic URL.
    Saves to a persistent cache directory based on SHA256 of the URL.
    Returns: local_file_path
    """
    import hashlib
    import glob
    parsed = urllib.parse.urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path
    
    # Persistent cache directory in system temp
    cache_dir = os.path.join(tempfile.gettempdir(), "aura_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cached file name using sha256 hash of URL
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    
    # Check if already cached as a directory or a file
    url_folder = os.path.join(cache_dir, url_hash)
    if os.path.exists(url_folder) and os.path.isdir(url_folder):
        print_progress(0.10, "Found cached dataset directory...")
        
        # Check if it has a YOLO structure first
        has_yolo = False
        for root, _, filenames in os.walk(url_folder):
            if any(f.lower() in ("dataset.yaml", "data.yaml", "yolo.yaml") for f in filenames):
                has_yolo = True
                break
        if has_yolo:
            return url_folder
            
        for root, _, filenames in os.walk(url_folder):
            for f in filenames:
                if f.lower().endswith((".csv", ".parquet", ".xlsx", ".xls", ".tsv", ".npz", ".json", ".jsonl")):
                    return os.path.join(root, f)
        
        # If no tabular/NPZ files, check if there are images
        image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")
        for root, _, filenames in os.walk(url_folder):
            for f in filenames:
                if f.lower().endswith(image_extensions):
                    return url_folder
                    
    # For generic URLs only, check if cached as a file
    if "huggingface.co" not in netloc and "kaggle.com" not in netloc:
        matching_files = glob.glob(os.path.join(cache_dir, f"{url_hash}.*"))
        if matching_files:
            print_progress(0.10, "Found cached dataset...")
            return matching_files[0]
        
    print_progress(0.05, "Downloading remote dataset...")
    
    with contextlib.redirect_stdout(sys.stderr):
        # 1. Hugging Face Datasets
        if "huggingface.co" in netloc:
            if "/datasets/" in path:
                parts = [p for p in path.split("/") if p]
                if len(parts) >= 3:
                    repo_id = f"{parts[1]}/{parts[2]}"
                    
                    # Check if specific file is requested in the URL
                    file_in_url = None
                    for keyword in ["blob", "resolve"]:
                        if keyword in parts:
                            idx = parts.index(keyword)
                            if len(parts) > idx + 2:
                                file_in_url = "/".join(parts[idx+2:])
                                break
                    
                    try:
                        from huggingface_hub import hf_hub_download, HfApi
                    except ImportError:
                        raise ImportError("The 'huggingface_hub' library is required to download Hugging Face datasets. Please run: pip install huggingface_hub")
                    
                    token = os.environ.get("HF_TOKEN")
                    if token == "":
                        token = None
                    
                    if file_in_url:
                        ext = os.path.splitext(file_in_url)[1].lower()
                        target_path = os.path.join(cache_dir, f"{url_hash}{ext}")
                        local_path = hf_hub_download(repo_id=repo_id, filename=file_in_url, repo_type="dataset", token=token)
                        shutil.copy2(local_path, target_path)
                        return target_path
                    else:
                        api = HfApi()
                        try:
                            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
                        except Exception as e:
                            raise Exception(f"Failed to list files in Hugging Face dataset '{repo_id}': {str(e)}")
                        
                        url_folder = os.path.join(cache_dir, url_hash)
                        if os.path.exists(url_folder):
                            shutil.rmtree(url_folder)
                        os.makedirs(url_folder, exist_ok=True)
                        
                        data_files = [f for f in files if f.lower().endswith((".csv", ".parquet", ".xlsx", ".xls", ".tsv", ".npz", ".json", ".jsonl"))]
                        if not data_files:
                            raise Exception(f"No CSV, Parquet, TSV, NPZ, or JSON file found in Hugging Face repository '{repo_id}'. Files found: {files}")
                        
                        # Limit to first 10 data files to prevent huge downloads
                        data_files = data_files[:10]
                        
                        first_local_path = None
                        for f in data_files:
                            try:
                                local_path = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset", token=token)
                                target_dest = os.path.join(url_folder, f)
                                os.makedirs(os.path.dirname(target_dest), exist_ok=True)
                                shutil.copy2(local_path, target_dest)
                                if not first_local_path:
                                    first_local_path = target_dest
                            except Exception as dl_err:
                                sys.stderr.write(f"Warning: Failed to download HF file '{f}': {str(dl_err)}\n")
                                
                        if not first_local_path:
                            raise Exception(f"Failed to download any data files from Hugging Face dataset '{repo_id}'")
                        return first_local_path
                else:
                    raise Exception("Invalid Hugging Face dataset URL format.")
            else:
                raise Exception("Hugging Face URLs must point to a dataset under huggingface.co/datasets/...")
                    
        # 2. Kaggle Datasets / Notebooks
        elif "kaggle.com" in netloc:
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 3 and parts[0] in ["datasets", "code", "kernels"]:
                owner = parts[1]
                name = parts[2]
                identifier = f"{owner}/{name}"
                is_kernel = (parts[0] in ["code", "kernels"])
                
                # Check Kaggle API credentials
                if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
                    home = os.path.expanduser("~")
                    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
                    if not os.path.exists(kaggle_json):
                        raise Exception("Kaggle credentials are required. Please configure your Kaggle Username and API Key in Settings.")
                
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                except ImportError:
                    raise ImportError("The 'kaggle' library is required to download Kaggle files. Please run: pip install kaggle")
                
                temp_dir = tempfile.mkdtemp(prefix="kaggle_download_")
                try:
                    api = KaggleApi()
                    api.authenticate()
                    
                    if is_kernel:
                        print_progress(0.08, "Downloading Kaggle notebook outputs...")
                        try:
                            api.kernels_output(identifier, path=temp_dir)
                        except Exception as out_err:
                            sys.stderr.write(f"Warning: Failed to download kernel outputs: {str(out_err)}\n")
                        
                        # Also pull metadata to get input datasets
                        print_progress(0.09, "Retrieving notebook metadata...")
                        try:
                            api.kernels_pull(identifier, path=temp_dir, metadata=True)
                            metadata_file = os.path.join(temp_dir, "kernel-metadata.json")
                            if os.path.exists(metadata_file):
                                with open(metadata_file, "r") as f:
                                    meta = json.load(f)
                                    if "dataset_sources" in meta and meta["dataset_sources"]:
                                        ds_id = meta["dataset_sources"][0]
                                        print_progress(0.11, f"Downloading input dataset '{ds_id}'...")
                                        api.dataset_download_files(ds_id, path=temp_dir, unzip=True)
                        except Exception as meta_err:
                            sys.stderr.write(f"Warning: Failed to pull metadata/datasets: {str(meta_err)}\n")
                    else:
                        print_progress(0.08, "Downloading Kaggle dataset files...")
                        api.dataset_download_files(identifier, path=temp_dir, unzip=True)
                    
                    # Copy the entire downloaded folder to cache as a directory
                    url_folder = os.path.join(cache_dir, url_hash)
                    if os.path.exists(url_folder):
                        shutil.rmtree(url_folder)
                    shutil.copytree(temp_dir, url_folder)
                    
                    # Search downloaded directory for data files
                    downloaded_files = []
                    for root, _, filenames in os.walk(url_folder):
                        for f in filenames:
                            downloaded_files.append(os.path.join(root, f))
                            
                    # Check if it has a YOLO structure first
                    if any(os.path.basename(f).lower() in ("dataset.yaml", "data.yaml", "yolo.yaml") for f in downloaded_files):
                        return url_folder
                        
                    target_file = None
                    for f in downloaded_files:
                        if f.lower().endswith((".csv", ".parquet", ".xlsx", ".xls", ".tsv", ".npz", ".json", ".jsonl")):
                            target_file = f
                            break
                            
                    if not target_file:
                        # Check if we have image files
                        image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")
                        has_images = any(f.lower().endswith(image_extensions) for f in downloaded_files)
                        if has_images:
                            target_file = url_folder
                            
                    if not target_file:
                        raise Exception("No CSV, Parquet, TSV, NPZ, or Image files found in downloaded Kaggle files.")
                        
                    return target_file
                except Exception as e:
                    type_str = "notebook" if is_kernel else "dataset"
                    raise Exception(f"Failed to download from Kaggle {type_str} '{identifier}': {str(e)}")
                finally:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            else:
                raise Exception("Invalid Kaggle URL format. Must point to a dataset or notebook (e.g. kaggle.com/datasets/... or kaggle.com/code/...).")
                
        # 3. Generic HTTP/HTTPS Direct Link
        else:
            try:
                # Guess extension from URL path
                url_ext = os.path.splitext(path)[1].lower()
                query = urllib.parse.parse_qs(parsed.query)
                if not url_ext and "select" in query:
                    url_ext = os.path.splitext(query["select"][0])[1].lower()
                
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
                )
                with urllib.request.urlopen(req) as response:
                    shutil.copyfileobj(response, temp_file)
                temp_file.close()
                
                # Deduce extension based on magic bytes or fallback
                ext = url_ext
                if not ext or ext not in [".csv", ".parquet", ".xlsx", ".xls", ".tsv", ".npz", ".json", ".jsonl"]:
                    with open(temp_file.name, "rb") as f:
                        magic = f.read(4)
                    if magic.startswith(b"PAR1"):
                        ext = ".parquet"
                    elif magic.startswith(b"PK\x03\x04"):
                        ext = ".npz"
                    elif magic.startswith(b"{") or magic.startswith(b"["):
                        ext = ".json"
                    elif b"," in magic or b"\t" in magic or b"\n" in magic:
                        ext = ".tsv" if b"\t" in magic else ".csv"
                    else:
                        ext = ".csv"  # fallback default
                        
                target_path = os.path.join(cache_dir, f"{url_hash}{ext}")
                shutil.move(temp_file.name, target_path)
                return target_path
            except Exception as e:
                if 'temp_file' in locals() and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise Exception(f"Failed to download from direct URL: {str(e)}")

def load_dataset(file_path, nrows=None):
    ext = os.path.splitext(file_path)[1].lower()
    magic = b""
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
    except Exception:
        pass

    if magic.startswith(b"PAR1") or ext == ".parquet":
        try:
            df = pd.read_parquet(file_path)
            if nrows:
                df = df.head(nrows)
            return df.replace([np.inf, -np.inf], np.nan)
        except ImportError:
            raise ImportError("PyArrow or fastparquet is required to read Parquet. Please install: pip install pyarrow")
    elif ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(file_path)
            if nrows:
                df = df.head(nrows)
            df.columns = df.columns.astype(str).str.strip()
            return df.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            raise Exception(f"Failed to load Excel dataset: {str(e)}")
    elif ext == ".jsonl" or (ext == ".json" and magic.startswith(b"[")):
        try:
            df = pd.read_json(file_path, lines=(ext == ".jsonl"))
            if nrows:
                df = df.head(nrows)
            df.columns = df.columns.astype(str).str.strip()
            return df.replace([np.inf, -np.inf], np.nan)
        except Exception:
            pass
    elif ext == ".json":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                list_key = None
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        list_key = k
                        break
                if list_key:
                    df = pd.DataFrame(data[list_key])
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame()
            if nrows:
                df = df.head(nrows)
            df.columns = df.columns.astype(str).str.strip()
            return df.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            raise Exception(f"Failed to load JSON dataset: {str(e)}")
    elif magic.startswith(b"PK\x03\x04") or ext == ".npz":
        npz = np.load(file_path, allow_pickle=True)
        keys = list(npz.keys())
        
        X_arr = None
        y_arr = None
        
        # Look for typical X (data, features, images, x_train) and y (target, labels, y_train) keys
        x_keys = [k for k in keys if k.lower() in ["x", "x_train", "train_x", "data", "features", "images"]]
        y_keys = [k for k in keys if k.lower() in ["y", "y_train", "train_y", "labels", "target", "classes"]]
        
        if x_keys and y_keys:
            X_arr = npz[x_keys[0]]
            y_arr = npz[y_keys[0]]
        elif len(keys) >= 2:
            # Fallback: largest dimension is X, second is y
            sorted_keys = sorted(keys, key=lambda k: len(npz[k].shape), reverse=True)
            X_arr = npz[sorted_keys[0]]
            y_arr = npz[sorted_keys[1]]
        elif len(keys) == 1:
            X_arr = npz[keys[0]]
            
        if X_arr is not None:
            # Flatten X if > 2D (e.g., image datasets like CIFAR-10)
            orig_shape = X_arr.shape
            if len(orig_shape) > 2:
                num_samples = orig_shape[0]
                flat_dim = int(np.prod(orig_shape[1:]))
                X_arr = X_arr.reshape((num_samples, flat_dim))
                
            # If doing analysis, downsample to 1500 samples and 100 features max to keep performance fast
            if not nrows:
                max_samples = 1500
                max_features = 100
                
                num_samples = X_arr.shape[0]
                num_features = X_arr.shape[1] if len(X_arr.shape) > 1 else 1
                
                if num_samples > max_samples:
                    np.random.seed(42)
                    row_indices = np.random.choice(num_samples, max_samples, replace=False)
                    X_arr = X_arr[row_indices]
                    if y_arr is not None:
                        y_arr = y_arr[row_indices]
                
                if num_features > max_features:
                    X_arr = X_arr[:, :max_features]
            else:
                # Preview mode: limit to nrows and max 100 features
                X_arr = X_arr[:nrows]
                if y_arr is not None:
                    y_arr = y_arr[:nrows]
                if X_arr.shape[1] > 100:
                    X_arr = X_arr[:, :100]
                    
            # Build DataFrame
            col_names = [f"pixel_{i}" for i in range(X_arr.shape[1])]
            df = pd.DataFrame(X_arr, columns=col_names)
            
            if y_arr is not None:
                y_flat = y_arr.flatten()
                if len(y_flat) == len(df):
                    df["target"] = y_flat
            df.columns = df.columns.str.strip()
            return df.replace([np.inf, -np.inf], np.nan)
        else:
            raise ValueError("Could not find suitable data arrays in .npz archive.")
    else:
        # Check if tab-separated or comma-separated
        sep = '\t' if (ext == ".tsv" or (magic and b"\t" in magic and b"," not in magic)) else ','
        
        try:
            if nrows:
                df = pd.read_csv(file_path, sep=sep, nrows=nrows, on_bad_lines='skip', encoding='utf-8')
            else:
                df = pd.read_csv(file_path, sep=sep, on_bad_lines='skip', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                if nrows:
                    df = pd.read_csv(file_path, sep=sep, nrows=nrows, on_bad_lines='skip', encoding='latin-1')
                else:
                    df = pd.read_csv(file_path, sep=sep, on_bad_lines='skip', encoding='latin-1')
            except Exception as e:
                try:
                    if nrows:
                        df = pd.read_csv(file_path, sep=sep, nrows=nrows, on_bad_lines='skip', encoding='utf-8-sig')
                    else:
                        df = pd.read_csv(file_path, sep=sep, on_bad_lines='skip', encoding='utf-8-sig')
                except Exception:
                    raise e
        df.columns = df.columns.str.strip()
        return df.replace([np.inf, -np.inf], np.nan)

def _infer_dataset_type(df, file_path=""):
    """Best-effort heuristic to guess dataset type from the DataFrame and file path."""
    if file_path and os.path.isdir(file_path):
        # Check for YOLO dataset structure
        has_yaml = False
        for name in ("dataset.yaml", "data.yaml", "yolo.yaml"):
            if os.path.exists(os.path.join(file_path, name)):
                has_yaml = True
                break
        if not has_yaml:
            for entry in os.scandir(file_path):
                if entry.is_dir():
                    for name in ("dataset.yaml", "data.yaml", "yolo.yaml"):
                        if os.path.exists(os.path.join(entry.path, name)):
                            has_yaml = True
                            break
                if has_yaml:
                    break
        if has_yaml:
            return "object_detection"

        # Check for images and labels/annotations dirs
        has_images_and_labels = False
        for root, dirs, files in os.walk(file_path):
            if "images" in dirs and ("labels" in dirs or "annotations" in dirs):
                has_images_and_labels = True
                break
        if has_images_and_labels:
            return "object_detection"

        # Fallback to image if any images are found
        image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")
        for root, dirs, files in os.walk(file_path):
            if any(f.lower().endswith(image_extensions) for f in files):
                return "image"

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".npz":
        return "image"
    cols_lower = [c.lower() for c in df.columns]
    # Time series: has a date/time-like column or 'timestamp', 'date', 'time', 'year' col
    ts_keywords = {"date", "time", "timestamp", "datetime", "year", "month", "period", "week"}
    if any(any(kw in c for kw in ts_keywords) for c in cols_lower):
        return "timeseries"
    # NLP: check if any column has long average string length (>60 chars) AND
    # has real word content (avg word count > 4). This prevents long Base64
    # identifiers or UUID-like strings from being misclassified as text/NLP.
    for col in df.select_dtypes(include=[object]).columns:
        try:
            sample = df[col].dropna().astype(str)
            avg_len = sample.str.len().mean()
            if avg_len > 60:
                avg_words = sample.str.count(r'\s+').mean() + 1
                if avg_words > 4:
                    return "nlp"
        except Exception:
            pass
    return "tabular"
