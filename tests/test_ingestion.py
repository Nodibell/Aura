import os
import sys
import tempfile
import shutil
import zipfile
import numpy as np
import pytest
import pandas as pd
import json
from PIL import Image

# Ensure Aura directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Aura')))

from utils.ingestion import (
    detect_dataset_format,
    parse_metadata_file,
    convert_to_npz,
    ingest_dataset
)

def create_dummy_image(path):
    img = Image.new('RGB', (16, 16), color='red')
    img.save(path)

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)

def test_detect_dataset_format_empty(temp_dir):
    assert detect_dataset_format(temp_dir) == 'unknown'

def test_detect_dataset_format_yolo_yaml(temp_dir):
    yaml_content = "names: ['cat', 'dog']\nnc: 2\ntrain: images/train"
    with open(os.path.join(temp_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    assert detect_dataset_format(temp_dir) == 'yolo'

def test_detect_dataset_format_yolo_folders(temp_dir):
    os.makedirs(os.path.join(temp_dir, "images"))
    labels_dir = os.path.join(temp_dir, "labels")
    os.makedirs(labels_dir)
    with open(os.path.join(labels_dir, "label1.txt"), "w") as f:
        f.write("0 0.5 0.5 0.2 0.2")
    assert detect_dataset_format(temp_dir) == 'yolo'

def test_detect_dataset_format_segmentation(temp_dir):
    images_dir = os.path.join(temp_dir, "images")
    masks_dir = os.path.join(temp_dir, "masks")
    os.makedirs(images_dir)
    os.makedirs(masks_dir)
    create_dummy_image(os.path.join(images_dir, "img1.png"))
    create_dummy_image(os.path.join(masks_dir, "img1.png"))
    assert detect_dataset_format(temp_dir) == 'segmentation'

def test_detect_dataset_format_class_hierarchy(temp_dir):
    os.makedirs(os.path.join(temp_dir, "cats"))
    os.makedirs(os.path.join(temp_dir, "dogs"))
    create_dummy_image(os.path.join(temp_dir, "cats", "c1.png"))
    create_dummy_image(os.path.join(temp_dir, "dogs", "d1.png"))
    assert detect_dataset_format(temp_dir) == 'class_hierarchy'

def test_detect_dataset_format_flat(temp_dir):
    create_dummy_image(os.path.join(temp_dir, "c1.png"))
    create_dummy_image(os.path.join(temp_dir, "d1.png"))
    assert detect_dataset_format(temp_dir) == 'flat'

def test_parse_metadata_csv(temp_dir):
    metadata = pd.DataFrame({
        'filename': ['img1.png', 'img2.png'],
        'label': ['cat', 'dog']
    })
    metadata.to_csv(os.path.join(temp_dir, "labels.csv"), index=False)
    parsed = parse_metadata_file(temp_dir)
    assert parsed == {'img1.png': 'cat', 'img2.png': 'dog'}

def test_parse_metadata_json(temp_dir):
    metadata = {
        'img1.png': 'cat',
        'img2.png': 'dog'
    }
    with open(os.path.join(temp_dir, "labels.json"), "w") as f:
        json.dump(metadata, f)
    parsed = parse_metadata_file(temp_dir)
    assert parsed == {'img1.png': 'cat', 'img2.png': 'dog'}

def test_convert_to_npz_class_hierarchy(temp_dir):
    os.makedirs(os.path.join(temp_dir, "cats"))
    os.makedirs(os.path.join(temp_dir, "dogs"))
    create_dummy_image(os.path.join(temp_dir, "cats", "c1.png"))
    create_dummy_image(os.path.join(temp_dir, "dogs", "d1.png"))
    
    npz_path = convert_to_npz(temp_dir, "class_hierarchy")
    assert os.path.exists(npz_path)
    assert npz_path.endswith(".npz")
    
    data = np.load(npz_path, allow_pickle=True)
    assert 'X' in data.keys()
    assert 'y' in data.keys()
    assert data['X'].shape == (2, 32, 32, 3)
    assert list(data['y']) == ['cats', 'dogs']
    os.remove(npz_path)

def test_convert_to_npz_flat_with_csv(temp_dir):
    create_dummy_image(os.path.join(temp_dir, "img1.png"))
    create_dummy_image(os.path.join(temp_dir, "img2.png"))
    metadata = pd.DataFrame({
        'image': ['img1.png', 'img2.png'],
        'category': ['apple', 'banana']
    })
    metadata.to_csv(os.path.join(temp_dir, "metadata.csv"), index=False)
    
    npz_path = convert_to_npz(temp_dir, "flat")
    assert os.path.exists(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    assert list(data['y']) == ['apple', 'banana']
    os.remove(npz_path)

def test_ingest_dataset_yolo(temp_dir):
    # Setup YOLO structure
    yaml_content = "names: ['cat', 'dog']\nnc: 2\ntrain: images/train"
    with open(os.path.join(temp_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    
    resolved_path, resolved_type = ingest_dataset(temp_dir)
    assert resolved_path == temp_dir
    assert resolved_type == "object_detection"

def test_ingest_dataset_zip_classification(temp_dir):
    # Create a source classification directory to zip
    src_dir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(src_dir, "cats"))
        create_dummy_image(os.path.join(src_dir, "cats", "c1.png"))
        
        # Zip it
        zip_path = os.path.join(temp_dir, "dataset.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(os.path.join(src_dir, "cats", "c1.png"), "cats/c1.png")
            
        resolved_path, resolved_type = ingest_dataset(zip_path)
        assert os.path.isdir(resolved_path)
        assert resolved_type == "image"
        
        # Check that it extracted the image
        img_path = os.path.join(resolved_path, "cats", "c1.png")
        assert os.path.exists(img_path)
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)
