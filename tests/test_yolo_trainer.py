import os
import shutil
import tempfile
import yaml
import numpy as np
from PIL import Image
import pytest
from Aura.pipelines.yolo_trainer import train_yolo_and_evaluate

def test_yolo_trainer_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create folder structure
        images_train_dir = os.path.join(tmp_dir, "images", "train")
        images_val_dir = os.path.join(tmp_dir, "images", "val")
        labels_train_dir = os.path.join(tmp_dir, "labels", "train")
        labels_val_dir = os.path.join(tmp_dir, "labels", "val")
        
        os.makedirs(images_train_dir, exist_ok=True)
        os.makedirs(images_val_dir, exist_ok=True)
        os.makedirs(labels_train_dir, exist_ok=True)
        os.makedirs(labels_val_dir, exist_ok=True)
        
        # 1. Create a dummy image for training and validation (128x128 pixel black image)
        img_train_path = os.path.join(images_train_dir, "img1.jpg")
        img_val_path = os.path.join(images_val_dir, "img2.jpg")
        
        img = Image.new("RGB", (128, 128), "black")
        img.save(img_train_path)
        img.save(img_val_path)
        
        # 2. Create labels: one object of class 0 in center
        label_content = "0 0.5 0.5 0.4 0.4\n"
        
        with open(os.path.join(labels_train_dir, "img1.txt"), "w") as f:
            f.write(label_content)
        with open(os.path.join(labels_val_dir, "img2.txt"), "w") as f:
            f.write(label_content)
            
        # 3. Create dataset.yaml
        dataset_yaml = {
            "path": tmp_dir,
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "dummy_class"
            }
        }
        
        with open(os.path.join(tmp_dir, "dataset.yaml"), "w") as f:
            yaml.dump(dataset_yaml, f)
            
        # Run YOLO training for 1 epoch on 128x128 image size
        res = train_yolo_and_evaluate(
            root_dir=tmp_dir,
            model_export_path=None,
            epochs=1,
            imgsz=128
        )
        
        # Verify output
        assert res is not None
        if "error" in res and res["error"] is not None:
            # If download fails, we allow it to pass as a skipped test to handle sandboxed/offline builders
            if "download" in str(res["error"]).lower() or "connection" in str(res["error"]).lower():
                pytest.skip(f"YOLO weights download failed: {res['error']}")
            else:
                pytest.fail(f"YOLO training failed with error: {res['error']}")
            
        assert "metrics" in res
        assert res["metrics"]["score_type"] == "mAP@0.5"
        assert isinstance(res["metrics"]["score"], float)
        assert isinstance(res["metrics"]["additional_metrics"]["mAP@0.5:0.95"], float)
        
        # Clean up ultralytics runs
        if os.path.exists("runs"):
            shutil.rmtree("runs")
