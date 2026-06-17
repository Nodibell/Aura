import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Aura")))
import shutil
import tempfile
import numpy as np
from PIL import Image
from analyze import analyze

def test_segmentation():
    # 1. Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        images_dir = os.path.join(temp_dir, "images")
        masks_dir = os.path.join(temp_dir, "1st_manual")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # 2. Save 5 dummy image-mask pairs
        np.random.seed(42)
        for i in range(5):
            # RGB image (e.g. 128x128)
            img_arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            # Binary mask (e.g. 128x128 with some simulated vessels)
            mask_arr = np.zeros((128, 128), dtype=np.uint8)
            mask_arr[40:80, 50:70] = 255  # mock vessel
            
            img = Image.fromarray(img_arr)
            mask = Image.fromarray(mask_arr)
            
            img.save(os.path.join(images_dir, f"sample_{i}.png"))
            mask.save(os.path.join(masks_dir, f"sample_{i}.png"))
            
        print("Created mock paired image/mask dataset.")
        
        # 3. Call analyze on the directory
        print("Running analyze...")
        res = analyze(temp_dir, dataset_type="image")
        
        if "error" in res and res["error"] is not None:
            print("FAILED with error:", res["error"])
        else:
            print("SUCCESS!")
            print("Task Type:", res.get("task_type"))
            print("Row Count:", res.get("row_count"))
            print("Metrics:", res.get("metrics"))
            print("Charts count:", len(res.get("charts", [])))
            print("Chart titles:", [c["title"] for c in res.get("charts", [])])
            
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_segmentation()
