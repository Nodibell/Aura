import os
import json
from PIL import Image

src_path = "/Users/oleksiichumak/.gemini/antigravity-ide/brain/353fd22b-ebbf-4af3-b434-323f84fe1606/autoeda_app_icon_fullscreen_processed.png"
dest_dir = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/Assets.xcassets/AppIcon.appiconset"

os.makedirs(dest_dir, exist_ok=True)

# Define sizes and filenames for macOS app icon set
icon_specs = [
    {"size_px": 16,   "scale": "1x", "filename": "icon_16x16.png"},
    {"size_px": 32,   "scale": "2x", "filename": "icon_16x16@2x.png"},
    {"size_px": 32,   "scale": "1x", "filename": "icon_32x32.png"},
    {"size_px": 64,   "scale": "2x", "filename": "icon_32x32@2x.png"},
    {"size_px": 128,  "scale": "1x", "filename": "icon_128x128.png"},
    {"size_px": 256,  "scale": "2x", "filename": "icon_128x128@2x.png"},
    {"size_px": 256,  "scale": "1x", "filename": "icon_256x256.png"},
    {"size_px": 512,  "scale": "2x", "filename": "icon_256x256@2x.png"},
    {"size_px": 512,  "scale": "1x", "filename": "icon_512x512.png"},
    {"size_px": 1024, "scale": "2x", "filename": "icon_512x512@2x.png"}
]

# Open source image
img = Image.open(src_path).convert("RGBA")

# Resize and save each icon
for spec in icon_specs:
    size = spec["size_px"]
    filename = spec["filename"]
    save_path = os.path.join(dest_dir, filename)
    
    # Resize with high quality Resampling
    resized_img = img.resize((size, size), Image.Resampling.LANCZOS)
    resized_img.save(save_path, "PNG")
    print(f"Generated {size}x{size} icon: {filename}")

# Create Xcode AppIcon Contents.json
contents = {
  "images" : [
    {
      "idiom" : "mac",
      "size" : "16x16",
      "scale" : "1x",
      "filename" : "icon_16x16.png"
    },
    {
      "idiom" : "mac",
      "size" : "16x16",
      "scale" : "2x",
      "filename" : "icon_16x16@2x.png"
    },
    {
      "idiom" : "mac",
      "size" : "32x32",
      "scale" : "1x",
      "filename" : "icon_32x32.png"
    },
    {
      "idiom" : "mac",
      "size" : "32x32",
      "scale" : "2x",
      "filename" : "icon_32x32@2x.png"
    },
    {
      "idiom" : "mac",
      "size" : "128x128",
      "scale" : "1x",
      "filename" : "icon_128x128.png"
    },
    {
      "idiom" : "mac",
      "size" : "128x128",
      "scale" : "2x",
      "filename" : "icon_128x128@2x.png"
    },
    {
      "idiom" : "mac",
      "size" : "256x256",
      "scale" : "1x",
      "filename" : "icon_256x256.png"
    },
    {
      "idiom" : "mac",
      "size" : "256x256",
      "scale" : "2x",
      "filename" : "icon_256x256@2x.png"
    },
    {
      "idiom" : "mac",
      "size" : "512x512",
      "scale" : "1x",
      "filename" : "icon_512x512.png"
    },
    {
      "idiom" : "mac",
      "size" : "512x512",
      "scale" : "2x",
      "filename" : "icon_512x512@2x.png"
    }
  ],
  "info" : {
    "author" : "xcode",
    "version" : 1
  }
}

contents_json_path = os.path.join(dest_dir, "Contents.json")
with open(contents_json_path, "w") as f:
    json.dump(contents, f, indent=2)
    
print("Updated Contents.json successfully.")
