import os
from PIL import Image

src_path = "/Users/oleksiichumak/.gemini/antigravity-ide/brain/353fd22b-ebbf-4af3-b434-323f84fe1606/autoeda_app_icon_fullscreen_1781688316372.png"
if not os.path.exists(src_path):
    print("Source image not found.")
    exit(1)

img = Image.open(src_path)
print(f"Image format: {img.format}, size: {img.size}, mode: {img.mode}")

# Inspect corner pixels (top-left, top-right, bottom-left, bottom-right)
w, h = img.size
pixels = img.load()
corners = [
    (0, 0),
    (w - 1, 0),
    (0, h - 1),
    (w - 1, h - 1),
    (w // 2, 0),
    (0, h // 2)
]
for x, y in corners:
    print(f"Pixel at ({x}, {y}): {pixels[x, y]}")
