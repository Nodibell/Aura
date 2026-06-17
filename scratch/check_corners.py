from PIL import Image
import numpy as np

img_path = "/Users/oleksiichumak/.gemini/antigravity-ide/brain/353fd22b-ebbf-4af3-b434-323f84fe1606/autoeda_app_icon_1781688166336.png"
img = Image.open(img_path).convert("RGB")
arr = np.array(img)

# Unique RGB values in top-left 100x100 pixels
colors = {}
for y in range(100):
    for x in range(100):
        color = tuple(arr[y, x])
        colors[color] = colors.get(color, 0) + 1

# Sort by frequency and print top 10
sorted_colors = sorted(colors.items(), key=lambda item: item[1], reverse=True)
print("Top 10 colors in top-left 100x100 corner:")
for c, count in sorted_colors[:10]:
    print(f"- RGB {c}: count {count}")
