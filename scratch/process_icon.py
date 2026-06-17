import os
from PIL import Image, ImageDraw

src_path = "/Users/oleksiichumak/.gemini/antigravity-ide/brain/353fd22b-ebbf-4af3-b434-323f84fe1606/autoeda_app_icon_fullscreen_1781688316372.png"
output_path = "/Users/oleksiichumak/.gemini/antigravity-ide/brain/353fd22b-ebbf-4af3-b434-323f84fe1606/autoeda_app_icon_fullscreen_processed.png"

if not os.path.exists(src_path):
    print(f"Source image not found: {src_path}")
    exit(1)

# Open the image and convert to RGBA
img = Image.open(src_path).convert("RGBA")
w, h = img.size

# We will perform flood fill from the four corners to turn the white background transparent.
# We use a threshold to handle near-white pixels from compression.
pixels = img.load()

# Flood fill helper
def flood_fill_to_transparent(image, start_x, start_y, target_color=(255, 255, 255, 255), tolerance=15):
    width, height = image.size
    pix = image.load()
    
    # Check if starting pixel is already transparent or not matching target
    start_val = pix[start_x, start_y]
    if start_val[3] == 0:
        return
    
    # Standard queue-based flood fill
    queue = [(start_x, start_y)]
    visited = set(queue)
    
    while queue:
        x, y = queue.pop(0)
        curr_val = pix[x, y]
        
        # Check tolerance (closeness to target_color)
        diff = sum(abs(curr_val[i] - target_color[i]) for i in range(3))
        if diff <= tolerance * 3:
            # Set to transparent
            pix[x, y] = (0, 0, 0, 0)
            
            # Check neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

print("Flood filling corners to transparent...")
# Flood fill from the four corners
flood_fill_to_transparent(img, 0, 0)
flood_fill_to_transparent(img, w - 1, 0)
flood_fill_to_transparent(img, 0, h - 1)
flood_fill_to_transparent(img, w - 1, h - 1)

# Find bounding box of the non-transparent area
bbox = img.getbbox()
if bbox:
    print(f"Detected bounding box of squircle: {bbox}")
    # Crop to the bounding box
    img_cropped = img.crop(bbox)
    
    # Resize back to 1024x1024 with high quality LANCZOS
    img_final = img_cropped.resize((1024, 1024), Image.Resampling.LANCZOS)
    img_final.save(output_path, "PNG")
    print(f"Saved processed fullscreen icon to: {output_path}")
else:
    img.save(output_path, "PNG")
    print("Could not detect bounding box. Saved flood-filled image directly.")
