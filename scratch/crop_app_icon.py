import os
from PIL import Image, ImageDraw

img_path = "/Users/oleksiichumak/.gemini/antigravity-ide/brain/353fd22b-ebbf-4af3-b434-323f84fe1606/autoeda_app_icon_1781688166336.png"
out_path = "/Users/oleksiichumak/.gemini/antigravity-ide/brain/353fd22b-ebbf-4af3-b434-323f84fe1606/autoeda_app_icon_clean.png"

# Load image
img = Image.open(img_path).convert("RGBA")
w, h = img.size

# 1. Create a rounded rectangle mask (standard macOS squircle approximation)
# macOS app icon squircle size is ~824x824 at 1024x1024, leaving a 100px margin.
# We will make it slightly larger (844x844, 90px margin) to preserve the outer blue/purple glow.
margin = 90
mask = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask)
# Draw a smooth white rounded rectangle
draw.rounded_rectangle(
    [margin, margin, w - margin, h - margin],
    radius=180,
    fill=255
)

# 2. Extract pixels
pixels = img.load()
mask_pixels = mask.load()

# 3. Clean up the pixels
# Everything outside the mask is transparent.
# Inside the mask, if it's within 40px of the edge and matches the checkerboard colors (grayscale, high brightness),
# we also set it to transparent to clean up the transitions.
for y in range(h):
    for x in range(w):
        if mask_pixels[x, y] == 0:
            # Set to fully transparent
            r, g, b, a = pixels[x, y]
            pixels[x, y] = (r, g, b, 0)
        else:
            # Check if near the mask boundary and matches checkerboard
            dist_to_edge = min(x - margin, (w - margin) - x, y - margin, (h - margin) - y)
            if dist_to_edge < 35:
                r, g, b, a = pixels[x, y]
                # Is it light grayscale (checkerboard)?
                is_gray = abs(r - g) <= 5 and abs(g - b) <= 5 and abs(r - b) <= 5
                is_light = r > 210 and g > 210 and b > 210
                if is_gray and is_light:
                    pixels[x, y] = (r, g, b, 0)

# Save the clean image
img.save(out_path, "PNG")
print("Successfully processed and saved clean icon to:", out_path)
