import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# === CONFIGURATION ===
background_dir = "BACKGROUNDS"
blade_dir = "BLADE"
veins_dir = "VEINS"
mask_dir = "MASKS"
os.makedirs(mask_dir, exist_ok=True)

# === GET IMAGE NAMES ===
image_files = [f for f in os.listdir(background_dir) if f.lower().endswith(".png")]

# === PROCESS EACH IMAGE ===
for fname in tqdm(image_files):
    name = os.path.splitext(fname)[0]
    background_path = os.path.join(background_dir, fname)
    blade_path = os.path.join(blade_dir, f"{name}_blade.txt")
    vein_path = os.path.join(veins_dir, f"{name}_veins.txt")
    mask_path = os.path.join(mask_dir, f"{name}.png")

    # Skip if already exists
    if os.path.exists(mask_path):
        continue

    # Load image to get size
    bg_img = Image.open(background_path)
    w, h = bg_img.size

    # Create blank mask (uint8): background = 0
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Draw blade: label = 1
    if os.path.exists(blade_path):
        blade_coords = np.loadtxt(blade_path)
        blade_coords_tuple = [tuple(pt) for pt in blade_coords]
        draw.polygon(blade_coords_tuple, fill=1)

    # Draw veins: label = 2 (overwrite blade if overlapping)
    if os.path.exists(vein_path):
        vein_coords = np.loadtxt(vein_path)
        vein_coords_tuple = [tuple(pt) for pt in vein_coords]
        draw.polygon(vein_coords_tuple, fill=2)

    # Save mask as image
    mask.save(mask_path)

print("✅ All masks created and saved to MASKS")