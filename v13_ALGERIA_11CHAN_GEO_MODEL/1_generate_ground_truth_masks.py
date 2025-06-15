# 1_generate_ground_truth_masks.py

# === IMPORTS ===
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# === CONFIGURATION ===
# Input directories from previous processing step
PROCESSED_RGB_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/RGB_IMAGES"
PROCESSED_BLADE_COORDS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/BLADE_COORDS"
PROCESSED_VEIN_COORDS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/VEIN_COORDS"

# Output directory for generated masks
GROUND_TRUTH_MASKS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/GROUND_TRUTH_MASKS"

# Ensure output directory exists
os.makedirs(GROUND_TRUTH_MASKS_DIR, exist_ok=True)

# === MAIN PROCESSING ===
# Get a list of all processed RGB image filenames (they define the common set of samples)
image_files = sorted([f for f in os.listdir(PROCESSED_RGB_DIR) if f.lower().endswith(".png")])

print(f"Generating pixel segmentation masks for {len(image_files)} images...")

for fname in tqdm(image_files, desc="Creating masks"):
    base_name = os.path.splitext(fname)[0] # e.g., 'sample_id_1'

    rgb_path = os.path.join(PROCESSED_RGB_DIR, fname)
    blade_coords_path = os.path.join(PROCESSED_BLADE_COORDS_DIR, f"{base_name}_blade.txt")
    vein_coords_path = os.path.join(PROCESSED_VEIN_COORDS_DIR, f"{base_name}_veins.txt")
    mask_output_path = os.path.join(GROUND_TRUTH_MASKS_DIR, f"{base_name}.png")

    # Skip if mask already exists
    if os.path.exists(mask_output_path):
        continue

    # Validate existence of necessary files
    if not os.path.exists(rgb_path):
        print(f"⚠️ Warning: RGB image '{rgb_path}' not found, skipping mask generation for {base_name}.")
        continue
    if not os.path.exists(blade_coords_path):
        print(f"⚠️ Warning: Blade coordinates '{blade_coords_path}' not found, skipping mask generation for {base_name}.")
        continue
    if not os.path.exists(vein_coords_path):
        print(f"⚠️ Warning: Vein coordinates '{vein_coords_path}' not found, skipping mask generation for {base_name}.")
        continue

    try:
        # Load the processed RGB image to get dimensions (all images should now be TARGET_SIZE)
        img = Image.open(rgb_path)
        width, height = img.size

        # Create a blank mask image (L-mode for single channel, 8-bit, 0 for background)
        mask = Image.new("L", (width, height), 0) # Background = 0
        draw = ImageDraw.Draw(mask)

        # Load blade coordinates and draw blade polygon (label = 1)
        blade_coords = np.loadtxt(blade_coords_path)
        if blade_coords.ndim == 1: # Handle case of single point (convert to 2D array)
            blade_coords = blade_coords[np.newaxis, :]
        if blade_coords.shape[0] >= 3: # Need at least 3 points for a polygon
            # Convert numpy array to list of tuples for Pillow's polygon drawing
            blade_coords_tuple = [tuple(pt) for pt in blade_coords.tolist()]
            draw.polygon(blade_coords_tuple, fill=1) # Blade = 1
        else:
            print(f"⚠️ Warning: Blade coordinates for {base_name} have less than 3 points, skipping blade mask.")

        # Load vein coordinates and draw vein polygon (label = 2)
        # Note: Veins are drawn *after* blade, so overlapping vein pixels will overwrite blade pixels with 2.
        vein_coords = np.loadtxt(vein_coords_path)
        if vein_coords.ndim == 1: # Handle case of single point
            vein_coords = vein_coords[np.newaxis, :]
        if vein_coords.shape[0] >= 3: # Need at least 3 points for a polygon
            vein_coords_tuple = [tuple(pt) for pt in vein_coords.tolist()]
            draw.polygon(vein_coords_tuple, fill=2) # Vein = 2
        else:
            print(f"⚠️ Warning: Vein coordinates for {base_name} have less than 3 points, skipping vein mask.")

        # Save the generated mask
        mask.save(mask_output_path)

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

print(f"\n✅ All masks generated and saved to '{GROUND_TRUTH_MASKS_DIR}'.")