# 0_main_data_preprocessing.py

# === IMPORTS ===
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm

# === CONFIGURATION ===
RAW_TRAINING_DIR = "TRAINING_DATA" # Original raw data folder
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG", ".JPEG", ".TIF", ".TIFF", ".PNG"] # Comprehensive list

# Output directories for processed data and visual checks
OUTPUT_ROOT_DIR = "PROCESSED_DATA_FOR_SEGMENTATION"
PROCESSED_RGB_DIR = os.path.join(OUTPUT_ROOT_DIR, "RGB_IMAGES")
PROCESSED_BLADE_COORDS_DIR = os.path.join(OUTPUT_ROOT_DIR, "BLADE_COORDS")
PROCESSED_VEIN_COORDS_DIR = os.path.join(OUTPUT_ROOT_DIR, "VEIN_COORDS")
PREPROCESSING_PLOTS_DIR = os.path.join(OUTPUT_ROOT_DIR, "PREPROCESSING_PLOTS")

# Create output directories
for d in [PROCESSED_RGB_DIR, PROCESSED_BLADE_COORDS_DIR, PROCESSED_VEIN_COORDS_DIR, PREPROCESSING_PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Plotting parameters for visual checks
BLADE_PLOT_COLOR = "orange"
VEIN_PLOT_COLOR = "magenta"
OVERLAY_ALPHA = 0.7
PLOT_DPI = 100

# Image standardization parameters
MAX_DIMENSION_CAP = 2048 # Absolute maximum dimension for any image (e.g., for memory limits)
DIMENSION_PERCENTILE = 95 # Use 95th percentile of existing image dimensions for target size

# === HELPERS ===
def read_coords(path):
    """Reads coordinate data from a text file."""
    return np.loadtxt(path)

def resolve_image_path(folder, image_value):
    """Resolves the full path to an image file, handling various extensions."""
    name, ext = os.path.splitext(image_value)
    candidates = [image_value] if ext else [f"{image_value}{e}" for e in IMAGE_EXTENSIONS]
    all_files = os.listdir(folder)
    for candidate in candidates:
        for f in all_files:
            if f.lower() == candidate.lower(): # Case-insensitive match
                return os.path.join(folder, f)
    raise FileNotFoundError(f"No image found for base name '{image_value}' in '{folder}'")

def rotate_to_wide(img, coords):
    """Rotates an image and its coordinates so width >= height."""
    if img.width >= img.height:
        return img, coords
    img_rot = img.transpose(Image.Transpose.ROTATE_90)
    
    # Transform coordinates: (x, y) -> (y, original_width - x)
    coords_rot = np.copy(coords)
    coords_rot[:, 0] = coords[:, 1] # New x is old y
    coords_rot[:, 1] = img.width - coords[:, 0] # New y is original_width - old x
    
    return img_rot, coords_rot

def calculate_target_size(image_sizes, percentile, max_dim_cap):
    """
    Calculates a uniform target size for all images based on a percentile
    of existing dimensions, ensuring dimensions are even and capped.
    """
    widths = [s[0] for s in image_sizes]
    heights = [s[1] for s in image_sizes]

    # Calculate percentile for width and height
    target_w = int(np.percentile(widths, percentile))
    target_h = int(np.percentile(heights, percentile))

    # Ensure dimensions are even
    target_w = target_w if target_w % 2 == 0 else target_w - 1
    target_h = target_h if target_h % 2 == 0 else target_h - 1

    # Apply overall maximum dimension cap
    scale_factor = min(max_dim_cap / max(target_w, target_h), 1.0)
    target_w = int(target_w * scale_factor)
    target_h = int(target_h * scale_factor)

    # Re-ensure even after cap scaling
    target_w = target_w if target_w % 2 == 0 else target_w - 1
    target_h = target_h if target_h % 2 == 0 else target_h - 1

    return (target_w, target_h)

def rescale_and_pad_image(img, target_size):
    """Rescales an image to fit target_size while maintaining aspect ratio, then pads."""
    # Use Image.LANCZOS for high-quality downsampling/upsampling
    img = ImageOps.contain(img, target_size, Image.LANCZOS)
    padded = Image.new("RGB", target_size, (255, 255, 255)) # Pad with white
    paste_pos = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
    padded.paste(img, paste_pos)
    return padded, (img.width, img.height), paste_pos # Return actual pasted size and position

def transform_coords_to_target_size(coords, original_img_size, pasted_img_size, paste_pos):
    """
    Transforms coordinates based on the scaling and padding applied to the image.
    original_img_size: size of the image *after* rotation (width, height)
    pasted_img_size: size of the image *after* ImageOps.contain but *before* padding (width, height)
    paste_pos: (x_offset, y_offset) from ImageOps.contain padding
    """
    scale_x = pasted_img_size[0] / original_img_size[0]
    scale_y = pasted_img_size[1] / original_img_size[1]
    
    coords_scaled = np.copy(coords).astype(float)
    coords_scaled[:, 0] *= scale_x
    coords_scaled[:, 1] *= scale_y
    
    # Add padding offset
    coords_trans = coords_scaled + np.array(paste_pos)
    
    return coords_trans

# === PHASE 1: Determine Target Size and Collect Initial Info ===
info_files = [f for f in os.listdir(RAW_TRAINING_DIR) if f.endswith("_info.csv")]
image_sizes_before_rotation = []
valid_fids_for_target_size = []

print("🔍 Scanning for image dimensions to determine target size...")
for info_file in tqdm(info_files):
    try:
        fid = info_file.replace("_info.csv", "")
        meta = pd.read_csv(os.path.join(RAW_TRAINING_DIR, info_file))
        imgval = meta.loc[meta["factor"] == "image", "value"].values[0]
        imgpath = resolve_image_path(RAW_TRAINING_DIR, imgval)
        
        im = Image.open(imgpath).convert("RGB")
        
        # Store original dimensions (before any rotation)
        image_sizes_before_rotation.append(im.size)
        valid_fids_for_target_size.append(fid)

    except Exception as e:
        print(f"⚠️ Skipping {info_file} for target size calculation: {e}")

if not image_sizes_before_rotation:
    raise RuntimeError("No valid images found to calculate target size. Please check TRAINING_DATA directory.")

target_width, target_height = calculate_target_size(image_sizes_before_rotation, DIMENSION_PERCENTILE, MAX_DIMENSION_CAP)
TARGET_SIZE = (target_width, target_height)
print(f"📐 Calculated Target image size (95th percentile, even dims, max {MAX_DIMENSION_CAP}px): {TARGET_SIZE}")

# === PHASE 2: Process, Transform, and Save All Data ===
print("🚀 Processing, transforming, and saving all data...")
name_counter = {} # To handle duplicate base names (e.g., if different IDs share a name)

for info_file in tqdm(info_files):
    try:
        fid = info_file.replace("_info.csv", "")
        img_info = pd.read_csv(os.path.join(RAW_TRAINING_DIR, info_file))
        imgval = img_info.loc[img_info["factor"] == "image", "value"].values[0]
        imgpath = resolve_image_path(RAW_TRAINING_DIR, imgval)
        blade_path = os.path.join(RAW_TRAINING_DIR, f"{fid}_blade.txt")
        vein_path = os.path.join(RAW_TRAINING_DIR, f"{fid}_veins.txt")

        if not os.path.exists(blade_path) or not os.path.exists(vein_path):
            print(f"⚠️ Skipping {fid}: Missing blade or vein coordinate file(s).")
            continue

        # Handle duplicate filenames (e.g., if different original sample IDs map to same processed base name)
        # This creates a unique output name like "SAMPLE_NAME_1", "SAMPLE_NAME_2"
        name_base = fid.strip()
        count = name_counter.get(name_base, 0) + 1
        name_counter[name_base] = count
        outname = f"{name_base}_{count}" if count > 1 else name_base

        # 1. Load original image and coordinates
        img_orig = Image.open(imgpath).convert("RGB")
        blade_coords_raw = read_coords(blade_path)
        vein_coords_raw = read_coords(vein_path)

        # 2. Rotate to wide format (and transform coordinates)
        img_rot, blade_coords_rot = rotate_to_wide(img_orig, blade_coords_raw)
        _, vein_coords_rot = rotate_to_wide(img_orig, vein_coords_raw)

        # 3. NO SHAPELY FIXING FOR NOW. Use the rotated coordinates directly.
        final_blade_coords = blade_coords_rot
        final_vein_coords = vein_coords_rot
        
        # 4. Rescale and pad image to target size
        img_final, pasted_img_size, paste_pos = rescale_and_pad_image(img_rot, TARGET_SIZE)
        
        # 5. Transform coordinates to final image scale and position
        blade_coords_final = transform_coords_to_target_size(final_blade_coords, img_rot.size, pasted_img_size, paste_pos)
        vein_coords_final = transform_coords_to_target_size(final_vein_coords, img_rot.size, pasted_img_size, paste_pos)

        # 6. Save processed RGB image and transformed coordinates
        img_final.save(os.path.join(PROCESSED_RGB_DIR, outname + ".png"))
        np.savetxt(os.path.join(PROCESSED_BLADE_COORDS_DIR, outname + "_blade.txt"), blade_coords_final, fmt="%.2f")
        np.savetxt(os.path.join(PROCESSED_VEIN_COORDS_DIR, outname + "_veins.txt"), vein_coords_final, fmt="%.2f")

        # 7. Generate visual check plot
        fig, ax = plt.subplots(figsize=(TARGET_SIZE[0] / PLOT_DPI, TARGET_SIZE[1] / PLOT_DPI), dpi=PLOT_DPI)
        ax.imshow(img_final)
        
        # Plot fixed polygons (will use coordinates as they are, without Shapely pre-processing)
        ax.add_patch(Polygon(blade_coords_final, closed=True, color=BLADE_PLOT_COLOR, alpha=OVERLAY_ALPHA))
        ax.add_patch(Polygon(vein_coords_final, closed=True, color=VEIN_PLOT_COLOR, alpha=OVERLAY_ALPHA))
        
        ax.axis("off")
        ax.set_title(f"Processed: {outname}", fontsize=8) # Add title for clarity
        plt.tight_layout(pad=0)
        fig.savefig(os.path.join(PREPROCESSING_PLOTS_DIR, outname + ".png"), dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig) # Explicitly close the figure to free memory

    except Exception as e:
        print(f"❌ Critical error processing {info_file}: {e}")

print(f"\n✅ All files processed. Outputs saved to '{OUTPUT_ROOT_DIR}'.")
print(f"    Processed RGB images: '{PROCESSED_RGB_DIR}'")
print(f"    Processed blade coordinates: '{PROCESSED_BLADE_COORDS_DIR}'")
print(f"    Processed vein coordinates: '{PROCESSED_VEIN_COORDS_DIR}'")
print(f"    Visual check plots: '{PREPROCESSING_PLOTS_DIR}'")