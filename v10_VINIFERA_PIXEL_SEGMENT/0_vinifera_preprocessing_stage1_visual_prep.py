# vinifera_preprocessing_stage1_visual_prep.py

# === IMPORTS ===
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm

# === CONFIGURATION ===
# Path to your original data folder
RAW_DATA_DIR = "original_data" # <--- MAKE SURE THIS PATH IS CORRECT

# Comprehensive list of image extensions to search for
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG", ".JPEG", ".TIF", ".TIFF", ".PNG"]

# Output directories
OUTPUT_ROOT_DIR = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA" # Changed root to distinguish from Algeria
ORIGINAL_DATA_PLOTS_DIR = os.path.join(OUTPUT_ROOT_DIR, "ORIGINAL_DATA_PLOTS")
MODIFIED_BACKGROUNDS_DIR = os.path.join(OUTPUT_ROOT_DIR, "MODIFIED_BACKGROUNDS")

# Create output directories
for d in [OUTPUT_ROOT_DIR, ORIGINAL_DATA_PLOTS_DIR, MODIFIED_BACKGROUNDS_DIR]:
    os.makedirs(d, exist_ok=True)

# Plotting parameters
BLADE_PLOT_COLOR = "orange"
VEIN_PLOT_COLOR = "magenta"
OVERLAY_ALPHA = 0.7
PLOT_DPI = 150 # Higher DPI for clearer plots

# === HELPERS ===
def read_coords(path):
    """Reads coordinate data from a text file."""
    return np.loadtxt(path)

def resolve_image_path(folder, image_value, fid_basename, image_exts):
    """
    Resolves the full path to an image file, handling various extensions
    and discrepancies between _info.csv value and annotation base name.
    """
    # Try finding the image exactly as specified in _info.csv
    candidate_path_exact = os.path.join(folder, image_value)
    if os.path.exists(candidate_path_exact):
        return candidate_path_exact, os.path.splitext(image_value)[1] # Return exact path and its actual extension

    # If not found, try matching the info.csv base name with common extensions
    name_from_info, ext_from_info = os.path.splitext(image_value)
    if not ext_from_info: # If image_value in info.csv has no extension, try common ones
        for ext in image_exts:
            candidate_path = os.path.join(folder, f"{name_from_info}{ext}")
            if os.path.exists(candidate_path):
                print(f"  --> Found image for FID '{fid_basename}' using info.csv base name '{name_from_info}' with extension '{ext}'.")
                return candidate_path, ext
    
    # If still not found, try matching the FID base name with common extensions
    for ext in image_exts:
        candidate_path = os.path.join(folder, f"{fid_basename}{ext}")
        if os.path.exists(candidate_path):
            print(f"  --> Found image for FID '{fid_basename}' using annotation base name '{fid_basename}' with extension '{ext}' (Info.csv value was '{image_value}').")
            return candidate_path, ext

    raise FileNotFoundError(f"No image found for '{image_value}' (from info.csv) or '{fid_basename}' (from annotations) in '{folder}'.")


# === MAIN PROCESSING LOGIC ===
print(f"🔍 Starting Vinifera preprocessing stage 1: Visual preparation...")
print(f"   Looking for data in: {RAW_DATA_DIR}")
print(f"   Plots will be saved to: {ORIGINAL_DATA_PLOTS_DIR}")
print(f"   Images for background modification will be saved to: {MODIFIED_BACKGROUNDS_DIR}")

info_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith("_info.csv")]

if not info_files:
    print(f"⚠️ No '_info.csv' files found in '{RAW_DATA_DIR}'. Please check the path and file naming.")
    exit()

for info_file in tqdm(info_files, desc="Processing Vinifera images"):
    fid = info_file.replace("_info.csv", "") # This is our base ID for output files
    
    try:
        # Read info.csv to get the image filename
        info_path = os.path.join(RAW_DATA_DIR, info_file)
        meta = pd.read_csv(info_path)
        
        # Ensure 'image' factor exists
        if 'image' not in meta['factor'].values:
            print(f"⚠️ Skipping {info_file}: 'image' factor not found in _info.csv. Skipping.")
            continue
            
        img_value_from_info = meta.loc[meta["factor"] == "image", "value"].values[0]
        
        # Resolve image path and get its actual extension
        image_path, actual_img_ext = resolve_image_path(RAW_DATA_DIR, img_value_from_info, fid, IMAGE_EXTENSIONS)
        
        # Load image
        img_orig = Image.open(image_path).convert("RGB") # Ensure RGB for plotting
        
        # Load blade and vein coordinates
        blade_path = os.path.join(RAW_DATA_DIR, f"{fid}_blade.txt")
        vein_path = os.path.join(RAW_DATA_DIR, f"{fid}_veins.txt")

        if not os.path.exists(blade_path):
            print(f"⚠️ Skipping {fid}: Missing blade coordinate file at '{blade_path}'.")
            continue
        if not os.path.exists(vein_path):
            print(f"⚠️ Skipping {fid}: Missing vein coordinate file at '{vein_path}'.")
            continue
            
        blade_coords = read_coords(blade_path)
        vein_coords = read_coords(vein_path)

        # === 1. Save original image for background modification ===
        # Use the FID as the base name for the image, saved as PNG for consistency and quality
        output_image_filename = f"{fid}.png" # User will modify this PNG
        img_orig.save(os.path.join(MODIFIED_BACKGROUNDS_DIR, output_image_filename))

        # === 2. Generate and save plot of original image with annotations ===
        fig, ax = plt.subplots(figsize=(img_orig.width / PLOT_DPI, img_orig.height / PLOT_DPI), dpi=PLOT_DPI)
        ax.imshow(img_orig)
        
        # Plot blade polygon
        if blade_coords.size > 0:
            ax.add_patch(Polygon(blade_coords, closed=True, color=BLADE_PLOT_COLOR, alpha=OVERLAY_ALPHA))
        
        # Plot vein polygons
        if vein_coords.size > 0:
            ax.add_patch(Polygon(vein_coords, closed=True, color=VEIN_PLOT_COLOR, alpha=OVERLAY_ALPHA))
        
        ax.axis("off")
        ax.set_title(f"Original: {fid}", fontsize=10)
        plt.tight_layout(pad=0)
        
        # Save plot using the FID as filename for easy matching
        output_plot_filename = f"{fid}_plot.png" # Added _plot to distinguish from image output
        fig.savefig(os.path.join(ORIGINAL_DATA_PLOTS_DIR, output_plot_filename), dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig) # Close the figure to free memory

    except FileNotFoundError as e:
        print(f"❌ Error for {fid}: {e}. Skipping this entry.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing {fid}: {e}. Skipping this entry.")

print(f"\n✅ Vinifera preprocessing stage 1 complete.")
print(f"   Plots saved to '{ORIGINAL_DATA_PLOTS_DIR}'")
print(f"   Original images for modification saved to '{MODIFIED_BACKGROUNDS_DIR}'")