import os
import numpy as np
from PIL import Image, ImageDraw
import re
from tqdm import tqdm
import pandas as pd
import json

# --- CONFIGURATION ---
# Base directory where your data and image folders are located
BASE_DIR = os.getcwd() # Assumes script is run from the directory containing msu_data, algeria_images, etc.

# Data and Image folder pairs
DATASET_PAIRS = {
    "msu": {"data_dir": "msu_data", "images_dir": "msu_images"},
    "ucd": {"data_dir": "ucd_data", "images_dir": "ucd_images"},
    "algeria": {"data_dir": "algeria_data", "images_dir": "algeria_images"},
    "original": {"data_dir": "original_data", "images_dir": "original_images"}, # Assuming 'original' refers to Vinifera/your main dataset
}

# Image and Mask Processing Parameters (from your preprocessing config)
TARGET_WIDTH = 2048
TARGET_HEIGHT = 2040
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG", ".JPEG", ".TIF", ".TIFF", ".PNG"]

# Output directory for foreground assets
OUTPUT_FOREGROUND_DIR = "foreground_library"
os.makedirs(OUTPUT_FOREGROUND_DIR, exist_ok=True)

# --- HELPER FUNCTIONS (Adapted from your previous scripts) ---

def rotate_to_wide(image_pil):
    """Rotates an image so its width is greater than its height."""
    width, height = image_pil.size
    rotation_applied = False
    if height > width:
        image_pil = image_pil.transpose(Image.Transpose.ROTATE_270)
        rotation_applied = True
    return image_pil, rotation_applied

def rescale_and_pad_image(image_pil, target_size):
    """
    Rescales an image to fit within target_size while maintaining aspect ratio,
    then pads with white (for RGB) to reach target_size.
    Returns the padded image, paste offset, scaled dimensions, and scale factor.
    """
    original_width, original_height = image_pil.size
    target_width, target_height = target_size
    
    if original_width == 0 or original_height == 0:
        return Image.new("RGB", target_size, (255, 255, 255)), (0, 0), (0, 0), 0.0

    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale_factor = min(scale_w, scale_h)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    scaled_img = image_pil.resize((new_width, new_height), Image.LANCZOS)

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    padded_img = Image.new("RGB", target_size, (255, 255, 255))
    padded_img.paste(scaled_img, (paste_x, paste_y))
    
    return padded_img, (paste_x, paste_y), (new_width, new_height), scale_factor

def load_coordinates(filepath):
    """Loads 2D coordinates from a text file, handling potential float parsing issues."""
    if not os.path.exists(filepath):
        return None
    try:
        # Load as float, then convert to integer type to handle potential decimals gracefully
        coords = np.loadtxt(filepath, dtype=np.float32).astype(np.int64) 
        if coords.ndim == 1: # Handle single point case (e.g., if only one coordinate pair exists)
            coords = coords.reshape(1, 2)
        # Ensure it's not empty after reshape, as an empty file might still result in some shape
        if coords.size == 0:
            return None
        return coords.tolist() # ImageDraw.polygon expects list of tuples or list of lists
    except Exception as e:
        print(f"  WARNING: Could not load coordinates from {filepath}: {e}")
        return None

def find_image_path(base_images_dir, image_filename):
    """Searches for an image file with various extensions."""
    for ext in IMAGE_EXTENSIONS:
        full_path = os.path.join(base_images_dir, image_filename + ext)
        if os.path.exists(full_path):
            return full_path
        full_path_lower = os.path.join(base_images_dir, image_filename + ext.lower())
        if os.path.exists(full_path_lower): # Check for lowercase extension too
            return full_path_lower
    return None

# --- MAIN FOREGROUND GENERATION LOGIC ---

print("Starting foreground asset generation...")

for dataset_name, paths in DATASET_PAIRS.items():
    data_folder = os.path.join(BASE_DIR, paths["data_dir"])
    images_folder = os.path.join(BASE_DIR, paths["images_dir"])
    
    # Create output subfolder for this dataset
    output_sub_dir = os.path.join(OUTPUT_FOREGROUND_DIR, dataset_name)
    os.makedirs(output_sub_dir, exist_ok=True)
    print(f"\nProcessing dataset: {dataset_name} (data: {data_folder}, images: {images_folder})")

    # Get a list of all info.csv files to iterate through samples
    info_files = [f for f in os.listdir(data_folder) if f.endswith("_info.csv")]

    if not info_files:
        print(f"  No _info.csv files found in {data_folder}. Skipping this dataset.")
        continue

    for info_filename in tqdm(info_files, desc=f"  Generating assets for {dataset_name}"):
        base_fid = os.path.splitext(info_filename)[0].replace("_info", "") # Extract base file ID
        
        info_filepath = os.path.join(data_folder, info_filename)
        
        # Try both _blade.txt and _blades.txt
        blade_filepath_option1 = os.path.join(data_folder, f"{base_fid}_blade.txt")
        blade_filepath_option2 = os.path.join(data_folder, f"{base_fid}_blades.txt")

        # Try both _vein.txt and _veins.txt
        vein_filepath_option1 = os.path.join(data_folder, f"{base_fid}_vein.txt")
        vein_filepath_option2 = os.path.join(data_folder, f"{base_fid}_veins.txt")

        # 1. Read image filename from info.csv using pandas
        image_name_from_csv = None
        try:
            # Try reading with tab separator first
            df = pd.read_csv(info_filepath, sep='\t')
            # If 'factor' and 'value' columns are not found, try without specifying separator (comma default)
            if "factor" not in df.columns or "value" not in df.columns:
                df = pd.read_csv(info_filepath) # Fallback to comma or auto-detect if tab failed
            
            if "factor" in df.columns and "value" in df.columns:
                image_name_row = df[df["factor"] == "image"]
                if not image_name_row.empty:
                    image_name_from_csv = image_name_row["value"].iloc[0]
            
        except Exception as e:
            print(f"  ERROR: Could not parse {info_filepath} for image filename: {e}. Skipping {base_fid}.")
            continue

        if image_name_from_csv is None:
            print(f"  ERROR: Image filename 'factor' not found or 'value' is empty in {info_filepath}. Skipping {base_fid}.")
            continue
        
        # Remove extension from image_name_from_csv for robust path finding
        image_filename_no_ext = os.path.splitext(image_name_from_csv)[0]
        
        # Find the actual image file path
        original_image_path = find_image_path(images_folder, image_filename_no_ext)

        if original_image_path is None:
            print(f"  ERROR: Image file not found for '{image_name_from_csv}' (derived from {info_filepath}) in '{images_folder}'. Skipping {base_fid}.")
            continue

        # 2. Load coordinates
        blade_coords = None
        if os.path.exists(blade_filepath_option1):
            blade_coords = load_coordinates(blade_filepath_option1)
        elif os.path.exists(blade_filepath_option2):
            blade_coords = load_coordinates(blade_filepath_option2)

        vein_coords = None
        if os.path.exists(vein_filepath_option1):
            vein_coords = load_coordinates(vein_filepath_option1)
        elif os.path.exists(vein_filepath_option2):
            vein_coords = load_coordinates(vein_filepath_option2)

        if blade_coords is None or vein_coords is None:
            print(f"  ERROR: Missing or unreadable blade/vein coordinates for {base_fid}. Skipping.")
            continue
        
        # Construct output paths
        output_foreground_path = os.path.join(output_sub_dir, f"{base_fid}_foreground.png")
        output_blade_mask_path = os.path.join(output_sub_dir, f"{base_fid}_blade_mask.png")
        output_vein_mask_path = os.path.join(output_sub_dir, f"{base_fid}_vein_mask.png")
        output_metadata_path = os.path.join(output_sub_dir, f"{base_fid}_metadata.json") # New metadata file

        # Skip if outputs already exist (useful for resuming runs)
        if os.path.exists(output_foreground_path) and \
           os.path.exists(output_blade_mask_path) and \
           os.path.exists(output_vein_mask_path) and \
           os.path.exists(output_metadata_path):
            # tqdm handles progress, no need for verbose skipping unless debugging
            continue

        try:
            # 3. Load and preprocess image
            original_pil_img = Image.open(original_image_path).convert("RGB")
            
            img_pil_preprocessed_rot, rotation_applied = rotate_to_wide(original_pil_img.copy())
            padded_img, (paste_x, paste_y), (new_width, new_height), scale_factor = \
                rescale_and_pad_image(img_pil_preprocessed_rot, TARGET_SIZE)

            # 4. Transform coordinates
            # Get original dimensions for coordinate transformation during rotation
            original_img_width, original_img_height = original_pil_img.size

            # Apply rotation to original coordinates
            transformed_blade_coords_pre_crop = []
            if rotation_applied: # if original was portrait, it was rotated 270 degrees
                for x, y in blade_coords:
                    transformed_blade_coords_pre_crop.append((original_img_height - 1 - y, x))
            else:
                transformed_blade_coords_pre_crop = list(blade_coords) # Make a copy

            transformed_vein_coords_pre_crop = []
            if rotation_applied:
                for x, y in vein_coords:
                    transformed_vein_coords_pre_crop.append((original_img_height - 1 - y, x))
            else:
                transformed_vein_coords_pre_crop = list(vein_coords) # Make a copy

            # Now, apply scaling and padding offsets to the already rotated coordinates
            # Store these as "transformed_X_coords" in metadata before the final crop
            final_transformed_blade_coords = [(int(x * scale_factor) + paste_x, int(y * scale_factor) + paste_y)
                                        for x, y in transformed_blade_coords_pre_crop]
            final_transformed_vein_coords = [(int(x * scale_factor) + paste_x, int(y * scale_factor) + paste_y)
                                       for x, y in transformed_vein_coords_pre_crop]
            
            # Calculate geodesic origin if vein coordinates exist and have at least 2 points
            geodesic_origin = None
            if final_transformed_vein_coords and len(final_transformed_vein_coords) >= 2:
                first_point = np.array(final_transformed_vein_coords[0])
                last_point = np.array(final_transformed_vein_coords[-1])
                geodesic_origin = ((first_point + last_point) / 2).astype(int).tolist() # Convert to int and list
            
            # 5. Generate Masks
            blade_mask_pil = Image.new('L', TARGET_SIZE, 0) # L mode for grayscale (binary)
            vein_mask_pil = Image.new('L', TARGET_SIZE, 0)

            draw_blade = ImageDraw.Draw(blade_mask_pil)
            draw_vein = ImageDraw.Draw(vein_mask_pil)

            if final_transformed_blade_coords:
                draw_blade.polygon(final_transformed_blade_coords, fill=1) # Fill with 1 for binary mask
            if final_transformed_vein_coords:
                draw_vein.polygon(final_transformed_vein_coords, fill=1) # Fill with 1

            # 6. Extract Foreground RGB Image (with transparent background)
            # Start with a transparent canvas
            foreground_rgba = Image.new('RGBA', TARGET_SIZE, (0, 0, 0, 0))
            padded_img_rgba = padded_img.convert('RGBA')

            blade_mask_np = np.array(blade_mask_pil)
            
            # Copy pixels from the padded RGB image where the blade mask is positive
            # and set their alpha to opaque.
            fg_data = np.array(foreground_rgba)
            padded_data = np.array(padded_img_rgba)
            
            # Find indices where blade_mask_np is 1 (or any non-zero value if you used other values)
            y_coords, x_coords = np.where(blade_mask_np > 0)
            
            # Copy RGB values and set alpha to 255 for these pixels
            fg_data[y_coords, x_coords, :3] = padded_data[y_coords, x_coords, :3]
            fg_data[y_coords, x_coords, 3] = 255 # Set alpha to fully opaque

            final_foreground_img = Image.fromarray(fg_data, mode='RGBA')

            # --- CROPPING TO BOUNDING BOX ---
            # Find the bounding box of the non-transparent pixels (i.e., the leaf content)
            alpha_channel = np.array(final_foreground_img.getchannel('A'))
            non_transparent_pixels = np.where(alpha_channel > 0)

            if non_transparent_pixels[0].size == 0:
                # No leaf content found, skip this sample after logging
                print(f"  WARNING: No leaf content found after masking for {base_fid}. Skipping sample.")
                continue

            min_y, max_y = np.min(non_transparent_pixels[0]), np.max(non_transparent_pixels[0])
            min_x, max_x = np.min(non_transparent_pixels[1]), np.max(non_transparent_pixels[1])

            # Crop all three assets (RGB foreground, blade mask, vein mask)
            cropped_foreground_img = final_foreground_img.crop((min_x, min_y, max_x + 1, max_y + 1))
            cropped_blade_mask = blade_mask_pil.crop((min_x, min_y, max_x + 1, max_y + 1))
            cropped_vein_mask = vein_mask_pil.crop((min_x, min_y, max_x + 1, max_y + 1))

            # Store cropped dimensions
            cropped_width, cropped_height = cropped_foreground_img.size
            metadata = {
                "original_filename": image_name_from_csv,
                "target_size_padded_width": TARGET_WIDTH,
                "target_size_padded_height": TARGET_HEIGHT,
                "cropped_width": cropped_width,
                "cropped_height": cropped_height,
                "crop_bbox_min_x": int(min_x),
                "crop_bbox_min_y": int(min_y),
                "crop_bbox_max_x": int(max_x),
                "crop_bbox_max_y": int(max_y),
                "transformed_blade_coords": final_transformed_blade_coords, # Added
                "transformed_vein_coords": final_transformed_vein_coords,   # Added
                "geodesic_origin_coords": geodesic_origin                   # Added
            }

            # 7. Save Assets
            cropped_foreground_img.save(output_foreground_path)
            cropped_blade_mask.save(output_blade_mask_path)
            cropped_vein_mask.save(output_vein_mask_path)
            
            with open(output_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
        except Exception as e:
            print(f"  ❌ ERROR processing {base_fid}: {e}. Skipping this sample.")
            # Clean up any partially created files if an error occurs mid-way
            for p in [output_foreground_path, output_blade_mask_path, output_vein_mask_path, output_metadata_path]:
                if os.path.exists(p):
                    os.remove(p)
            continue

print("\n--- Foreground asset generation complete! ---")
print(f"Check the '{OUTPUT_FOREGROUND_DIR}' folder for generated assets.")