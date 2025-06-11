# vinifera_preprocessing_stage2_7channel_masks.py

# === IMPORTS ===
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import color, filters
from tqdm import tqdm
import json
import re # For regular expressions to get file IDs

# === CONFIGURATION ===
# --- Input and Output Directories ---
# Original raw data folder with _blade.txt, _veins.txt
RAW_DATA_DIR = "original_data" 

# Your manually cleaned images (from original 15 leaves)
MODIFIED_BACKGROUNDS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA/MODIFIED_BACKGROUNDS" 

# NEW: Path to your additional data folder with raw .jpg, _blade.txt, _veins.txt
ADDITIONAL_DATA_DIR = "additional_data" 

OUTPUT_ROOT_DIR = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA"
SEVEN_CHANNEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "7_CHANNEL_INPUTS")
BLADE_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "BLADE_MASKS")
VEIN_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "VEIN_MASKS")
CONFIG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "config") # To save preprocessing config

# Create output directories
for d in [SEVEN_CHANNEL_OUTPUT_DIR, BLADE_MASKS_OUTPUT_DIR, VEIN_MASKS_OUTPUT_DIR, CONFIG_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Image and Mask Processing Parameters ---
TARGET_WIDTH = 2048 
TARGET_HEIGHT = 2040 
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG", ".JPEG", ".TIF", ".TIFF", ".PNG"]

# --- Ridge Filter Parameters (Optimized for Vinifera) ---
sato_sigmas = [0.5, 1.0, 1.5, 2.0, 4.0]
meijering_sigmas = [0.5, 1.0, 1.5, 2.0, 4.0]
frangi_sigmas = [0.5, 1.0, 1.5, 2.0, 4.0]
hessian_sigmas = [0.5, 1.0, 1.5, 2.0, 4.0]

# --- Contrast Enhancement Parameter ---
ENHANCE_PERCENTILE = 99.0

# --- Save preprocessing configuration for future reference ---
PREPROCESSING_CONFIG = {
    "TARGET_SIZE": TARGET_SIZE,
    "sato_sigmas": sato_sigmas,
    "meijering_sigmas": meijering_sigmas,
    "frangi_sigmas": frangi_sigmas,
    "hessian_sigmas": hessian_sigmas,
    "ENHANCE_PERCENTILE": ENHANCE_PERCENTILE,
    "RAW_DATA_DIR": RAW_DATA_DIR, # Added for clarity
    "MODIFIED_BACKGROUNDS_DIR": MODIFIED_BACKGROUNDS_DIR,
    "ADDITIONAL_DATA_DIR": ADDITIONAL_DATA_DIR, # Added for clarity
    "SEVEN_CHANNEL_OUTPUT_DIR": SEVEN_CHANNEL_OUTPUT_DIR,
    "BLADE_MASKS_OUTPUT_DIR": BLADE_MASKS_OUTPUT_DIR,
    "VEIN_MASKS_OUTPUT_DIR": VEIN_MASKS_OUTPUT_DIR
}
with open(os.path.join(CONFIG_OUTPUT_DIR, "preprocessing_config.json"), 'w') as f:
    json.dump(PREPROCESSING_CONFIG, f, indent=4)
print(f"Preprocessing configuration saved to: {os.path.join(CONFIG_OUTPUT_DIR, 'preprocessing_config.json')}")


# === HELPER FUNCTIONS ===
def read_coords(path):
    """Reads coordinate data from a text file."""
    return np.loadtxt(path)

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
    then pads with white (for RGB) or black (for L) to reach target_size.
    """
    original_width, original_height = image_pil.size
    target_width, target_height = target_size
    
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale_factor = min(scale_w, scale_h)

    if original_width == 0 or original_height == 0:
        new_width = 0
        new_height = 0
    else:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

    scaled_img = image_pil.resize((new_width, new_height), Image.LANCZOS)

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    if image_pil.mode == 'L':
        padded_img = Image.new("L", target_size, 0) # Use 0 for mask background (black)
    else: # Assume RGB
        padded_img = Image.new("RGB", target_size, (255, 255, 255)) # Use white for RGB background
        
    padded_img.paste(scaled_img, (paste_x, paste_y))
    
    return padded_img, (paste_x, paste_y), (new_width, new_height) # Return padded image and paste offset

def enhance_contrast(arr, percentile_val):
    """Applies contrast enhancement based on percentile."""
    vmax = np.percentile(arr, percentile_val)
    if vmax == 0: 
        return np.zeros_like(arr, dtype=np.float32)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled.astype(np.float32)

def apply_ridge_filters(image_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p):
    """
    Applies various ridge filters to a grayscale image and returns their enhanced outputs.
    Takes a PIL Image that is already padded to TARGET_SIZE.
    """
    image_rgb_float = np.array(image_pil_padded).astype(np.float32) / 255.0
    
    if image_rgb_float.ndim == 3 and image_rgb_float.shape[2] == 3:
        gray_image = color.rgb2gray(image_rgb_float)
    elif image_rgb_float.ndim == 2:
        gray_image = image_rgb_float
    else:
        raise ValueError(f"Unexpected image dimensions or mode for filter application: {image_rgb_float.shape}")

    sato_raw = filters.sato(gray_image, sigmas=sato_s, black_ridges=False, mode='reflect')
    meijering_raw = filters.meijering(gray_image, sigmas=meijering_s, black_ridges=False, mode='reflect')
    frangi_raw = filters.frangi(gray_image, sigmas=frangi_s, black_ridges=False, mode='reflect')
    hessian_raw = filters.hessian(gray_image, sigmas=hessian_s, black_ridges=True, mode='reflect')

    sato_processed = enhance_contrast(sato_raw, enhance_p)
    meijering_processed = enhance_contrast(meijering_raw, enhance_p)
    frangi_processed = enhance_contrast(frangi_raw, enhance_p)
    hessian_processed = enhance_contrast(hessian_raw, enhance_p)

    return sato_processed, meijering_processed, frangi_processed, hessian_processed

def create_7channel_input(img_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p):
    """
    Creates the 7-channel input array for the UNet model.
    Channels: L, A, B, Sato, Meijering, Frangi, Hessian.
    """
    img_rgb_float = np.array(img_pil_padded).astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_rgb_float)

    L_channel = img_lab[:, :, 0] / 100.0 
    A_channel = (img_lab[:, :, 1] + 128) / 255.0 
    B_channel = (img_lab[:, :, 2] + 128) / 255.0 

    sato_ch, meijering_ch, frangi_ch, hessian_ch = \
        apply_ridge_filters(img_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p)

    seven_channels = np.stack([L_channel, A_channel, B_channel,
                               sato_ch, meijering_ch, frangi_ch, hessian_ch], axis=-1)
    
    return seven_channels

def create_mask_from_coords(coords, target_size, original_size_before_rotation, rotation_applied, paste_offset, scaled_dims):
    """
    Creates a binary mask from polygon coordinates, handling original image dimensions
    and scaling/padding to target_size.
    
    Args:
        coords (np.array): Nx2 array of (x, y) coordinates for the polygon.
        target_size (tuple): (width, height) of the final mask.
        original_size_before_rotation (tuple): (width, height) of the original image before rotation.
        rotation_applied (bool): True if rotate_to_wide was applied.
        paste_offset (tuple): (x_offset, y_offset) from rescale_and_pad_image.
        scaled_dims (tuple): (width, height) of the image after scaling but before padding.
            
    Returns:
        np.array: Binary mask (0s and 1s) of target_size.
    """
    mask = Image.new("L", target_size, 0) # Black background
    draw = ImageDraw.Draw(mask)

    if coords.size == 0:
        return np.array(mask) # Return empty mask if no coordinates

    # Determine original size *after* rotation (which affects scaling factor calculation)
    if rotation_applied:
        # If rotated, effective dimensions for scaling calculations are swapped
        rotated_original_width, rotated_original_height = original_size_before_rotation[1], original_size_before_rotation[0]
    else:
        rotated_original_width, rotated_original_height = original_size_before_rotation[0], original_size_before_rotation[1]

    # Calculate scaling factor based on the *rotated* original size and the *scaled dimensions* after resize
    # This factor ensures consistency with the image resizing
    scale_factor_x = scaled_dims[0] / rotated_original_width if rotated_original_width > 0 else 0
    scale_factor_y = scaled_dims[1] / rotated_original_height if rotated_original_height > 0 else 0
    # In rescale_and_pad_image, a single min(scale_w, scale_h) is used for both dimensions.
    # So, scale_factor_x and scale_factor_y should be effectively the same here.
    # We can just use the overall scale factor that was applied.
    # Let's derive it from the scaled_dims and original_size_after_rotation for robustness.
    
    # We need to compute the scaling factor that was applied during rescale_and_pad_image
    # This can be found from the scaled_dims and the original dimensions *after* rotation
    original_dims_after_rot = (rotated_original_width, rotated_original_height)
    scale_factor = min(target_size[0] / original_dims_after_rot[0], target_size[1] / original_dims_after_rot[1]) if original_dims_after_rot[0] > 0 and original_dims_after_rot[1] > 0 else 0


    # Transform coordinates to the scaled and padded target size
    transformed_coords = []
    for x, y in coords:
        # Scale coordinates based on the overall scale factor
        scaled_x = x * scale_factor
        scaled_y = y * scale_factor
        
        # Apply paste offset
        transformed_x = scaled_x + paste_offset[0]
        transformed_y = scaled_y + paste_offset[1]
        transformed_coords.append((transformed_x, transformed_y))
    
    draw.polygon([tuple(p) for p in transformed_coords], fill=1)
    
    return np.array(mask)

# --- Consolidated Processing Function ---
def process_data_source(image_dir, trace_dir, description):
    """
    Processes images and generates 7-channel inputs and masks from a given source.
    Handles both raw JPGs (for new data) and pre-modified PNGs (for existing data).
    """
    print(f"\n--- Processing {description} ---")
    
    # Get all file IDs from both image and trace files in this source
    fids = set()
    for root, _, files in os.walk(image_dir): # Check image directory for FIDs
        for f in files:
            if any(f.endswith(ext) for ext in IMAGE_EXTENSIONS):
                fids.add(os.path.splitext(f)[0])
    for root, _, files in os.walk(trace_dir): # Check trace directory for FIDs
        for f in files:
            match = re.match(r'(.+?)(_blade|_veins)?\.txt', f)
            if match:
                fids.add(match.group(1))
    
    fids = sorted(list(fids))

    if not fids:
        print(f"⚠️ No relevant files found in '{image_dir}' or '{trace_dir}'. Skipping this source.")
        return

    for fid in tqdm(fids, desc=f"Processing {description}"):
        seven_channel_output_path = os.path.join(SEVEN_CHANNEL_OUTPUT_DIR, f"{fid}.npy")
        blade_mask_output_path = os.path.join(BLADE_MASKS_OUTPUT_DIR, f"{fid}_blade_mask.npy")
        vein_mask_output_path = os.path.join(VEIN_MASKS_OUTPUT_DIR, f"{fid}_vein_mask.npy")

        # Skip if already processed (for all outputs)
        if (os.path.exists(seven_channel_output_path) and
            os.path.exists(blade_mask_output_path) and
            os.path.exists(vein_mask_output_path)):
            # print(f"Skipping {fid}: All outputs already exist.") # Uncomment for verbose skipping
            continue
        
        try:
            # --- 1. Load Image (from appropriate source) ---
            img_pil = None
            original_img_path = None # Will store the actual path to the raw JPG/PNG

            # Try to load from image_dir (e.g., MODIFIED_BACKGROUNDS_DIR or ADDITIONAL_DATA_DIR for JPGs)
            for ext in IMAGE_EXTENSIONS:
                temp_path = os.path.join(image_dir, f"{fid}{ext}")
                if os.path.exists(temp_path):
                    original_img_path = temp_path
                    break
            
            if original_img_path is None:
                print(f"⚠️ Skipping {fid}: No suitable image file found in '{image_dir}'.")
                continue

            img_pil = Image.open(original_img_path).convert("RGB")
            original_img_size_before_rotation = img_pil.size # Store original size (W, H)

            # Apply initial rotation
            img_pil_preprocessed_rot, rotation_applied = rotate_to_wide(img_pil)
            
            # Rescale and pad the image
            img_pil_padded, paste_offset, scaled_dims = rescale_and_pad_image(img_pil_preprocessed_rot, TARGET_SIZE)
            
            # Generate 7-channel input
            seven_channel_data = create_7channel_input(
                img_pil_padded, sato_sigmas, meijering_sigmas, frangi_sigmas, hessian_sigmas, ENHANCE_PERCENTILE
            )
            
            # Save 7-channel data (as .npy file)
            np.save(seven_channel_output_path, seven_channel_data)

            # --- 2. Generate and save masks ---
            blade_coords_path = os.path.join(trace_dir, f"{fid}_blade.txt")
            vein_coords_path = os.path.join(trace_dir, f"{fid}_veins.txt")

            if not os.path.exists(blade_coords_path):
                print(f"⚠️ Skipping mask generation for {fid}: Blade coordinates file missing in '{trace_dir}'.")
                continue
            if not os.path.exists(vein_coords_path):
                print(f"⚠️ Skipping mask generation for {fid}: Vein coordinates file missing in '{trace_dir}'.")
                continue

            blade_coords = read_coords(blade_coords_path)
            vein_coords = read_coords(vein_coords_path)

            # Create and save blade mask
            blade_mask = create_mask_from_coords(blade_coords, TARGET_SIZE, original_img_size_before_rotation, rotation_applied, paste_offset, scaled_dims)
            np.save(blade_mask_output_path, blade_mask)

            # Create and save vein mask
            vein_mask = create_mask_from_coords(vein_coords, TARGET_SIZE, original_img_size_before_rotation, rotation_applied, paste_offset, scaled_dims)
            np.save(vein_mask_output_path, vein_mask)
            
        except Exception as e:
            print(f"❌ Error processing {fid} from '{description}': {e}. Skipping this image.")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"--- Starting Vinifera Preprocessing ---")
    print(f"Output root: {OUTPUT_ROOT_DIR}")

    # Process original data (using modified backgrounds for images, raw_data_vinifera for traces)
    process_data_source(MODIFIED_BACKGROUNDS_DIR, RAW_DATA_DIR, "Original Data (Modified Backgrounds)")

    # Process additional data (using additional_data for both images and traces)
    # Check if the directory exists before attempting to process
    if os.path.exists(ADDITIONAL_DATA_DIR):
        process_data_source(ADDITIONAL_DATA_DIR, ADDITIONAL_DATA_DIR, "Additional Data (Raw Images)")
    else:
        print(f"Warning: Additional data directory '{ADDITIONAL_DATA_DIR}' not found. Skipping new data processing.")

    print(f"\n--- Vinifera Preprocessing Complete ---")
    print(f"Generated 7-channel inputs and masks are ready for transfer learning!")