# vinifera_preprocessing_stage2_7channel_masks.py

# === IMPORTS ===
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import color, filters
from tqdm import tqdm
import json # For saving metadata

# === CONFIGURATION ===
# --- Input and Output Directories ---
RAW_DATA_DIR = "original_data" # Path to your original data folder with _info.csv, _blade.txt, etc.
MODIFIED_BACKGROUNDS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA/MODIFIED_BACKGROUNDS" # Your manually cleaned images

OUTPUT_ROOT_DIR = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA"
SEVEN_CHANNEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "7_CHANNEL_INPUTS")
BLADE_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "BLADE_MASKS")
VEIN_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "VEIN_MASKS")
CONFIG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "config") # To save preprocessing config

# Create output directories
for d in [SEVEN_CHANNEL_OUTPUT_DIR, BLADE_MASKS_OUTPUT_DIR, VEIN_MASKS_OUTPUT_DIR, CONFIG_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Image and Mask Processing Parameters ---
# Target size for all processed images and masks
# Ensure these match the TARGET_SIZE you intend for your UNet training
TARGET_WIDTH = 2048 # <--- VERIFY THIS. Use same as Algeria model if applicable.
TARGET_HEIGHT = 2040 # <--- VERIFY THIS. Use same as Algeria model if applicable.
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# Image extensions to search for in original_data folder
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG", ".JPEG", ".TIF", ".TIFF", ".PNG"]

# --- Ridge Filter Parameters (Optimized for Vinifera) ---
# These can be a single float (e.g., [1.0]) or a list of floats (e.g., [1.0, 2.0, 3.0]).
# If a list is provided, skimage takes the maximum response across all scales.
sato_sigmas = [0.5, 1.0, 1.5, 2.0, 4.0]
meijering_sigmas = [0.5, 1.0, 1.5, 2.0, 4.0]
frangi_sigmas = [0.5, 1.0, 1.5, 2.0, 4.0]
hessian_sigmas = [0.5, 1.0, 1.5, 2.0, 4.0]

# --- Contrast Enhancement Parameter ---
# Percentile to clip and normalize image values for contrast enhancement
ENHANCE_PERCENTILE = 99.0

# --- Save preprocessing configuration for future reference ---
PREPROCESSING_CONFIG = {
    "TARGET_SIZE": TARGET_SIZE,
    "sato_sigmas": sato_sigmas,
    "meijering_sigmas": meijering_sigmas,
    "frangi_sigmas": frangi_sigmas,
    "hessian_sigmas": hessian_sigmas,
    "ENHANCE_PERCENTILE": ENHANCE_PERCENTILE,
    "MODIFIED_BACKGROUNDS_DIR": MODIFIED_BACKGROUNDS_DIR,
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
    if height > width:
        image_pil = image_pil.transpose(Image.Transpose.ROTATE_270)
    return image_pil

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

    if original_width == 0 or original_height == 0: # Handle empty or invalid image size
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
    
    return padded_img

def enhance_contrast(arr, percentile_val):
    """Applies contrast enhancement based on percentile."""
    vmax = np.percentile(arr, percentile_val)
    if vmax == 0: # Avoid division by zero if all values are zero
        return np.zeros_like(arr, dtype=np.float32)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled.astype(np.float32) # Ensure float32 for consistency

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

    # Extract LAB channels
    L_channel = img_lab[:, :, 0] / 100.0 # L channel is 0-100, normalize to 0-1
    A_channel = (img_lab[:, :, 1] + 128) / 255.0 # A channel is -128 to 127, normalize to 0-1
    B_channel = (img_lab[:, :, 2] + 128) / 255.0 # B channel is -128 to 127, normalize to 0-1

    # Apply ridge filters
    sato_ch, meijering_ch, frangi_ch, hessian_ch = \
        apply_ridge_filters(img_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p)

    # Stack channels
    seven_channels = np.stack([L_channel, A_channel, B_channel,
                               sato_ch, meijering_ch, frangi_ch, hessian_ch], axis=-1)
    
    return seven_channels

def create_mask_from_coords(coords, target_size, original_size, rotation_applied, paste_offset):
    """
    Creates a binary mask from polygon coordinates, handling original image dimensions
    and scaling/padding to target_size.
    
    Args:
        coords (np.array): Nx2 array of (x, y) coordinates for the polygon.
        target_size (tuple): (width, height) of the final mask.
        original_size (tuple): (width, height) of the original image *before* rotation.
        rotation_applied (bool): True if rotate_to_wide was applied.
        paste_offset (tuple): (x_offset, y_offset) from rescale_and_pad_image.
        
    Returns:
        np.array: Binary mask (0s and 1s) of target_size.
    """
    mask = Image.new("L", target_size, 0) # Black background
    draw = ImageDraw.Draw(mask)

    if coords.size == 0:
        return np.array(mask) # Return empty mask if no coordinates

    # Determine original size *after* rotation (which affects scaling)
    if rotation_applied:
        # If rotated, width and height of image_pil_preprocessed were original_height, original_width
        rotated_original_width, rotated_original_height = original_size[1], original_size[0]
    else:
        rotated_original_width, rotated_original_height = original_size[0], original_size[1]


    # Calculate scaling factor based on the *rotated* original size and target_size
    scale_w = target_size[0] / rotated_original_width
    scale_h = target_size[1] / rotated_original_height
    scale_factor = min(scale_w, scale_h)

    # Transform coordinates to the scaled and padded target size
    transformed_coords = []
    for x, y in coords:
        # Scale coordinates
        scaled_x = x * scale_factor
        scaled_y = y * scale_factor
        
        # Apply paste offset
        transformed_x = scaled_x + paste_offset[0]
        transformed_y = scaled_y + paste_offset[1]
        transformed_coords.append((transformed_x, transformed_y))
    
    # Draw polygon on the mask
    draw.polygon([tuple(p) for p in transformed_coords], fill=1) # Fill with 1 for the object
    
    return np.array(mask)


# === MAIN PROCESSING LOGIC ===
print(f"--- Starting Vinifera Preprocessing Stage 2 ---")
print(f"Loading images from: {MODIFIED_BACKGROUNDS_DIR}")
print(f"Saving 7-channel inputs to: {SEVEN_CHANNEL_OUTPUT_DIR}")
print(f"Saving blade masks to: {BLADE_MASKS_OUTPUT_DIR}")
print(f"Saving vein masks to: {VEIN_MASKS_OUTPUT_DIR}")


modified_image_files = [f for f in os.listdir(MODIFIED_BACKGROUNDS_DIR) if f.endswith(".png")]

if not modified_image_files:
    print(f"⚠️ No PNG images found in '{MODIFIED_BACKGROUNDS_DIR}'. Please ensure you've run Stage 1 and performed Photoshop modifications.")
    exit()

for img_filename in tqdm(modified_image_files, desc="Processing Vinifera Images"):
    fid = os.path.splitext(img_filename)[0] # e.g., "000_CHARDONNAY"
    
    try:
        # Load the modified image
        img_path_modified = os.path.join(MODIFIED_BACKGROUNDS_DIR, img_filename)
        img_pil = Image.open(img_path_modified).convert("RGB")
        original_img_size_before_rotation = img_pil.size # Store original size (W, H)
        
        # Apply initial rotation
        img_pil_preprocessed_rot = rotate_to_wide(img_pil)
        
        # Store whether rotation occurred (for mask transformation)
        rotation_applied = (original_img_size_before_rotation[1] > original_img_size_before_rotation[0])
        
        # Rescale and pad the image
        img_pil_padded = rescale_and_pad_image(img_pil_preprocessed_rot, TARGET_SIZE)
        
        # Calculate paste offset for mask transformation
        original_width_after_rot, original_height_after_rot = img_pil_preprocessed_rot.size
        scale_w_calc = TARGET_SIZE[0] / original_width_after_rot
        scale_h_calc = TARGET_SIZE[1] / original_height_after_rot
        scale_factor_calc = min(scale_w_calc, scale_h_calc)

        new_width_calc = int(original_width_after_rot * scale_factor_calc)
        new_height_calc = int(original_height_after_rot * scale_factor_calc)
        paste_x_offset_calc = (TARGET_SIZE[0] - new_width_calc) // 2
        paste_y_offset_calc = (TARGET_SIZE[1] - new_height_calc) // 2
        paste_offset = (paste_x_offset_calc, paste_y_offset_calc)

        # Generate 7-channel input
        seven_channel_data = create_7channel_input(
            img_pil_padded, sato_sigmas, meijering_sigmas, frangi_sigmas, hessian_sigmas, ENHANCE_PERCENTILE
        )
        
        # Save 7-channel data (as .npy file)
        np.save(os.path.join(SEVEN_CHANNEL_OUTPUT_DIR, f"{fid}.npy"), seven_channel_data)

        # --- Generate and save masks ---
        blade_coords_path = os.path.join(RAW_DATA_DIR, f"{fid}_blade.txt")
        vein_coords_path = os.path.join(RAW_DATA_DIR, f"{fid}_veins.txt")

        if not os.path.exists(blade_coords_path):
            print(f"⚠️ Skipping mask generation for {fid}: Blade coordinates file missing.")
            continue
        if not os.path.exists(vein_coords_path):
            print(f"⚠️ Skipping mask generation for {fid}: Vein coordinates file missing.")
            continue

        blade_coords = read_coords(blade_coords_path)
        vein_coords = read_coords(vein_coords_path)

        # Create and save blade mask
        blade_mask = create_mask_from_coords(blade_coords, TARGET_SIZE, original_img_size_before_rotation, rotation_applied, paste_offset)
        np.save(os.path.join(BLADE_MASKS_OUTPUT_DIR, f"{fid}_blade_mask.npy"), blade_mask)

        # Create and save vein mask
        vein_mask = create_mask_from_coords(vein_coords, TARGET_SIZE, original_img_size_before_rotation, rotation_applied, paste_offset)
        np.save(os.path.join(VEIN_MASKS_OUTPUT_DIR, f"{fid}_vein_mask.npy"), vein_mask)
        
    except Exception as e:
        print(f"❌ Error processing {fid}: {e}. Skipping this image.")

print(f"\n--- Vinifera Preprocessing Stage 2 Complete ---")
print(f"Generated 7-channel inputs and masks are ready for transfer learning!")