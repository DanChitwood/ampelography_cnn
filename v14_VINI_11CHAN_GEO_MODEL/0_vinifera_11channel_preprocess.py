# vinifera_preprocessing_stage2_7channel_masks.py

# === IMPORTS ===
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import color, filters, morphology
from tqdm import tqdm
import json
import re # For regular expressions to get file IDs
# from scipy.ndimage import distance_transform_edt # No longer needed for geodesic
import collections # NEW: For BFS queue

# === CONFIGURATION ===
# --- Input and Output Directories ---
RAW_DATA_DIR = "original_data"
MODIFIED_BACKGROUNDS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA/MODIFIED_BACKGROUNDS"
ADDITIONAL_DATA_DIR = "additional_data"
# NEW: Path to your additional data folder with multiple leaves
ADDITIONAL_DATA_MULTIPLE_LEAVES_DIR = "additional_data_multiple_leaves"

OUTPUT_ROOT_DIR = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA"
ELEVEN_CHANNEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "11_CHANNEL_INPUTS") # Renamed from 7_CHANNEL
BLADE_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "BLADE_MASKS")
VEIN_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "VEIN_MASKS")
GEODESIC_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "GEODESIC_MASKS") # NEW: Geodesic mask output directory
OVERLAY_CHECK_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "OVERLAY_CHECKS") # NEW: Overlay visualization output directory
CONFIG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "config") # To save preprocessing config

# Create output directories
for d in [ELEVEN_CHANNEL_OUTPUT_DIR, BLADE_MASKS_OUTPUT_DIR, VEIN_MASKS_OUTPUT_DIR, GEODESIC_MASKS_OUTPUT_DIR, OVERLAY_CHECK_OUTPUT_DIR, CONFIG_OUTPUT_DIR]:
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
    "RAW_DATA_DIR": RAW_DATA_DIR,
    "MODIFIED_BACKGROUNDS_DIR": MODIFIED_BACKGROUNDS_DIR,
    "ADDITIONAL_DATA_DIR": ADDITIONAL_DATA_DIR,
    "ADDITIONAL_DATA_MULTIPLE_LEAVES_DIR": ADDITIONAL_DATA_MULTIPLE_LEAVES_DIR, # NEW
    "ELEVEN_CHANNEL_OUTPUT_DIR": ELEVEN_CHANNEL_OUTPUT_DIR, # Renamed
    "BLADE_MASKS_OUTPUT_DIR": BLADE_MASKS_OUTPUT_DIR,
    "VEIN_MASKS_OUTPUT_DIR": VEIN_MASKS_OUTPUT_DIR,
    "GEODESIC_MASKS_OUTPUT_DIR": GEODESIC_MASKS_OUTPUT_DIR # NEW
}
with open(os.path.join(CONFIG_OUTPUT_DIR, "preprocessing_config.json"), 'w') as f:
    json.dump(PREPROCESSING_CONFIG, f, indent=4)
print(f"Preprocessing configuration saved to: {os.path.join(CONFIG_OUTPUT_DIR, 'preprocessing_config.json')}")


# === HELPER FUNCTIONS ===
def read_coords(path):
    """Reads coordinate data from a text file. Returns empty array if file not found or empty."""
    if not os.path.exists(path):
        return np.array([])
    # Handle empty files: np.loadtxt will raise an error if file is empty
    try:
        coords = np.loadtxt(path)
        if coords.ndim == 1: # If only one point, make it 2D (1, 2)
            coords = coords.reshape(1, -1)
        return coords
    except ValueError: # Empty file or malformed
        return np.array([])


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
    
    return padded_img, (paste_x, paste_y), (new_width, new_height), scale_factor # NEW: return scale_factor

def enhance_contrast(arr, percentile_val):
    """Applies contrast enhancement based on percentile."""
    vmax = np.percentile(arr, percentile_val)
    if vmax == 0: 
        return np.zeros_like(arr, dtype=np.float32)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled.astype(np.float32)

# MODIFIED: apply_ridge_filters to generate both black_ridges=True and False versions
def apply_ridge_filters(image_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p):
    """
    Applies various ridge filters to a grayscale image and returns their enhanced outputs.
    Generates both black_ridge=True and False versions for Sato, Meijering, Frangi.
    Generates both black_ridge=True and False versions for Hessian.
    Takes a PIL Image that is already padded to TARGET_SIZE.
    """
    image_rgb_float = np.array(image_pil_padded).astype(np.float32) / 255.0
    
    if image_rgb_float.ndim == 3 and image_rgb_float.shape[2] == 3:
        gray_image = color.rgb2gray(image_rgb_float)
    elif image_rgb_float.ndim == 2:
        gray_image = image_rgb_float
    else:
        raise ValueError(f"Unexpected image dimensions or mode for filter application: {image_rgb_float.shape}")

    # --- Sato Filter ---
    sato_br_false_raw = filters.sato(gray_image, sigmas=sato_s, black_ridges=False, mode='reflect')
    sato_br_true_raw = filters.sato(gray_image, sigmas=sato_s, black_ridges=True, mode='reflect')
    sato_br_false_processed = enhance_contrast(sato_br_false_raw, enhance_p)
    sato_br_true_processed = enhance_contrast(sato_br_true_raw, enhance_p)

    # --- Meijering Filter ---
    meijering_br_false_raw = filters.meijering(gray_image, sigmas=meijering_s, black_ridges=False, mode='reflect')
    meijering_br_true_raw = filters.meijering(gray_image, sigmas=meijering_s, black_ridges=True, mode='reflect')
    meijering_br_false_processed = enhance_contrast(meijering_br_false_raw, enhance_p)
    meijering_br_true_processed = enhance_contrast(meijering_br_true_raw, enhance_p)

    # --- Frangi Filter ---
    frangi_br_false_raw = filters.frangi(gray_image, sigmas=frangi_s, black_ridges=False, mode='reflect')
    frangi_br_true_raw = filters.frangi(gray_image, sigmas=frangi_s, black_ridges=True, mode='reflect')
    frangi_br_false_processed = enhance_contrast(frangi_br_false_raw, enhance_p)
    frangi_br_true_processed = enhance_contrast(frangi_br_true_raw, enhance_p)

    # --- Hessian Filter ---
    hessian_br_true_raw = filters.hessian(gray_image, sigmas=hessian_s, black_ridges=True, mode='reflect')
    hessian_br_false_raw = filters.hessian(gray_image, sigmas=hessian_s, black_ridges=False, mode='reflect') # NEW: opposite
    hessian_br_true_processed = enhance_contrast(hessian_br_true_raw, enhance_p)
    hessian_br_false_processed = enhance_contrast(hessian_br_false_raw, enhance_p)

    return (sato_br_false_processed, sato_br_true_processed,
            meijering_br_false_processed, meijering_br_true_processed,
            frangi_br_false_processed, frangi_br_true_processed,
            hessian_br_true_processed, hessian_br_false_processed) # 8 channels


# MODIFIED: create_11channel_input (renamed from create_7channel_input)
def create_11channel_input(img_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p):
    """
    Creates the 11-channel input array for the UNet model.
    Channels: L, A, B, Sato_F, Sato_T, Meijering_F, Meijering_T, Frangi_F, Frangi_T, Hessian_T, Hessian_F.
    """
    img_rgb_float = np.array(img_pil_padded).astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_rgb_float)

    L_channel = img_lab[:, :, 0] / 100.0  # L channel normalized to 0-1
    A_channel = (img_lab[:, :, 1] + 128) / 255.0  # A channel normalized to 0-1
    B_channel = (img_lab[:, :, 2] + 128) / 255.0  # B channel normalized to 0-1

    sato_f, sato_t, meijering_f, meijering_t, frangi_f, frangi_t, hessian_t, hessian_f = \
        apply_ridge_filters(img_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p)

    eleven_channels = np.stack([L_channel, A_channel, B_channel,
                                sato_f, sato_t, meijering_f, meijering_t,
                                frangi_f, frangi_t, hessian_t, hessian_f], axis=-1)
    
    return eleven_channels

# MODIFIED: create_mask_from_coords to accept scale_factor directly
def create_mask_from_coords(coords, target_size, original_size_before_rotation, rotation_applied, paste_offset, scale_factor):
    """
    Creates a binary mask from polygon coordinates, handling original image dimensions
    and scaling/padding to target_size.
    
    Args:
        coords (np.array): Nx2 array of (x, y) coordinates for the polygon.
        target_size (tuple): (width, height) of the final mask.
        original_size_before_rotation (tuple): (width, height) of the original image before rotation.
        rotation_applied (bool): True if rotate_to_wide was applied.
        paste_offset (tuple): (x_offset, y_offset) from rescale_and_pad_image.
        scale_factor (float): The actual scale factor applied during image resizing.
            
    Returns:
        np.array: Binary mask (0s and 1s) of target_size.
    """
    mask = Image.new("L", target_size, 0) # Black background
    draw = ImageDraw.Draw(mask)

    if coords.size == 0:
        return np.array(mask) # Return empty mask if no coordinates

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

# MODIFIED: Function to calculate geodesic distance map (using BFS)
def calculate_geodesic_distance_map(vein_mask_binary, vein_coords_raw, target_size, original_size_before_rotation, rotation_applied, paste_offset, scale_factor):
    """
    Calculates the geodesic distance for each vein pixel from its origin using BFS.
    The origin is the average of the first and last points of the raw vein coordinates.
    Distances are normalized to 0-1.
    """
    geodesic_map = np.full(target_size[::-1], np.inf, dtype=np.float32) # Initialize with infinity (H, W)

    if vein_coords_raw.size < 2: # Need at least 2 points for origin calculation
        return np.zeros(target_size[::-1], dtype=np.float32) # Return empty map if not enough points

    # Calculate origin in original coordinates (x, y)
    origin_x_orig = (vein_coords_raw[0, 0] + vein_coords_raw[-1, 0]) / 2
    origin_y_orig = (vein_coords_raw[0, 1] + vein_coords_raw[-1, 1]) / 2

    # Transform origin to padded image coordinates
    if rotation_applied:
        # If original image was (W_orig, H_orig) and rotated 270 deg (clockwise) to (H_orig, W_orig),
        # then new_x = old_y, new_y = old_W_orig - 1 - old_x
        # So, the original_img_size_before_rotation (W,H) corresponds to (width, height)
        # origin_x_orig is from width dimension, origin_y_orig from height dimension
        transformed_origin_x = origin_y_orig # old_y becomes new_x
        transformed_origin_y = original_size_before_rotation[0] - 1 - origin_x_orig # old_W_orig - 1 - old_x becomes new_y
    else:
        transformed_origin_x = origin_x_orig
        transformed_origin_y = origin_y_orig

    origin_x_padded = int(transformed_origin_x * scale_factor + paste_offset[0])
    origin_y_padded = int(transformed_origin_y * scale_factor + paste_offset[1])

    # Ensure origin is within padded image bounds
    origin_x_padded = np.clip(origin_x_padded, 0, target_size[0] - 1) # target_size[0] is width
    origin_y_padded = np.clip(origin_y_padded, 0, target_size[1] - 1) # target_size[1] is height

    # Find the nearest vein pixel to the transformed origin to start BFS
    # It's crucial the BFS starts *on* a vein pixel.
    
    # Create a small region around the origin to search for vein pixels
    search_radius = 10 # Increase search radius slightly
    y_min, y_max = max(0, origin_y_padded - search_radius), min(target_size[1], origin_y_padded + search_radius + 1)
    x_min, x_max = max(0, origin_x_padded - search_radius), min(target_size[0], origin_x_padded + search_radius + 1)
    
    # Extract sub-mask for efficient search
    sub_mask_view = vein_mask_binary[y_min:y_max, x_min:x_max]
    
    # Get coordinates of all vein pixels in the sub-mask
    vein_pixels_in_sub = np.argwhere(sub_mask_view == 1)
    
    if vein_pixels_in_sub.size == 0:
        # If no vein pixels found in the search radius, return an empty map
        return np.zeros(target_size[::-1], dtype=np.float32) 

    # Calculate Euclidean distances from the target origin to all vein pixels in the sub-mask
    distances_to_origin_in_sub = np.sqrt(
        (vein_pixels_in_sub[:, 1] + x_min - origin_x_padded)**2 +
        (vein_pixels_in_sub[:, 0] + y_min - origin_y_padded)**2
    )

    # Find the index of the closest vein pixel in the `vein_pixels_in_sub` list
    closest_vein_pixel_idx = np.argmin(distances_to_origin_in_sub)
    
    # Get the global (y, x) coordinates of the start node for BFS
    start_node_y, start_node_x = vein_pixels_in_sub[closest_vein_pixel_idx] + [y_min, x_min]

    # --- BFS for Geodesic Distance ---
    # Queue stores (y, x, distance)
    q = collections.deque([(start_node_y, start_node_x, 0)])
    geodesic_map[start_node_y, start_node_x] = 0 # Set origin distance to 0

    # 8-connectivity neighbors (including diagonals)
    # Changed from list of tuples to array for potential future speedup/consistency, though list is fine
    neighbors = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1]
    ])

    while q:
        r, c, dist = q.popleft()

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc

            # Check bounds
            if not (0 <= nr < target_size[1] and 0 <= nc < target_size[0]): # Check bounds: height then width
                continue
            
            # Check if neighbor is a vein pixel AND its current distance is infinity (unvisited)
            # This ensures we take the shortest path and don't re-process visited nodes.
            if vein_mask_binary[nr, nc] == 1 and geodesic_map[nr, nc] == np.inf:
                geodesic_map[nr, nc] = dist + 1 # Each step increments distance by 1
                q.append((nr, nc, dist + 1))
    
    # Set non-vein pixels (still at np.inf) to 0 for consistent output and normalization
    geodesic_map[geodesic_map == np.inf] = 0 

    # Normalize the valid distances (only for vein pixels) to 0-1
    # Find max distance *only among vein pixels*
    max_dist = np.max(geodesic_map[vein_mask_binary == 1]) # Max among pixels that are actually veins
    if max_dist > 0:
        geodesic_map[vein_mask_binary == 1] = geodesic_map[vein_mask_binary == 1] / max_dist
    else:
        # If max_dist is 0 (e.g., a single pixel vein or no vein), ensure map is all zeros
        geodesic_map = np.zeros_like(geodesic_map) 

    return geodesic_map

# NEW: Function for visualization overlay
def plot_overlay_check(original_img_pil, blade_mask_np, vein_mask_np, geodesic_map_np, fid, output_dir):
    """
    Generates and saves an overlay plot for visual verification.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original_img_pil)

    # Blade mask (Dodgerblue, alpha=0.7)
    # Ensure blade_mask_np is boolean or uint8 for correct masking
    blade_mask_bool = blade_mask_np.astype(bool)
    blade_overlay_rgba = np.zeros((*blade_mask_np.shape, 4), dtype=np.float32)
    blade_overlay_rgba[blade_mask_bool] = [0.1176, 0.5647, 1.0, 0.7] # Dodgerblue RGB + alpha
    ax.imshow(blade_overlay_rgba)

    # Vein mask with geodesic distance (Inferno, alpha=1)
    vein_mask_bool = vein_mask_np.astype(bool)
    if np.any(vein_mask_bool): # Only plot if there are actual vein pixels
        vein_color_map = plt.cm.inferno
        
        # Create an RGB array from the geodesic map for vein pixels
        # Ensure geodesic_map_np is normalized 0-1
        geodesic_colors_rgb = vein_color_map(geodesic_map_np)[:, :, :3]
        
        # Create an alpha channel: 1 for vein pixels, 0 for non-vein
        vein_alpha_channel = np.zeros(vein_mask_np.shape, dtype=np.float32)
        vein_alpha_channel[vein_mask_bool] = 1.0

        # Combine RGB colors with the correct alpha channel
        vein_overlay_rgba = np.concatenate((geodesic_colors_rgb, vein_alpha_channel[:, :, np.newaxis]), axis=2)
        
        ax.imshow(vein_overlay_rgba)
    else:
        # If no vein pixels, just draw a transparent layer or nothing
        pass


    ax.set_title(f"Overlay Check: {fid}")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{fid}_overlay_check.png"), dpi=200)
    plt.close(fig)


# --- Consolidated Processing Function ---
def process_data_source(image_dir, trace_dir, description, is_multi_leaf_data=False): # NEW: is_multi_leaf_data flag
    """
    Processes images and generates 11-channel inputs and masks from a given source.
    Handles both raw JPGs (for new data) and pre-modified PNGs (for existing data).
    NEW: Handles multiple leaves per image and geodesic distance.
    """
    print(f"\n--- Processing {description} ---")
    
    fids = set()
    if is_multi_leaf_data:
        # For multi-leaf data, FIDs are base names like K0369A
        # We find them by looking for any image or any _bladeX.txt / _veinsX.txt
        # Match pattern like K0369A_bladeA.txt -> K0369A
        for f in os.listdir(trace_dir):
            if f.endswith(".txt"):
                match = re.match(r'(.+?)([A-Z])?(_blade|_veins)\.txt', f)
                if match:
                    fids.add(match.group(1)) # Add the base FID (e.g., K0369A)
        # Also ensure we get FIDs from the image directory itself
        for f in os.listdir(image_dir):
            if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                 fids.add(os.path.splitext(f)[0]) # Add base FID for images
    else:
        # For single-leaf data, FIDs are the full base names
        for f in os.listdir(image_dir):
            if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                fids.add(os.path.splitext(f)[0])
        # Also ensure we get FIDs from the trace directory for cases where image might be missing but trace exists
        for f in os.listdir(trace_dir):
            if f.endswith(("_blade.txt", "_veins.txt")):
                fids.add(os.path.splitext(f.replace("_blade", "").replace("_veins", ""))[0])
    
    fids = sorted(list(fids))

    if not fids:
        print(f"⚠️ No relevant files found in '{image_dir}' or '{trace_dir}'. Skipping this source.")
        return

    for fid in tqdm(fids, desc=f"Processing {description}"):
        eleven_channel_output_path = os.path.join(ELEVEN_CHANNEL_OUTPUT_DIR, f"{fid}.npy")
        blade_mask_output_path = os.path.join(BLADE_MASKS_OUTPUT_DIR, f"{fid}_blade_mask.npy")
        vein_mask_output_path = os.path.join(VEIN_MASKS_OUTPUT_DIR, f"{fid}_vein_mask.npy")
        geodesic_mask_output_path = os.path.join(GEODESIC_MASKS_OUTPUT_DIR, f"{fid}_geodesic_mask.npy") # NEW

        # Skip if all outputs already exist
        if (os.path.exists(eleven_channel_output_path) and
            os.path.exists(blade_mask_output_path) and
            os.path.exists(vein_mask_output_path) and
            os.path.exists(geodesic_mask_output_path)): # NEW: Check geodesic mask
            continue
        
        try:
            # --- 1. Load Image (from appropriate source) ---
            img_pil = None
            original_img_path = None

            for ext in IMAGE_EXTENSIONS:
                temp_path = os.path.join(image_dir, f"{fid}{ext}")
                if os.path.exists(temp_path):
                    original_img_path = temp_path
                    break
            
            if original_img_path is None:
                print(f"⚠️ Skipping {fid}: No suitable image file found in '{image_dir}'.")
                continue

            img_pil = Image.open(original_img_path).convert("RGB")
            original_img_size_before_rotation = img_pil.size

            # Apply initial rotation
            img_pil_preprocessed_rot, rotation_applied = rotate_to_wide(img_pil)
            
            # Rescale and pad the image
            img_pil_padded, paste_offset, scaled_dims, scale_factor_applied = rescale_and_pad_image(img_pil_preprocessed_rot, TARGET_SIZE) # NEW: get scale_factor_applied
            
            # Generate 11-channel input
            eleven_channel_data = create_11channel_input(
                img_pil_padded, sato_sigmas, meijering_sigmas, frangi_sigmas, hessian_sigmas, ENHANCE_PERCENTILE
            )
            
            # Save 11-channel data (as .npy file)
            np.save(eleven_channel_output_path, eleven_channel_data)

            # --- 2. Generate and save masks ---
            final_blade_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8) # H, W
            final_vein_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8) # H, W
            # Use float32 for geodesic map as it stores normalized distances
            final_geodesic_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.float32) # H, W
            
            # Collect all relevant blade/vein IDs (e.g., K0369A_bladeA, K0369A_veinsB)
            leaf_suffixes = [''] # For single leaf data, suffix is empty string
            if is_multi_leaf_data:
                # Find all unique single-char suffixes (A, B, etc.) for this FID
                # We need to list files in trace_dir and find suffixes
                found_suffixes = set()
                for f in os.listdir(trace_dir):
                    if f.startswith(f"{fid}_") and f.endswith(".txt"):
                        match = re.match(f'{fid}_(blade|veins)([A-Z])\.txt', f)
                        if match:
                            found_suffixes.add(match.group(2))
                leaf_suffixes = sorted(list(found_suffixes))
                if not leaf_suffixes: # If no A, B, etc. found, it might still be a single-leaf image in a multi-leaf folder
                    leaf_suffixes = [''] # Treat as a single leaf if no suffix is found


            # Iterate through each leaf (or the single leaf)
            found_traces_for_fid = False
            for suffix_char in leaf_suffixes:
                blade_file_name = f"{fid}_blade{suffix_char}.txt" if suffix_char else f"{fid}_blade.txt"
                vein_file_name = f"{fid}_veins{suffix_char}.txt" if suffix_char else f"{fid}_veins.txt"
                
                blade_coords_path = os.path.join(trace_dir, blade_file_name)
                vein_coords_path = os.path.join(trace_dir, vein_file_name)

                # Check if both blade and vein files exist for this specific leaf suffix
                if os.path.exists(blade_coords_path) and os.path.exists(vein_coords_path):
                    found_traces_for_fid = True
                    blade_coords = read_coords(blade_coords_path)
                    vein_coords = read_coords(vein_coords_path)

                    # Create and accumulate blade mask
                    if blade_coords.size > 0:
                        current_blade_mask = create_mask_from_coords(blade_coords, TARGET_SIZE, original_img_size_before_rotation, rotation_applied, paste_offset, scale_factor_applied)
                        final_blade_mask = np.logical_or(final_blade_mask, current_blade_mask).astype(np.uint8)

                    # Create and accumulate vein mask
                    if vein_coords.size > 0:
                        current_vein_mask = create_mask_from_coords(vein_coords, TARGET_SIZE, original_img_size_before_rotation, rotation_applied, paste_offset, scale_factor_applied)
                        final_vein_mask = np.logical_or(final_vein_mask, current_vein_mask).astype(np.uint8)
                        
                        # NEW: Calculate and accumulate geodesic distance for this specific leaf
                        # geodesic_map_for_leaf will be 0-1 normalized for THIS leaf
                        geodesic_map_for_leaf = calculate_geodesic_distance_map(
                            current_vein_mask, vein_coords, TARGET_SIZE, original_img_size_before_rotation, rotation_applied, paste_offset, scale_factor_applied
                        )
                        # Accumulate by taking the maximum value. This handles potential overlaps gracefully.
                        final_geodesic_mask = np.maximum(final_geodesic_mask, geodesic_map_for_leaf)
                
            if not found_traces_for_fid:
                print(f"⚠️ Skipping mask generation for {fid}: No blade/vein coordinate files found for any suffix in '{trace_dir}'.")
                continue

            # Save masks
            np.save(blade_mask_output_path, final_blade_mask)
            np.save(vein_mask_output_path, final_vein_mask)
            np.save(geodesic_mask_output_path, final_geodesic_mask) # NEW: Save geodesic mask

            # NEW: Generate and save overlay check plot
            plot_overlay_check(img_pil_padded, final_blade_mask, final_vein_mask, final_geodesic_mask, fid, OVERLAY_CHECK_OUTPUT_DIR)
                
        except Exception as e:
            print(f"❌ Error processing {fid} from '{description}': {e}. Skipping this image.")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"--- Starting Vinifera Preprocessing ---")
    print(f"Output root: {OUTPUT_ROOT_DIR}")

    # Process original data (using modified backgrounds for images, raw_data_vinifera for traces)
    # This assumes RAW_DATA_DIR contains the _blade.txt, _veins.txt for original_data's FIDs
    process_data_source(MODIFIED_BACKGROUNDS_DIR, RAW_DATA_DIR, "Original Data (Modified Backgrounds)")

    # Process additional data (single leaf)
    if os.path.exists(ADDITIONAL_DATA_DIR):
        process_data_source(ADDITIONAL_DATA_DIR, ADDITIONAL_DATA_DIR, "Additional Data (Single Leaf)")
    else:
        print(f"Warning: Additional data directory '{ADDITIONAL_DATA_DIR}' not found. Skipping single leaf data processing.")

    # NEW: Process additional data (multiple leaves)
    if os.path.exists(ADDITIONAL_DATA_MULTIPLE_LEAVES_DIR):
        process_data_source(ADDITIONAL_DATA_MULTIPLE_LEAVES_DIR, ADDITIONAL_DATA_MULTIPLE_LEAVES_DIR, "Additional Data (Multiple Leaves)", is_multi_leaf_data=True)
    else:
        print(f"Warning: Multiple leaves data directory '{ADDITIONAL_DATA_MULTIPLE_LEAVES_DIR}' not found. Skipping multiple leaf data processing.")

    print(f"\n--- Vinifera Preprocessing Complete ---")
    print(f"Generated 11-channel inputs and masks are ready for transfer learning!")