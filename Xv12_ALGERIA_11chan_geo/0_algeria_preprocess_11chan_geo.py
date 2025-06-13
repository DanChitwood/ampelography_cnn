# algeria_preprocessing_stage2_11channel_geodesic.py

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
import collections # For BFS queue

# === CONFIGURATION ===
# --- Input and Output Directories ---
# MODIFIED: Points to the single 'TRAINING DATA' directory for Algeria
ALGERIA_TRAINING_DATA_DIR = "TRAINING_DATA"

# MODIFIED: Output root specific to Algeria
OUTPUT_ROOT_DIR = "PROCESSED_DATA_FOR_SEGMENTATION_ALGERIA"
ELEVEN_CHANNEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "11_CHANNEL_INPUTS")
BLADE_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "BLADE_MASKS")
VEIN_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "VEIN_MASKS")
GEODESIC_MASKS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "GEODESIC_MASKS")
OVERLAY_CHECK_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "OVERLAY_CHECKS")
CONFIG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "config") # To save preprocessing config

# Create output directories
for d in [ELEVEN_CHANNEL_OUTPUT_DIR, BLADE_MASKS_OUTPUT_DIR, VEIN_MASKS_OUTPUT_DIR, GEODESIC_MASKS_OUTPUT_DIR, OVERLAY_CHECK_OUTPUT_DIR, CONFIG_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Image and Mask Processing Parameters ---
TARGET_WIDTH = 2048
TARGET_HEIGHT = 2040
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG", ".JPEG", ".TIF", ".TIFF", ".PNG"]

# --- Ridge Filter Parameters (Carried over from Vinifera - may need re-tuning for Algeria scans) ---
# NOTE: These sigmas were optimized for Vinifera photos.
# For Algeria (scans, different lighting, fresh leaves), you might consider re-evaluating these.
# Start with them for consistency, but keep in mind they are a potential tuning knob.
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
    "ALGERIA_TRAINING_DATA_DIR": ALGERIA_TRAINING_DATA_DIR, # MODIFIED
    "ELEVEN_CHANNEL_OUTPUT_DIR": ELEVEN_CHANNEL_OUTPUT_DIR,
    "BLADE_MASKS_OUTPUT_DIR": BLADE_MASKS_OUTPUT_DIR,
    "VEIN_MASKS_OUTPUT_DIR": VEIN_MASKS_OUTPUT_DIR,
    "GEODESIC_MASKS_OUTPUT_DIR": GEODESIC_MASKS_OUTPUT_DIR
}
with open(os.path.join(CONFIG_OUTPUT_DIR, "preprocessing_config.json"), 'w') as f:
    json.dump(PREPROCESSING_CONFIG, f, indent=4)
print(f"Preprocessing configuration saved to: {os.path.join(CONFIG_OUTPUT_DIR, 'preprocessing_config.json')}")


# === HELPER FUNCTIONS (Most remain unchanged as they are general purpose) ===
def read_coords(path):
    """Reads coordinate data from a text file. Returns empty array if file not found or empty."""
    if not os.path.exists(path):
        return np.array([])
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
    
    return padded_img, (paste_x, paste_y), (new_width, new_height), scale_factor

def enhance_contrast(arr, percentile_val):
    """Applies contrast enhancement based on percentile."""
    vmax = np.percentile(arr, percentile_val)
    if vmax == 0:  # Avoid division by zero if all values are 0
        return np.zeros_like(arr, dtype=np.float32)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled.astype(np.float32)

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
    hessian_br_false_raw = filters.hessian(gray_image, sigmas=hessian_s, black_ridges=False, mode='reflect')
    hessian_br_true_processed = enhance_contrast(hessian_br_true_raw, enhance_p)
    hessian_br_false_processed = enhance_contrast(hessian_br_false_raw, enhance_p)

    return (sato_br_false_processed, sato_br_true_processed,
            meijering_br_false_processed, meijering_br_true_processed,
            frangi_br_false_processed, frangi_br_true_processed,
            hessian_br_true_processed, hessian_br_false_processed) # 8 channels


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
    neighbors = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],           [ 0, 1],
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

def plot_overlay_check(original_img_pil, blade_mask_np, vein_mask_np, geodesic_map_np, fid, output_dir):
    """
    Generates and saves an overlay plot for visual verification.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original_img_pil)

    # Blade mask (Dodgerblue, alpha=0.7)
    blade_mask_bool = blade_mask_np.astype(bool)
    blade_overlay_rgba = np.zeros((*blade_mask_np.shape, 4), dtype=np.float32)
    blade_overlay_rgba[blade_mask_bool] = [0.1176, 0.5647, 1.0, 0.7] # Dodgerblue RGB + alpha
    ax.imshow(blade_overlay_rgba)

    # Vein mask with geodesic distance (Inferno, alpha=1)
    vein_mask_bool = vein_mask_np.astype(bool)
    if np.any(vein_mask_bool): # Only plot if there are actual vein pixels
        vein_color_map = plt.cm.inferno
        
        # Create an RGB array from the geodesic map for vein pixels
        geodesic_colors_rgb = vein_color_map(geodesic_map_np)[:, :, :3]
        
        # Create an alpha channel: 1 for vein pixels, 0 for non-vein
        vein_alpha_channel = np.zeros(vein_mask_np.shape, dtype=np.float32)
        vein_alpha_channel[vein_mask_bool] = 1.0

        # Combine RGB colors with the correct alpha channel
        vein_overlay_rgba = np.concatenate((geodesic_colors_rgb, vein_alpha_channel[:, :, np.newaxis]), axis=2)
        
        ax.imshow(vein_overlay_rgba)
    else:
        pass # If no vein pixels, just draw nothing

    ax.set_title(f"Overlay Check: {fid}")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{fid}_overlay_check.png"), dpi=200)
    plt.close(fig)

# --- Consolidated Processing Function (MODIFIED for Algeria) ---
# Removed image_dir, trace_dir parameters as they are now consolidated to data_root
def process_algeria_data(data_root_dir, description):
    """
    Processes Algeria images and generates 11-channel inputs and masks.
    This function is tailored for the specific Algeria data structure.
    """
    print(f"\n--- Processing {description} ---")
    
    fids = set()
    # MODIFIED: Collect FIDs by looking for _info.csv files
    for f_name in os.listdir(data_root_dir):
        if f_name.endswith("_info.csv"):
            # FID is the part before _info.csv
            fid = f_name.replace("_info.csv", "")
            fids.add(fid)
            
    fids = sorted(list(fids))

    if not fids:
        print(f"⚠️ No info.csv files found in '{data_root_dir}'. Skipping this source.")
        return

    for fid in tqdm(fids, desc=f"Processing {description}"):
        eleven_channel_output_path = os.path.join(ELEVEN_CHANNEL_OUTPUT_DIR, f"{fid}.npy")
        blade_mask_output_path = os.path.join(BLADE_MASKS_OUTPUT_DIR, f"{fid}_blade_mask.npy")
        vein_mask_output_path = os.path.join(VEIN_MASKS_OUTPUT_DIR, f"{fid}_vein_mask.npy")
        geodesic_mask_output_path = os.path.join(GEODESIC_MASKS_OUTPUT_DIR, f"{fid}_geodesic_mask.npy")

        # Skip if all outputs already exist
        if (os.path.exists(eleven_channel_output_path) and
            os.path.exists(blade_mask_output_path) and
            os.path.exists(vein_mask_output_path) and
            os.path.exists(geodesic_mask_output_path)):
            continue
        
        try:
            # --- 1. Load Image using info.csv ---
            info_csv_path = os.path.join(data_root_dir, f"{fid}_info.csv")
            if not os.path.exists(info_csv_path):
                print(f"⚠️ Skipping {fid}: info.csv not found.")
                continue

            # Read info.csv to get the image filename
            info_df = pd.read_csv(info_csv_path)
            image_filename_row = info_df[info_df['factor'] == 'image']
            if image_filename_row.empty:
                print(f"⚠️ Skipping {fid}: 'image' factor not found in info.csv.")
                continue
            image_filename = image_filename_row['value'].iloc[0]

            original_img_path = os.path.join(data_root_dir, image_filename)
            
            if not os.path.exists(original_img_path):
                print(f"⚠️ Skipping {fid}: Image file '{image_filename}' not found at '{original_img_path}'.")
                continue

            img_pil = Image.open(original_img_path).convert("RGB")
            original_img_size_before_rotation = img_pil.size

            # Apply initial rotation
            img_pil_preprocessed_rot, rotation_applied = rotate_to_wide(img_pil)
            
            # Rescale and pad the image
            img_pil_padded, paste_offset, scaled_dims, scale_factor_applied = rescale_and_pad_image(img_pil_preprocessed_rot, TARGET_SIZE)
            
            # Generate 11-channel input
            eleven_channel_data = create_11channel_input(
                img_pil_padded, sato_sigmas, meijering_sigmas, frangi_sigmas, hessian_sigmas, ENHANCE_PERCENTILE
            )
            
            # Save 11-channel data (as .npy file)
            np.save(eleven_channel_output_path, eleven_channel_data)

            # --- 2. Generate and save masks ---
            final_blade_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8) # H, W
            final_vein_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8) # H, W
            final_geodesic_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.float32) # H, W
            
            # MODIFIED: Blade and vein file names are directly based on FID
            blade_coords_path = os.path.join(data_root_dir, f"{fid}_blade.txt")
            vein_coords_path = os.path.join(data_root_dir, f"{fid}_veins.txt")

            # Check if both blade and vein files exist for this FID
            if os.path.exists(blade_coords_path) and os.path.exists(vein_coords_path):
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
                    
                    # Calculate and accumulate geodesic distance for this specific leaf
                    geodesic_map_for_leaf = calculate_geodesic_distance_map(
                        current_vein_mask, vein_coords, TARGET_SIZE, original_img_size_before_rotation, rotation_applied, paste_offset, scale_factor_applied
                    )
                    final_geodesic_mask = np.maximum(final_geodesic_mask, geodesic_map_for_leaf)
            else:
                print(f"⚠️ Skipping mask generation for {fid}: Blade or vein coordinate files not found in '{data_root_dir}'.")
                continue

            # Save masks
            np.save(blade_mask_output_path, final_blade_mask)
            np.save(vein_mask_output_path, final_vein_mask)
            np.save(geodesic_mask_output_path, final_geodesic_mask)

            # Generate and save overlay check plot
            plot_overlay_check(img_pil_padded, final_blade_mask, final_vein_mask, final_geodesic_mask, fid, OVERLAY_CHECK_OUTPUT_DIR)
                
        except Exception as e:
            print(f"❌ Error processing {fid} from '{description}': {e}. Skipping this image.")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"--- Starting Algeria Preprocessing ---")
    print(f"Input data: {ALGERIA_TRAINING_DATA_DIR}")
    print(f"Output root: {OUTPUT_ROOT_DIR}")

    # Process the single Algeria training data directory
    if os.path.exists(ALGERIA_TRAINING_DATA_DIR):
        process_algeria_data(ALGERIA_TRAINING_DATA_DIR, "Algeria Training Data")
    else:
        print(f"ERROR: Algeria training data directory '{ALGERIA_TRAINING_DATA_DIR}' not found. Please ensure it exists.")

    print(f"\n--- Algeria Preprocessing Complete ---")
    print(f"Generated 11-channel inputs and masks are ready for transfer learning!")