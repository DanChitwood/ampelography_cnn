import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import color, filters, morphology
from tqdm import tqdm
import json
import re
import collections

# --- CONFIGURATION ---
BASE_DIR = os.getcwd() # Assumes script is run from the directory containing synthetic_dataset

# Input directory for synthetic images
SYNTHETIC_DATASET_DIR = os.path.join(BASE_DIR, "synthetic_dataset")

# Output directories for processed data
PROCESSED_OUTPUT_ROOT_DIR = os.path.join(BASE_DIR, "processed_synthetic_data") # New root for processed synthetic data
ELEVEN_CHANNEL_OUTPUT_DIR = os.path.join(PROCESSED_OUTPUT_ROOT_DIR, "11_CHANNEL_INPUTS")
BLADE_MASKS_OUTPUT_DIR = os.path.join(PROCESSED_OUTPUT_ROOT_DIR, "BLADE_MASKS")
VEIN_MASKS_OUTPUT_DIR = os.path.join(PROCESSED_OUTPUT_ROOT_DIR, "VEIN_MASKS")
GEODESIC_MASKS_OUTPUT_DIR = os.path.join(PROCESSED_OUTPUT_ROOT_DIR, "GEODESIC_MASKS")
OVERLAY_CHECK_OUTPUT_DIR = os.path.join(PROCESSED_OUTPUT_ROOT_DIR, "OVERLAY_CHECKS")
CONFIG_OUTPUT_DIR = os.path.join(PROCESSED_OUTPUT_ROOT_DIR, "config")

# Create output directories
for d in [ELEVEN_CHANNEL_OUTPUT_DIR, BLADE_MASKS_OUTPUT_DIR, VEIN_MASKS_OUTPUT_DIR, GEODESIC_MASKS_OUTPUT_DIR, OVERLAY_CHECK_OUTPUT_DIR, CONFIG_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Image and Mask Processing Parameters ---
TARGET_WIDTH = 2048
TARGET_HEIGHT = 2040
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT) # Expected size of synthetic RGB images

# --- Ridge Filter Parameters ---
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
    "SYNTHETIC_DATASET_DIR": SYNTHETIC_DATASET_DIR,
    "ELEVEN_CHANNEL_OUTPUT_DIR": ELEVEN_CHANNEL_OUTPUT_DIR,
    "BLADE_MASKS_OUTPUT_DIR": BLADE_MASKS_OUTPUT_DIR,
    "VEIN_MASKS_OUTPUT_DIR": VEIN_MASKS_OUTPUT_DIR,
    "GEODESIC_MASKS_OUTPUT_DIR": GEODESIC_MASKS_OUTPUT_DIR,
    "OVERLAY_CHECK_OUTPUT_DIR": OVERLAY_CHECK_OUTPUT_DIR
}
with open(os.path.join(CONFIG_OUTPUT_DIR, "preprocessing_config.json"), 'w') as f:
    json.dump(PREPROCESSING_CONFIG, f, indent=4)
print(f"Preprocessing configuration saved to: {os.path.join(CONFIG_OUTPUT_DIR, 'preprocessing_config.json')}")


# === HELPER FUNCTIONS ===
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

# The rotate_to_wide and rescale_and_pad_image functions are NOT needed for the input RGB images,
# as they are already standardized by 2_generate_synthetic_images.py.
# However, they are used by create_mask_from_coords below, but with simplified parameters now.

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

# MODIFIED: create_mask_from_coords is simplified for synthetic data
def create_mask_from_coords(coords, target_size):
    """
    Creates a binary mask from polygon coordinates.
    For synthetic data, coordinates are already in target_size space.
    
    Args:
        coords (np.array): Nx2 array of (x, y) coordinates for the polygon.
        target_size (tuple): (width, height) of the final mask.
            
    Returns:
        np.array: Binary mask (0s and 1s) of target_size.
    """
    mask = Image.new("L", target_size, 0) # Black background
    draw = ImageDraw.Draw(mask)

    if coords.size == 0:
        return np.array(mask) # Return empty mask if no coordinates
    
    # Coordinates are already transformed, just draw directly
    draw.polygon([tuple(p) for p in coords], fill=1)
    
    return np.array(mask)

# MODIFIED: calculate_geodesic_distance_map to accept explicit origin
def calculate_geodesic_distance_map(vein_mask_binary, geodesic_origin_xy, target_size):
    """
    Calculates the geodesic distance for each vein pixel from a given origin using BFS.
    Distances are normalized to 0-1.
    """
    geodesic_map = np.full(target_size[::-1], np.inf, dtype=np.float32) # Initialize with infinity (H, W)

    if geodesic_origin_xy is None or not (0 <= geodesic_origin_xy[0] < target_size[0] and 0 <= geodesic_origin_xy[1] < target_size[1]):
        # print(f"  WARNING: Invalid geodesic origin {geodesic_origin_xy}. Returning empty geodesic map.")
        return np.zeros(target_size[::-1], dtype=np.float32)

    origin_x_padded, origin_y_padded = int(geodesic_origin_xy[0]), int(geodesic_origin_xy[1])

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
    fig, ax = plt.subplots(figsize=(TARGET_WIDTH/100, TARGET_HEIGHT/100), dpi=100) # Adjust figsize/dpi for target_size
    ax.imshow(original_img_pil)

    # Blade mask (Dodgerblue, alpha=0.7)
    blade_mask_bool = blade_mask_np.astype(bool)
    blade_overlay_rgba = np.zeros((*blade_mask_np.shape, 4), dtype=np.float32)
    blade_overlay_rgba[blade_mask_bool] = [0.1176, 0.5647, 1.0, 0.7] # Dodgerblue RGB + alpha
    ax.imshow(blade_overlay_rgba)

    # Vein mask with geodesic distance (Inferno, alpha=1)
    vein_mask_bool = vein_mask_np.astype(bool)
    if np.any(vein_mask_bool):
        vein_color_map = plt.cm.inferno
        
        # Create an RGB array from the geodesic map for vein pixels
        geodesic_colors_rgb = vein_color_map(geodesic_map_np)[:, :, :3]
        
        # Create an alpha channel: 1 for vein pixels, 0 for non-vein
        vein_alpha_channel = np.zeros(vein_mask_np.shape, dtype=np.float32)
        vein_alpha_channel[vein_mask_bool] = 1.0

        vein_overlay_rgba = np.concatenate((geodesic_colors_rgb, vein_alpha_channel[:, :, np.newaxis]), axis=2)
        
        ax.imshow(vein_overlay_rgba)
    else:
        pass # No veins to plot

    ax.set_title(f"Overlay Check: {fid}")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{fid}_overlay_check.png"), dpi=200)
    plt.close(fig)


# --- Consolidated Processing Function for Synthetic Data ---
def process_synthetic_dataset():
    """
    Processes synthetic images and generates 11-channel inputs and masks.
    """
    print(f"\n--- Processing Synthetic Dataset from '{SYNTHETIC_DATASET_DIR}' ---")
    
    # List all synthetic image subdirectories (e.g., synthetic_00000)
    synthetic_fids = sorted([
        d for d in os.listdir(SYNTHETIC_DATASET_DIR)
        if os.path.isdir(os.path.join(SYNTHETIC_DATASET_DIR, d)) and d.startswith("synthetic_")
    ])

    if not synthetic_fids:
        print(f"⚠️ No synthetic image folders found in '{SYNTHETIC_DATASET_DIR}'. Please run 2_generate_synthetic_images.py first.")
        return

    for fid in tqdm(synthetic_fids, desc="Processing Synthetic Images"):
        synthetic_image_path = os.path.join(SYNTHETIC_DATASET_DIR, fid, f"{fid}_rgb.png")
        geodesic_origins_json_path = os.path.join(SYNTHETIC_DATASET_DIR, fid, f"{fid}_geodesic_origins.json")

        eleven_channel_output_path = os.path.join(ELEVEN_CHANNEL_OUTPUT_DIR, f"{fid}.npy")
        blade_mask_output_path = os.path.join(BLADE_MASKS_OUTPUT_DIR, f"{fid}_blade_mask.npy")
        vein_mask_output_path = os.path.join(VEIN_MASKS_OUTPUT_DIR, f"{fid}_vein_mask.npy")
        geodesic_mask_output_path = os.path.join(GEODESIC_MASKS_OUTPUT_DIR, f"{fid}_geodesic_mask.npy")

        # Skip if all outputs already exist
        if (os.path.exists(eleven_channel_output_path) and
            os.path.exists(blade_mask_output_path) and
            os.path.exists(vein_mask_output_path) and
            os.path.exists(geodesic_mask_output_path)):
            # print(f"  Skipping {fid}: All processed outputs already exist.")
            continue
        
        try:
            # --- 1. Load Synthetic RGB Image ---
            if not os.path.exists(synthetic_image_path):
                print(f"⚠️ Skipping {fid}: RGB image not found at '{synthetic_image_path}'.")
                continue
            img_pil = Image.open(synthetic_image_path).convert("RGB")
            
            # Verify target size
            if img_pil.size != TARGET_SIZE:
                print(f"  WARNING: Synthetic image {fid} has unexpected size {img_pil.size}. Resizing to {TARGET_SIZE}.")
                img_pil = img_pil.resize(TARGET_SIZE, Image.LANCZOS)

            # Generate 11-channel input
            eleven_channel_data = create_11channel_input(
                img_pil, sato_sigmas, meijering_sigmas, frangi_sigmas, hessian_sigmas, ENHANCE_PERCENTILE
            )
            np.save(eleven_channel_output_path, eleven_channel_data)

            # --- 2. Generate and save masks ---
            final_blade_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8) # H, W
            final_vein_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8) # H, W
            final_geodesic_mask = np.zeros(TARGET_SIZE[::-1], dtype=np.float32) # H, W
            
            # Load geodesic origins
            geodesic_origins_data = []
            if os.path.exists(geodesic_origins_json_path):
                with open(geodesic_origins_json_path, 'r') as f:
                    geodesic_origins_data = json.load(f)
            
            found_traces_for_fid = False
            # Iterate through individual leaves (blade_0.txt, blade_1.txt, etc.)
            leaf_indices = set()
            for f in os.listdir(os.path.join(SYNTHETIC_DATASET_DIR, fid)):
                match = re.match(r'.*_blade_(\d+)\.txt', f)
                if match:
                    leaf_indices.add(int(match.group(1)))
            leaf_indices = sorted(list(leaf_indices))

            for leaf_idx in leaf_indices:
                blade_coords_path = os.path.join(SYNTHETIC_DATASET_DIR, fid, f"{fid}_blade_{leaf_idx}.txt")
                vein_coords_path = os.path.join(SYNTHETIC_DATASET_DIR, fid, f"{fid}_vein_{leaf_idx}.txt") # Note: _vein not _veins for synthetic output

                if os.path.exists(blade_coords_path) and os.path.exists(vein_coords_path):
                    found_traces_for_fid = True
                    blade_coords = read_coords(blade_coords_path)
                    vein_coords = read_coords(vein_coords_path)

                    if blade_coords.size > 0:
                        current_blade_mask = create_mask_from_coords(blade_coords, TARGET_SIZE)
                        final_blade_mask = np.logical_or(final_blade_mask, current_blade_mask).astype(np.uint8)

                    if vein_coords.size > 0:
                        current_vein_mask = create_mask_from_coords(vein_coords, TARGET_SIZE)
                        final_vein_mask = np.logical_or(final_vein_mask, current_vein_mask).astype(np.uint8)
                        
                        # Get the specific geodesic origin for this leaf_idx
                        leaf_origin = next((item for item in geodesic_origins_data if item["leaf_idx"] == leaf_idx), None)
                        if leaf_origin and "x" in leaf_origin and "y" in leaf_origin:
                            geodesic_origin_xy = (leaf_origin["x"], leaf_origin["y"])
                        else:
                            geodesic_origin_xy = None # No origin found for this leaf

                        geodesic_map_for_leaf = calculate_geodesic_distance_map(
                            current_vein_mask, geodesic_origin_xy, TARGET_SIZE
                        )
                        final_geodesic_mask = np.maximum(final_geodesic_mask, geodesic_map_for_leaf)
            
            if not found_traces_for_fid:
                print(f"⚠️ Skipping mask generation for {fid}: No blade/vein coordinate files found for any leaf in '{os.path.join(SYNTHETIC_DATASET_DIR, fid)}'.")
                continue

            # Save masks
            np.save(blade_mask_output_path, final_blade_mask)
            np.save(vein_mask_output_path, final_vein_mask)
            np.save(geodesic_mask_output_path, final_geodesic_mask)

            # Generate and save overlay check plot
            plot_overlay_check(img_pil, final_blade_mask, final_vein_mask, final_geodesic_mask, fid, OVERLAY_CHECK_OUTPUT_DIR)
                
        except Exception as e:
            print(f"❌ Error processing {fid}: {e}. Skipping this image.")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"--- Starting Synthetic Data Preprocessing ---")
    print(f"Input root: {SYNTHETIC_DATASET_DIR}")
    print(f"Output root: {PROCESSED_OUTPUT_ROOT_DIR}")

    process_synthetic_dataset()

    print(f"\n--- Synthetic Data Preprocessing Complete ---")
    print(f"Generated 11-channel inputs and masks are ready!")