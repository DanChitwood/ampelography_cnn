# 2_generate_geodesic_masks.py (FINAL CORRECTION for Geodesic Distance ALONG VEINS)

# === IMPORTS ===
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt # For plotting overlays
import matplotlib.cm as cm # For colormaps
import collections # For deque, used in BFS
from scipy.spatial.distance import cdist # For finding closest point

# === CONFIGURATION ===
# Input directories from previous processing steps
PROCESSED_RGB_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/RGB_IMAGES"
PROCESSED_VEIN_COORDS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/VEIN_COORDS" # Need vein coords to identify base
GROUND_TRUTH_MASKS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/GROUND_TRUTH_MASKS" # The 0/1/2 mask

# Output directory for generated geodesic masks (as .npy)
GEODESIC_MASKS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/GEODESIC_MASKS"
# Output directory for visual checks of geodesic maps
GEODESIC_OVERLAY_PLOTS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/GEODESIC_OVERLAY_PLOTS"
# Debug output directory for intermediate masks (useful if issues persist)
DEBUG_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/DEBUG_GEODESIC"

# Ensure output directories exist
os.makedirs(GEODESIC_MASKS_DIR, exist_ok=True)
os.makedirs(GEODESIC_OVERLAY_PLOTS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True) # Create debug directory

# Plotting parameters for visual checks
PLOT_DPI = 100
GEODESIC_CMAP = 'inferno' # Colormap for geodesic distance
OVERLAY_ALPHA = 1.0 # Fully opaque overlay as requested

# Neighbors for BFS (8-connectivity)
NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1), ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

# === HELPERS ===
def calculate_petiolar_base_coords(vein_coords):
    """
    Calculates the average of the first and last coordinate pair as the petiolar base.
    Handles cases with fewer than 2 points gracefully.
    """
    if vein_coords.shape[0] < 1:
        return None
    elif vein_coords.shape[0] == 1:
        return vein_coords[0]
    else:
        # Average of the first and last point
        return (vein_coords[0] + vein_coords[-1]) / 2.0

# === MAIN PROCESSING ===
# Get a list of all processed RGB image filenames (they define the common set of samples)
image_files = sorted([f for f in os.listdir(PROCESSED_RGB_DIR) if f.lower().endswith(".png")])

print(f"Generating 'geodesic distance along veins' masks for {len(image_files)} images...")

for fname in tqdm(image_files, desc="Creating geodesic masks"):
    base_name = os.path.splitext(fname)[0] # e.g., 'sample_id_1'

    rgb_path = os.path.join(PROCESSED_RGB_DIR, fname)
    vein_coords_path = os.path.join(PROCESSED_VEIN_COORDS_DIR, f"{base_name}_veins.txt")
    segmentation_mask_path = os.path.join(GROUND_TRUTH_MASKS_DIR, f"{base_name}.png") # The 0/1/2 mask
    
    geodesic_output_npy_path = os.path.join(GEODESIC_MASKS_DIR, f"{base_name}_geodesic_mask.npy")
    geodesic_overlay_plot_path = os.path.join(GEODESIC_OVERLAY_PLOTS_DIR, f"{base_name}_geodesic_overlay.png")

    # Skip if NPY file already exists
    if os.path.exists(geodesic_output_npy_path):
        continue

    # Validate existence of necessary files
    if not os.path.exists(rgb_path):
        print(f"⚠️ Warning: RGB image '{rgb_path}' not found, skipping geodesic mask generation for {base_name}.")
        continue
    if not os.path.exists(segmentation_mask_path):
        print(f"⚠️ Warning: Segmentation mask '{segmentation_mask_path}' not found, skipping geodesic mask generation for {base_name}.")
        continue

    try:
        # Load the processed RGB image to get dimensions
        img_pil = Image.open(rgb_path)
        width, height = img_pil.size

        # Load the ground truth segmentation mask (0=BG, 1=Blade, 2=Vein)
        seg_mask = np.array(Image.open(segmentation_mask_path).convert('L'))

        # Create a binary mask of ONLY vein pixels (boolean array)
        vein_mask_binary = (seg_mask == 2) 

        # --- DEBUGGING: Save vein mask binary image ---
        Image.fromarray(vein_mask_binary.astype(np.uint8) * 255).save(os.path.join(DEBUG_DIR, f"DEBUG_{base_name}_vein_mask_binary.png"))

        # Initialize geodesic map with infinity
        geodesic_map_dist = np.full((height, width), np.inf, dtype=np.float32)

        # Handle cases with no vein pixels detected
        if not np.any(vein_mask_binary):
            print(f"⚠️ Warning: No vein pixels found in segmentation mask for {base_name}. Creating all-zeros geodesic mask.")
            np.save(geodesic_output_npy_path, np.zeros((height, width), dtype=np.float32))
            # Create a simple plot to indicate missing data
            fig, ax = plt.subplots(figsize=(width / PLOT_DPI, height / PLOT_DPI), dpi=PLOT_DPI)
            ax.imshow(img_pil)
            ax.text(0.5, 0.5, 'NO VEIN PIXELS / GEODESIC MAP', transform=ax.transAxes,
                    fontsize=10, color='red', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
            ax.axis('off')
            plt.tight_layout(pad=0)
            fig.savefig(geodesic_overlay_plot_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            continue
        
        # Load vein coordinates to determine petiolar base
        if not os.path.exists(vein_coords_path):
            print(f"⚠️ Warning: Vein coordinates '{vein_coords_path}' not found for {base_name}. Cannot determine petiolar base for geodesic.")
            np.save(geodesic_output_npy_path, np.zeros((height, width), dtype=np.float32)) # Save empty
            # Create a simple plot to indicate missing data
            fig, ax = plt.subplots(figsize=(width / PLOT_DPI, height / PLOT_DPI), dpi=PLOT_DPI)
            ax.imshow(img_pil)
            ax.text(0.5, 0.5, 'NO VEIN COORDS / GEODESIC MAP', transform=ax.transAxes,
                    fontsize=10, color='red', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
            ax.axis('off')
            plt.tight_layout(pad=0)
            fig.savefig(geodesic_overlay_plot_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            continue

        vein_coords_raw = np.loadtxt(vein_coords_path)
        if vein_coords_raw.ndim == 1:
            vein_coords_raw = vein_coords_raw[np.newaxis, :] # Ensure 2D for single points

        petiolar_base_point_xy = calculate_petiolar_base_coords(vein_coords_raw)

        if petiolar_base_point_xy is None:
            print(f"⚠️ Warning: Could not determine petiolar base for {base_name}. Vein coordinates too sparse. Creating empty geodesic mask.")
            np.save(geodesic_output_npy_path, np.zeros((height, width), dtype=np.float32)) # Save empty
            # Create a simple plot to indicate missing data
            fig, ax = plt.subplots(figsize=(width / PLOT_DPI, height / PLOT_DPI), dpi=PLOT_DPI)
            ax.imshow(img_pil)
            ax.text(0.5, 0.5, 'PETIOLAR BASE UNDETERMINED / GEODESIC MAP', transform=ax.transAxes,
                    fontsize=10, color='red', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
            ax.axis('off')
            plt.tight_layout(pad=0)
            fig.savefig(geodesic_overlay_plot_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            continue

        # Find the vein pixel closest to the calculated petiolar base point (this is our BFS start)
        vein_pixel_coords_yx = np.argwhere(vein_mask_binary) # Returns (N, 2) array of (y, x) coords
        
        if len(vein_pixel_coords_yx) == 0: # Double check if somehow vein_mask_binary was empty
             print(f"⚠️ Warning: No vein pixels found after initial check for {base_name}. Creating all-zeros geodesic mask.")
             np.save(geodesic_output_npy_path, np.zeros((height, width), dtype=np.float32))
             continue

        # Convert vein_pixel_coords_yx to (x, y) for cdist
        vein_pixel_coords_xy = vein_pixel_coords_yx[:, [1, 0]] # Swap columns to (x, y)
        
        # Calculate distances from petiolar_base_point_xy to all vein pixels
        distances_to_vein_pixels = cdist(petiolar_base_point_xy[np.newaxis, :], vein_pixel_coords_xy)
        
        # Get the index of the closest vein pixel
        closest_vein_pixel_idx = np.argmin(distances_to_vein_pixels)
        start_pixel_y, start_pixel_x = vein_pixel_coords_yx[closest_vein_pixel_idx]

        # --- Perform BFS-like propagation (using your proven logic) ---
        q = collections.deque()

        # Initialize the distance at the start pixel to 0
        geodesic_map_dist[start_pixel_y, start_pixel_x] = 0
        q.append((start_pixel_y, start_pixel_x, 0)) # Store (y, x, current_distance)

        while q:
            curr_y, curr_x, curr_dist = q.popleft()

            for dy, dx in NEIGHBORS:
                ny, nx = curr_y + dy, curr_x + dx

                # Check bounds
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                    
                # ONLY THE FOLLOWING LINE IS THE CRITICAL CHANGE (back to your working logic)
                # Check if neighbor is a vein pixel AND its current distance is infinity (unvisited)
                if vein_mask_binary[ny, nx] and geodesic_map_dist[ny, nx] == np.inf:
                    geodesic_map_dist[ny, nx] = curr_dist + 1 # Each step increments distance by 1
                    q.append((ny, nx, curr_dist + 1))

        # --- Finalize Geodesic Map ---
        # Set non-vein pixels (still at np.inf or 0 from initialization) to 0
        # This also handles cases where a vein pixel might not have been reached by BFS if isolated
        geodesic_map_final = np.where(vein_mask_binary, geodesic_map_dist, 0.0)
        geodesic_map_final[np.isinf(geodesic_map_final)] = 0 # Ensure any remaining inf are 0 (e.g., truly unreachable vein pixels)

        # Normalize the non-zero values to 0-1
        max_val_in_veins = np.max(geodesic_map_final)
        if max_val_in_veins > 0:
            geodesic_map_normalized = geodesic_map_final / max_val_in_veins
        else:
            geodesic_map_normalized = geodesic_map_final # All zeros if no max_val (e.g., single pixel vein or isolated start)

        # --- DEBUGGING: Save normalized geodesic map ---
        Image.fromarray((geodesic_map_normalized * 255).astype(np.uint8)).save(os.path.join(DEBUG_DIR, f"DEBUG_{base_name}_geodesic_map_normalized.png"))

        # Save the generated geodesic mask as .npy
        np.save(geodesic_output_npy_path, geodesic_map_normalized.astype(np.float32))

        # === Generate Geodesic Overlay Plot ===
        fig, ax = plt.subplots(figsize=(width / PLOT_DPI, height / PLOT_DPI), dpi=PLOT_DPI)
        ax.imshow(img_pil)
        
        # Overlay the normalized geodesic map on top of the RGB image
        # Mask out zeros so only vein area is colored
        geodesic_display_map = np.ma.masked_where(geodesic_map_normalized == 0, geodesic_map_normalized)
        
        im = ax.imshow(geodesic_display_map, cmap=GEODESIC_CMAP, alpha=OVERLAY_ALPHA, vmin=0, vmax=1)
        
        # Add a colorbar for reference
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Geodesic Distance (0-1)')

        # Plot petiolar base point and BFS start point for visual check
        ax.plot(petiolar_base_point_xy[0], petiolar_base_point_xy[1], 'o', color='yellow', markersize=8, label='Calc. Petiolar Base')
        ax.plot(start_pixel_x, start_pixel_y, 'x', color='lime', markersize=10, markeredgewidth=2, label='BFS Start Pixel')
        ax.legend()

        ax.axis('off')
        ax.set_title(f"Geodesic (Along Veins) for: {base_name}", fontsize=10)
        plt.tight_layout(pad=0)
        fig.savefig(geodesic_overlay_plot_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig) # Explicitly close the figure to free memory

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

print(f"\n✅ All 'geodesic distance along veins' masks generated and saved to '{GEODESIC_MASKS_DIR}'.")
print(f"✅ Geodesic overlay plots saved to '{GEODESIC_OVERLAY_PLOTS_DIR}'.")
print(f"✅ Debugging images saved to '{DEBUG_DIR}' for further inspection.")