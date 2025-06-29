import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
import cv2
import os
import matplotlib.cm as cm # For colormaps

# Ensure the ect library is installed and accessible
try:
    from ect import ECT, EmbeddedGraph
except ImportError:
    print("Error: The 'ect' library is not found. Please ensure it's installed and accessible.")
    print("Add its directory to PYTHONPATH or install it correctly (e.g., pip install ect-morphology).")
    sys.exit(1)

# --- Configuration Parameters ---
BOUND_RADIUS = 1
NUM_ECT_DIRECTIONS = 180 # INCREASED ECT RESOLUTION
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS)
IMAGE_SIZE = (256, 256) # Output size for all images (masks, ECT, RGB)

# *** PRIMARY OUTPUT DIRECTORY FOR ALL MODEL INPUTS AND RELATED VIZ ***
MODEL_INPUTS_BASE_DIR = Path("model_inputs/")

# Subdirectories for the TWO primary model inputs
MODEL_RADIAL_ECT_DIR = MODEL_INPUTS_BASE_DIR / "radial_ects" # Grayscale ECTs for CNN
MODEL_SHAPE_MASK_DIR = MODEL_INPUTS_BASE_DIR / "shape_masks" # Grayscale Shape Masks for CNN

# Subdirectories for visualization outputs (now also under model_inputs)
MODEL_COMBINED_VIZ_DIR = MODEL_INPUTS_BASE_DIR / "combined_viz"
MODEL_ORIENTED_RGB_DIR = MODEL_INPUTS_BASE_DIR / "oriented_rgb"

MODEL_METADATA_FILE = MODEL_INPUTS_BASE_DIR / "metadata.csv"

# --- Input Mask Pixel Values ---
# These are the values present in your raw FINAL_ALIGNED_LEAVES_512x512 masks
BACKGROUND_PIXEL = 0
BLADE_PIXEL = 1
VEIN_PIXEL = 2

# --- Output Grayscale Values for Saved Masks (for visual clarity) ---
# These are the gray levels to use when saving the shape_masks and coloring oriented_rgb
MASK_BACKGROUND_GRAY = 0    # Black
MASK_BLADE_GRAY = 128       # Gray
MASK_VEIN_GRAY = 255        # White

# --- Helper Functions for Transformations ---

def apply_transformation_with_affine_matrix(points: np.ndarray, affine_matrix: np.ndarray):
    """
    Applies a 3x3 affine matrix to a set of 2D points.
    Points are expected as (N, 2) array.
    """
    if points.size == 0: # Handle empty array specifically
        return np.array([])
    
    # Ensure points are 2D (N, 2)
    if points.ndim == 1:
        if points.shape[0] == 2: # Single point like [x, y]
            points = points.reshape(1, 2)
        else:
            raise ValueError(f"Input 'points' is 1D but not a single (x,y) pair. Got shape: {points.shape}")
    
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Input 'points' must be a (N, 2) array. Got shape: {points.shape}")

    if affine_matrix.shape != (3, 3):
        raise ValueError(f"Input 'affine_matrix' must be (3, 3). Got shape: {affine_matrix.shape}")

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    if points_homogeneous.shape[1] != affine_matrix.T.shape[0]:
        raise ValueError(f"matmul: Input operand 1 has a mismatch in its core dimension 0. Expected {points_homogeneous.shape[1]}, got {affine_matrix.T.shape[0]}.")

    transformed_homogeneous = points_homogeneous @ affine_matrix.T
    return transformed_homogeneous[:, :2] # Return only the 2D coordinates

def find_robust_affine_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray):
    """
    Finds a robust affine transformation matrix between source and destination points.
    It attempts to find 3 non-collinear points for cv2.getAffineTransform.
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        # Fallback for too few points: if 0 points, return identity for 0 points.
        if len(src_points) == 0:
            return np.eye(3)
        raise ValueError(f"Need at least 3 points to compute affine transformation. Got {len(src_points)}.")

    # Loop to find 3 non-collinear points
    chosen_src_pts = []
    chosen_dst_pts = []
    
    # Iterate through all combinations of 3 points
    for i in range(len(src_points)):
        for j in range(i + 1, len(src_points)):
            for k in range(j + 1, len(src_points)):
                p1_src, p2_src, p3_src = src_points[[i, j, k]]
                
                # Calculate area using cross product / determinant method for 2D points
                area_val = (p1_src[0] - p3_src[0]) * (p2_src[1] - p1_src[1]) - \
                               (p1_src[0] - p2_src[0]) * (p3_src[1] - p1_src[1])
                
                if np.abs(area_val) > 1e-6: # Use a small epsilon to check for non-collinearity
                    chosen_src_pts = np.float32([p1_src, p2_src, p3_src])
                    chosen_dst_pts = np.float32([dst_points[i], dst_points[j], dst_points[k]])
                    break
            if len(chosen_src_pts) == 3:
                break
        if len(chosen_src_pts) == 3:
            break

    if len(chosen_src_pts) < 3:
        raise ValueError("Could not find 3 non-collinear points for affine transformation. Shape is likely degenerate or a line.")

    M_2x3 = cv2.getAffineTransform(chosen_src_pts, chosen_dst_pts)
    
    if M_2x3.shape != (2, 3):
        raise ValueError(f"cv2.getAffineTransform returned a non-(2,3) matrix: {M_2x3.shape}")

    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    
    return affine_matrix_3x3

# --- Central Coordinate Transformation Function (Adjusted for 90 deg CCW + Reflection) ---
def ect_coords_to_pixels(coords_ect: np.ndarray, image_size: tuple, bound_radius: float):
    """
    Transforms coordinates from ECT space (mathematical, Y-up, origin center, range [-R, R])
    to image pixel space (Y-down, origin top-left, range [0, IMAGE_SIZE]).
    
    This function applies a 90-degree counter-clockwise rotation relative to ECT's North-up
    orientation, and then a reflection (effectively across the new X-axis if you consider
    the rotation).
    
    Transformation logic (mathematical conceptual coordinates, before pixel conversion):
    1. 90-degree CCW rotation: (x_ect, y_ect) -> (-y_ect, x_ect)
    2. Reflection (e.g., horizontal flip of the rotated result): (-y_ect, x_ect) -> (y_ect, x_ect)
    
    Then map (y_ect, x_ect) to pixel coordinates (Y-down).
    
    Args:
        coords_ect (np.ndarray): N x 2 array of points in ECT space (where tip is typically positive Y).
        image_size (tuple): (width, height) of the target image.
        bound_radius (float): The bound radius used for ECT scaling.
    
    Returns:
        np.ndarray: Transformed N x 2 array of points in pixel space (integers).
    """
    if len(coords_ect) == 0:
        return np.array([])
    
    # After 90-degree CCW rotation and reflection, the mapping is:
    # ECT's Y-coordinate becomes the conceptual X-coordinate.
    # ECT's X-coordinate becomes the conceptual Y-coordinate.
    display_x_conceptual = coords_ect[:, 1]  # ECT Y maps to conceptual X
    display_y_conceptual = coords_ect[:, 0]  # ECT X maps to conceptual Y

    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2 

    # Map to pixel coordinates. Remember image Y-axis is typically "down".
    pixel_x = (display_x_conceptual * scale_factor + offset_x).astype(int)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y).astype(int) # Negate display_y_conceptual for Y-down mapping
    
    return np.column_stack((pixel_x, pixel_y))


# --- Visualization Functions ---

def save_grayscale_shape_mask(transformed_blade_pixels: np.ndarray, transformed_vein_pixels: np.ndarray, save_path: Path):
    """
    Saves a grayscale image representing the transformed leaf mask.
    Pixels are filled based on transformed_blade_pixels and transformed_vein_pixels.
    These pixels are expected to be in ECT space (origin center, Y-up) and will be
    transformed to image pixels here using ect_coords_to_pixels.
    """
    img = Image.new("L", IMAGE_SIZE, MASK_BACKGROUND_GRAY) # Initialize with background color

    # Fill blade pixels
    if transformed_blade_pixels is not None and transformed_blade_pixels.size > 0:
        blade_pixel_coords = ect_coords_to_pixels(transformed_blade_pixels, IMAGE_SIZE, BOUND_RADIUS)
        for x, y in blade_pixel_coords:
            if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]:
                img.putpixel((x, y), MASK_BLADE_GRAY)

    # Fill vein pixels (overwriting blade pixels if they overlap, as veins are foreground)
    if transformed_vein_pixels is not None and transformed_vein_pixels.size > 0:
        vein_pixel_coords = ect_coords_to_pixels(transformed_vein_pixels, IMAGE_SIZE, BOUND_RADIUS)
        for x, y in vein_pixel_coords:
            if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]:
                img.putpixel((x, y), MASK_VEIN_GRAY)
    
    img.save(save_path)

def save_radial_ect_image(ect_result, save_path: Path, cmap_name: str = "gray"):
    """
    Saves the radial ECT plot as an image with the specified colormap.
    For model input, 'gray' cmap is usually preferred.
    """
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap=cmap_name)
    ax.set_theta_zero_location("N") # North (top)
    ax.set_theta_direction(-1) # Clockwise (standard for polar plots)
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, save_path: Path,
                                     blade_color=(255, 255, 255), blade_alpha=0.2,
                                     vein_points_transformed: np.ndarray = None,
                                     vein_color=(255, 255, 255), vein_size=1,
                                     transformed_blade_pixels: np.ndarray = None):
    """
    Creates a combined visualization by overlaying transformed elements (blade, veins)
    onto the ECT image. All overlaid elements are transformed to pixel space.
    Note: Landmarks are NOT drawn here as this script generates prediction inputs.
    """
    try:
        ect_img = Image.open(ect_image_path).convert("RGBA")
        img_width, img_height = ect_img.size

        composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw_composite = ImageDraw.Draw(composite_overlay)

        # Draw blade pixels 
        if transformed_blade_pixels is not None and transformed_blade_pixels.size > 0:
            blade_pixel_coords = ect_coords_to_pixels(transformed_blade_pixels, IMAGE_SIZE, BOUND_RADIUS)
            for x, y in blade_pixel_coords:
                if 0 <= x < img_width and 0 <= y < img_height:
                    draw_composite.rectangle([x, y, x, y], fill=(blade_color[0], blade_color[1], blade_color[2], int(255 * blade_alpha)))

        # Draw vein points 
        if vein_points_transformed is not None and vein_points_transformed.size > 0:
            vein_pixel_coords = ect_coords_to_pixels(vein_points_transformed, IMAGE_SIZE, BOUND_RADIUS)
            vein_fill_color = (vein_color[0], vein_color[1], vein_color[2], int(255 * blade_alpha))
            for x, y in vein_pixel_coords:
                if 0 <= x < img_width and 0 <= y < img_height:
                    draw_composite.ellipse([x - vein_size//2, y - vein_size//2,
                                            x + vein_size//2, y + vein_size//2],
                                            fill=vein_fill_color)
        
        final_combined_img = Image.alpha_composite(ect_img, composite_overlay).convert("RGB")
        final_combined_img.save(save_path)

    except FileNotFoundError:
        print(f"Error: ECT image file not found: {ect_image_path}")
    except Exception as e:
        print(f"Error creating combined visualization for {ect_image_path.stem}: {e}")


# --- Naming Standardization Function ---
def standardize_wolfskill_name(leaf_id: str) -> str:
    """
    Standardizes the 'name' part of a WOLFSKILL leaf_id if it matches
    certain problematic patterns (case-insensitively) or is any variation
    of 'UNKNOWN_CULTIVAR'.
    Expected format: COLLECTION_NAME_ID
    Example: WOLFSKILL_nan_9726 -> WOLFSKILL_UNKNOWN_CULTIVAR_9726
    WOLFSKILL_Unknown_Cultivar_6324 -> WOLFSKILL_UNKNOWN_CULTIVAR_6324
    """
    parts = leaf_id.split('_')
    if len(parts) >= 3 and parts[0].upper() == 'WOLFSKILL':
        collection = parts[0]
        name_part = parts[1]
        identifier = '_'.join(parts[2:]) # Rejoin the rest in case of multiple underscores

        # Define problematic names (lowercase for comparison) and also include
        # the standardized 'unknown_cultivar' itself in the set to catch all variants.
        problematic_names_to_match = {'nan', 'unknown_cultivar', 'unkown_cultivar'} 
        
        # Check if the current name_part (converted to lowercase) is in our set of problematic names
        # OR if its uppercase version is exactly 'UNKNOWN_CULTIVAR'
        if name_part.lower() in problematic_names_to_match or name_part.upper() == 'UNKNOWN_CULTIVAR':
            return f"{collection}_UNKNOWN_CULTIVAR_{identifier}" # Always use the exact desired string
    return leaf_id

# --- Main Data Processing Logic ---

def process_leaf_data_for_prediction(raw_input_dir: Path, clear_existing_model_data: bool = True):
    print(f"Starting processing of leaf data for prediction from: {raw_input_dir}")
    print(f"All outputs will be saved to: {MODEL_INPUTS_BASE_DIR}")

    # Clear ONLY the MODEL_INPUTS_BASE_DIR
    if clear_existing_model_data and MODEL_INPUTS_BASE_DIR.exists():
        print(f"Clearing existing model inputs directory: {MODEL_INPUTS_BASE_DIR}")
        shutil.rmtree(MODEL_INPUTS_BASE_DIR)

    # Create all output directories under MODEL_INPUTS_BASE_DIR
    MODEL_SHAPE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RADIAL_ECT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_COMBINED_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ORIENTED_RGB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories:\n  - {MODEL_SHAPE_MASK_DIR}\n  - {MODEL_RADIAL_ECT_DIR}\n  - {MODEL_COMBINED_VIZ_DIR}\n  - {MODEL_ORIENTED_RGB_DIR}")

    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)

    metadata_records = []
    processed_count = 0
    skipped_count = 0

    mask_files_full_paths = {}
    rgb_files_full_paths = {}

    # Collect mask files and apply standardization
    for f in (raw_input_dir / "MASKS").glob("*_mask.png"): # Explicitly look for _mask.png
        original_leaf_id = f.stem.replace('_mask', '')
        standardized_leaf_id = standardize_wolfskill_name(original_leaf_id)
        mask_files_full_paths[standardized_leaf_id] = f

    # Collect RGB files and apply standardization
    for f in (raw_input_dir / "RGB_CROPS").glob("*_rgb_crop.png"): # Explicitly look for _rgb_crop.png
        original_leaf_id = f.stem.replace('_rgb_crop', '')
        standardized_leaf_id = standardize_wolfskill_name(original_leaf_id)
        # Note: If a mask file exists for a standardized_leaf_id but no corresponding
        # RGB file exists (or vice-versa after standardization), the processing logic
        # will handle that (e.g., RGB will be skipped if not found).
        rgb_files_full_paths[standardized_leaf_id] = f

    # Determine which leaves to process (all masks found)
    leaves_to_process_ids = list(mask_files_full_paths.keys())
    
    total_files = len(leaves_to_process_ids)

    if total_files == 0:
        print(f"No .png mask files found in '{raw_input_dir / 'MASKS'}'. Exiting.")
        return

    print(f"Found {total_files} .png mask files to process (after name standardization).")

    # Create a temporary directory for inferno ECTs within MODEL_INPUTS_BASE_DIR
    ect_inferno_temp_dir = MODEL_INPUTS_BASE_DIR / "temp_ect_inferno"
    ect_inferno_temp_dir.mkdir(parents=True, exist_ok=True)

    for i, leaf_id in enumerate(leaves_to_process_ids):
        mask_file_path = mask_files_full_paths[leaf_id] # leaf_id is already standardized here

        output_image_name = f"{leaf_id}.png" # Standard output filename
        
        # Paths for all outputs, now all under MODEL_INPUTS_BASE_DIR
        model_mask_path = MODEL_SHAPE_MASK_DIR / output_image_name
        model_ect_grayscale_path = MODEL_RADIAL_ECT_DIR / output_image_name # Primary CNN input
        model_combined_viz_path = MODEL_COMBINED_VIZ_DIR / output_image_name
        model_oriented_rgb_path = MODEL_ORIENTED_RGB_DIR / output_image_name
        
        # Temporary path for ECT with inferno colormap (for combined viz only)
        ect_inferno_temp_path = ect_inferno_temp_dir / output_image_name

        print(f"Processing leaf shape {i+1}/{total_files} ({leaf_id})")

        # Initialize variables for current leaf, default to empty/None in case of early errors
        full_leaf_contour_points = None
        raw_blade_pixels = np.array([])
        raw_vein_pixels = np.array([])
        G = None
        ect_affine_matrix = None
        transformed_blade_pixels = np.array([])
        transformed_vein_pixels = np.array([])

        try:
            # --- Load Mask and Extract Contours/Pixels ---
            mask_img_cv = cv2.imread(str(mask_file_path), cv2.IMREAD_GRAYSCALE)
            if mask_img_cv is None:
                raise ValueError(f"Could not load mask image: {mask_file_path}")
            if mask_img_cv.ndim == 3: # Ensure it's truly grayscale
                mask_img_cv = cv2.cvtColor(mask_img_cv, cv2.COLOR_BGR2GRAY)

            full_leaf_binary_mask = np.zeros_like(mask_img_cv, dtype=np.uint8)
            full_leaf_binary_mask[mask_img_cv == BLADE_PIXEL] = 255
            full_leaf_binary_mask[mask_img_cv == VEIN_PIXEL] = 255

            full_contours, _ = cv2.findContours(full_leaf_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not full_contours:
                raise ValueError("No full leaf contours found in the mask image.")

            main_full_leaf_contour = max(full_contours, key=cv2.contourArea)
            full_leaf_contour_points = main_full_leaf_contour.squeeze().astype(np.float64)

            # Ensure full_leaf_contour_points is (N, 2)
            if full_leaf_contour_points.ndim == 1:
                if full_leaf_contour_points.shape[0] == 2:
                    full_leaf_contour_points = full_leaf_contour_points.reshape(1, 2)
                else:
                    raise ValueError(f"Unexpected full_leaf_contour_points dimension after squeeze: {full_leaf_contour_points.shape}. Expected (N, 2).")
            if full_leaf_contour_points.shape[0] < 3:
                raise ValueError(f"Full leaf shape has too few points ({full_leaf_contour_points.shape[0]}) to form a valid polygon.")


            blade_y, blade_x = np.where(mask_img_cv == BLADE_PIXEL)
            if len(blade_x) > 0:
                raw_blade_pixels = np.column_stack((blade_x, blade_y)).astype(np.float64)
            else:
                raw_blade_pixels = np.array([], dtype=np.float64).reshape(0, 2) # Ensure (0, 2) for empty

            vein_y, vein_x = np.where(mask_img_cv == VEIN_PIXEL)
            if len(vein_x) > 0:
                raw_vein_pixels = np.column_stack((vein_x, vein_y)).astype(np.float64)
            else:
                raw_vein_pixels = np.array([], dtype=np.float64).reshape(0, 2) # Ensure (0, 2) for empty

            
            # --- ECT Processing & Affine Matrix Derivation ---
            G = EmbeddedGraph()
            G.add_cycle(full_leaf_contour_points)

            original_G_coord_matrix = np.copy(G.coord_matrix)

            G.center_coordinates(center_type="origin")
            G.transform_coordinates() # This is where the internal ECT alignment happens (e.g., to positive Y-axis)
            G.scale_coordinates(BOUND_RADIUS)

            try:
                ect_affine_matrix = find_robust_affine_transformation_matrix(original_G_coord_matrix, G.coord_matrix)
            except ValueError as e:
                raise RuntimeError(f"Could not derive affine transformation matrix for ECT: {e}")
            
            # Transform blade and veins using the ECT's derived affine matrix
            if raw_blade_pixels.size > 0:
                transformed_blade_pixels = apply_transformation_with_affine_matrix(
                    raw_blade_pixels, ect_affine_matrix
                )
            if raw_vein_pixels.size > 0:
                transformed_vein_pixels = apply_transformation_with_affine_matrix(
                    raw_vein_pixels, ect_affine_matrix
                )

            if np.all(G.coord_matrix == 0):
                raise ValueError("Degenerate full leaf shape (all points at origin after scaling)")
            if G.coord_matrix.shape[0] < 3:
                raise ValueError(f"Processed full leaf shape has too few points ({G.coord_matrix.shape[0]}) to form a valid polygon.")

            ect_result = ect_calculator.calculate(G) # Calculate ECT using the ECT-aligned G.coord_matrix


            # --- Process RGB Image for oriented_rgb visualization ---
            oriented_rgb_success = False
            # Look up RGB path using the standardized leaf_id
            rgb_image_path = rgb_files_full_paths.get(leaf_id) 
            if rgb_image_path and rgb_image_path.exists():
                rgb_img_cv = cv2.imread(str(rgb_image_path))
                if rgb_img_cv is not None:
                    # Define the ECT-to-Output-Pixel matrix (maps ECT coords to display coords)
                    scale_to_pixels = IMAGE_SIZE[0] / (2 * BOUND_RADIUS)
                    offset_x_pixels = IMAGE_SIZE[0] / 2
                    offset_y_pixels = IMAGE_SIZE[1] / 2
                    
                    ect_space_to_output_pixel_matrix_2x3 = np.array([
                        [0, scale_to_pixels, offset_x_pixels],  # ECT_Y -> pixel_X
                        [-scale_to_pixels, 0, offset_y_pixels]   # -ECT_X -> pixel_Y (due to Y-down pixel system)
                    ], dtype=np.float32)

                    # Combine the ECT affine matrix (which aligns the leaf) with the
                    # ECT-to-Output-Pixel matrix (which scales and positions in the output image)
                    # This ensures the RGB image is transformed and then correctly placed/oriented.
                    final_transformation_matrix_3x3 = np.vstack([ect_space_to_output_pixel_matrix_2x3, [0, 0, 1]]) @ ect_affine_matrix
                    M_2x3_for_warp = final_transformation_matrix_3x3[:2, :]
                    
                    oriented_rgb_img_cv = cv2.warpAffine(rgb_img_cv, M_2x3_for_warp, IMAGE_SIZE, flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
                    
                    # --- Apply mask-based coloring to oriented_rgb_img_cv ---
                    oriented_mask_img = Image.new("L", IMAGE_SIZE, MASK_BACKGROUND_GRAY)
                    if transformed_blade_pixels.size > 0:
                        blade_pixel_coords_oriented = ect_coords_to_pixels(transformed_blade_pixels, IMAGE_SIZE, BOUND_RADIUS)
                        for px, py in blade_pixel_coords_oriented:
                            if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
                                oriented_mask_img.putpixel((px, py), MASK_BLADE_GRAY)
                    if transformed_vein_pixels.size > 0:
                        vein_pixel_coords_oriented = ect_coords_to_pixels(transformed_vein_pixels, IMAGE_SIZE, BOUND_RADIUS)
                        for px, py in vein_pixel_coords_oriented:
                            if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
                                oriented_mask_img.putpixel((px, py), MASK_VEIN_GRAY)
                    
                    oriented_mask_array = np.array(oriented_mask_img)

                    oriented_rgb_img_final = oriented_rgb_img_cv.copy()
                    
                    plasma_cmap = cm.get_cmap('plasma')

                    for y in range(IMAGE_SIZE[1]):
                        for x in range(IMAGE_SIZE[0]):
                            mask_val = oriented_mask_array[y, x]
                            if mask_val == MASK_BACKGROUND_GRAY:
                                oriented_rgb_img_final[y, x] = [0, 0, 0] # Black background
                            elif mask_val == MASK_VEIN_GRAY:
                                b, g, r = oriented_rgb_img_final[y, x]
                                grayscale_intensity = 0.2989 * r + 0.5870 * g + 0.1140 * b
                                normalized_intensity = grayscale_intensity / 255.0
                                
                                cmap_rgba = plasma_cmap(normalized_intensity)
                                new_rgb = (np.array(cmap_rgba[:3]) * 255).astype(np.uint8)
                                oriented_rgb_img_final[y, x] = [new_rgb[2], new_rgb[1], new_rgb[0]] # BGR for OpenCV
                    
                    # No landmarks are drawn here since this is for prediction input,
                    # and landmarks are the model's output.

                    cv2.imwrite(str(model_oriented_rgb_path), oriented_rgb_img_final)
                    oriented_rgb_success = True
                else:
                    print(f"Warning: Could not load RGB image: {rgb_image_path}. Oriented RGB will not be saved.")
            if not oriented_rgb_success:
                model_oriented_rgb_path = None # Ensure it's None if not successfully processed


            # Nested try for image saving
            try:
                # Save primary model inputs (mask and ECT)
                save_grayscale_shape_mask(transformed_blade_pixels, transformed_vein_pixels, model_mask_path)
                save_radial_ect_image(ect_result, model_ect_grayscale_path, cmap_name="gray")
                
                # Save ECT for combined visualization (inferno) to a temporary location
                save_radial_ect_image(ect_result, ect_inferno_temp_path, cmap_name="inferno")

                # Save combined visualization
                create_combined_viz_from_images(ect_inferno_temp_path, model_combined_viz_path,
                                                 blade_color=(255, 255, 255),
                                                 blade_alpha=0.2,
                                                 vein_points_transformed=transformed_vein_pixels,
                                                 vein_color=(255, 255, 255),
                                                 transformed_blade_pixels=transformed_blade_pixels)

            except Exception as e:
                print(f"  Error saving images for '{leaf_id}': {e}. Marking as invalid.")
                skipped_count += 1
                metadata_records.append({
                    "leaf_id": leaf_id,
                    "raw_mask_file_path": str(mask_file_path),
                    "is_processed_valid": False,
                    "reason_skipped": f"Image saving failed: {e}",
                    "num_raw_points_full_leaf": 0,
                    "num_processed_points_full_leaf": 0,
                    "file_shape_mask": "",
                    "file_radial_ect": "",
                    "file_combined_viz": "",
                    "file_oriented_rgb": "",
                    "has_veins": False, "has_blade": False
                })
                # Clean up temporary ECT inferno file if it exists
                if ect_inferno_temp_path.exists():
                    os.remove(ect_inferno_temp_path)
                continue
            
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_mask_file_path": str(mask_file_path),
                "is_processed_valid": True,
                "reason_skipped": "",
                "num_raw_points_full_leaf": full_leaf_contour_points.shape[0],
                "num_processed_points_full_leaf": G.coord_matrix.shape[0],
                "file_shape_mask": str(model_mask_path.relative_to(MODEL_INPUTS_BASE_DIR)),
                "file_radial_ect": str(model_ect_grayscale_path.relative_to(MODEL_INPUTS_BASE_DIR)),
                "file_combined_viz": str(model_combined_viz_path.relative_to(MODEL_INPUTS_BASE_DIR)),
                "file_oriented_rgb": str(model_oriented_rgb_path.relative_to(MODEL_INPUTS_BASE_DIR)) if oriented_rgb_success else "",
                "has_veins": raw_vein_pixels.size > 0,
                "has_blade": raw_blade_pixels.size > 0
            })
            processed_count += 1
            # Clean up temporary ECT inferno file
            if ect_inferno_temp_path.exists():
                os.remove(ect_inferno_temp_path)

        except Exception as e:
            print(f"  Skipping '{leaf_id}' due to an error during processing: {e}")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_mask_file_path": str(mask_file_path),
                "is_processed_valid": False,
                "reason_skipped": str(e),
                "num_raw_points_full_leaf": 0,
                "num_processed_points_full_leaf": 0,
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": "", "file_oriented_rgb": "",
                "has_veins": False, "has_blade": False
            })
            # Clean up temporary ECT inferno file if it exists
            if ect_inferno_temp_path.exists():
                os.remove(ect_inferno_temp_path)


    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(MODEL_METADATA_FILE, index=False)
        print(f"\nProcessing complete. Generated metadata file: {MODEL_METADATA_FILE}")
    else:
        print("\nProcessing complete. No metadata records to save.")

    print(f"Total leaves processed: {processed_count}")
    print(f"Total leaves skipped: {skipped_count}")

    # Remove the temporary directory for inferno ECTs after all processing is done
    if ect_inferno_temp_dir.exists():
        print(f"Cleaning up temporary directory: {ect_inferno_temp_dir}")
        shutil.rmtree(ect_inferno_temp_dir)

if __name__ == "__main__":
    # Set this to the directory containing your "MASKS" and "RGB_CROPS" subdirectories
    # This should be 'FINAL_ALIGNED_LEAVES_512x512'
    RAW_INPUT_DATA_DIR = Path("FINAL_ALIGNED_LEAVES_512x512")

    # This call will run the processing.
    # clear_existing_model_data=True will clear the entire 'model_inputs' directory before generating.
    process_leaf_data_for_prediction(RAW_INPUT_DATA_DIR, clear_existing_model_data=True)