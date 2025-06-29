import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
import cv2
import os

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

PROCESSED_DATA_OUTPUT_DIR = Path("processed_leaf_data/")
SHAPE_MASK_DIR = PROCESSED_DATA_OUTPUT_DIR / "shape_masks"
RADIAL_ECT_DIR = PROCESSED_DATA_OUTPUT_DIR / "radial_ects"
COMBINED_VIZ_DIR = PROCESSED_DATA_OUTPUT_DIR / "combined_viz"
ORIENTED_RGB_DIR = PROCESSED_DATA_OUTPUT_DIR / "oriented_rgb"
METADATA_FILE = PROCESSED_DATA_OUTPUT_DIR / "metadata.csv"
LANDMARK_DATA_FILE = Path("training_data_results.csv") 

BACKGROUND_PIXEL = 0
BLADE_PIXEL = 1
VEIN_PIXEL = 2

# Define grayscale values for the output mask file
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

    # --- CRITICAL DEBUGGING: Check affine_matrix shape at the point of call ---
    if affine_matrix.shape != (3, 3):
        print(f"DEBUG_AFFINE_MATRIX_SHAPE_AT_APPLY_CALL: affine_matrix input to apply_transformation_with_affine_matrix has unexpected shape: {affine_matrix.shape}")
        raise ValueError(f"Input 'affine_matrix' must be (3, 3). Got shape: {affine_matrix.shape}")

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # --- CRITICAL DEBUGGING: Check shapes immediately before matmul ---
    print(f"DEBUG_MATMUL_INPUTS_FINAL_CHECK: points_homogeneous shape: {points_homogeneous.shape}, affine_matrix.T shape: {affine_matrix.T.shape}")
    if points_homogeneous.shape[1] != affine_matrix.T.shape[0]:
        print(f"DEBUG_MATMUL_ERROR_FINAL: Mismatch in dimensions for matmul:")
        print(f"  points_homogeneous shape: {points_homogeneous.shape}")
        print(f"  affine_matrix.T shape: {affine_matrix.T.shape}")
        raise ValueError(f"matmul: Input operand 1 has a mismatch in its core dimension 0. Expected {points_homogeneous.shape[1]}, got {affine_matrix.T.shape[0]}.")
    # --- END CRITICAL DEBUGGING ---

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
    
    # --- CRITICAL DEBUGGING: Check M_2x3 shape ---
    if M_2x3.shape != (2, 3):
        print(f"DEBUG_AFFINE_ERROR_CV2_RETURN: cv2.getAffineTransform returned unexpected shape: {M_2x3.shape}")
        print(f"  Source points: {chosen_src_pts}")
        print(f"  Destination points: {chosen_dst_pts}")
        raise ValueError(f"cv2.getAffineTransform returned a non-(2,3) matrix: {M_2x3.shape}")
    # --- END CRITICAL DEBUGGING ---

    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    
    return affine_matrix_3x3

# --- Central Coordinate Transformation Function (Adjusted for 90 deg CCW + Reflection) ---
def ect_coords_to_pixels(coords_ect: np.ndarray, image_size: tuple, bound_radius: float):
    """
    Transforms coordinates from ECT space (mathematical, Y-up, origin center, range [-R, R])
    to image pixel space (Y-down, origin top-left, range [0, IMAGE_SIZE]).
    
    This function applies a 90-degree counter-clockwise rotation relative to ECT's North-up
    orientation, and then a reflection.
    
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
    
    # Step 1: Apply a 90-degree Counter-Clockwise rotation: (x_ect, y_ect) -> (-y_ect, x_ect)
    rotated_x = -coords_ect[:, 1] # -y_ect
    rotated_y = coords_ect[:, 0]  # x_ect

    # Step 2: Apply reflection. Assuming a horizontal reflection (flip across Y-axis of rotated coords)
    # This means negate the X component of the *rotated* coordinates.
    # (rotated_x, rotated_y) -> (-rotated_x, rotated_y)
    # So, display_x_conceptual = -(-coords_ect[:, 1]) = coords_ect[:, 1]
    # And display_y_conceptual = coords_ect[:, 0] (no change to Y from reflection)
    
    display_x_conceptual = coords_ect[:, 1]  # ECT Y maps to Display X (effectively flipped from previous)
    display_y_conceptual = coords_ect[:, 0]  # ECT X maps to Display Y

    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2 

    # Map to pixel coordinates. Remember image Y-axis is typically "down".
    pixel_x = (display_x_conceptual * scale_factor + offset_x).astype(int)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y).astype(int) # Negate display_y_conceptual for Y-down mapping
    
    return np.column_stack((pixel_x, pixel_y))


# --- Visualization Functions (unchanged, as they use ect_coords_to_pixels) ---

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

def save_grayscale_radial_ect(ect_result, save_path: Path):
    """
    Saves the radial ECT plot as a grayscale image.
    This plot's orientation is determined by the ECT library's internal logic
    and matplotlib's polar plot settings, which should be North-up already.
    """
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap="inferno") # Changed to inferno for better visual
    ax.set_theta_zero_location("N") # North (top)
    ax.set_theta_direction(-1) # Clockwise (standard for polar plots)
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, save_path: Path,
                                     blade_color=(255, 255, 255), blade_alpha=0.3,
                                     landmark_points_transformed: np.ndarray = None,
                                     landmark_color=(255, 255, 0), landmark_size=5,
                                     vein_points_transformed: np.ndarray = None,
                                     vein_color=(255, 255, 255), vein_size=1,
                                     transformed_blade_pixels: np.ndarray = None):
    """
    Creates a combined visualization by overlaying transformed elements (blade, veins, landmarks)
    onto the ECT image. All overlaid elements are transformed to pixel space.
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

        # Draw landmark points 
        if landmark_points_transformed is not None and landmark_points_transformed.size > 0:
            landmark_pixel_coords = ect_coords_to_pixels(landmark_points_transformed, IMAGE_SIZE, BOUND_RADIUS)
            for x, y in landmark_pixel_coords:
                if 0 <= x < img_width and 0 <= y < img_height:
                    draw_composite.ellipse([x - landmark_size//2, y - landmark_size//2,
                                            x + landmark_size//2, y + landmark_size//2],
                                            fill=landmark_color)
        
        final_combined_img = Image.alpha_composite(ect_img, composite_overlay).convert("RGB")
        final_combined_img.save(save_path)

    except FileNotFoundError:
        print(f"Error: ECT image file not found: {ect_image_path}")
    except Exception as e:
        print(f"Error creating combined visualization for {ect_image_path.stem}: {e}")


# --- Main Data Processing Logic ---

def process_raw_leaf_shapes(raw_input_dir: Path, output_base_dir: Path, landmark_data_file: Path, clear_existing_data: bool = True):
    print(f"Starting processing of raw leaf shapes from: {raw_input_dir}")
    print(f"Output will be saved to: {output_base_dir}")

    if clear_existing_data and output_base_dir.exists():
        print(f"Clearing existing output directory: {output_base_dir}")
        shutil.rmtree(output_base_dir)

    shape_mask_dir = output_base_dir / "shape_masks"
    radial_ect_dir = output_base_dir / "radial_ects"
    combined_viz_dir = output_base_dir / "combined_viz"
    oriented_rgb_dir = output_base_dir / "oriented_rgb"
    metadata_file = output_base_dir / "metadata.csv"

    shape_mask_dir.mkdir(parents=True, exist_ok=True)
    radial_ect_dir.mkdir(parents=True, exist_ok=True)
    combined_viz_dir.mkdir(parents=True, exist_ok=True)
    oriented_rgb_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories: {shape_mask_dir}, {radial_ect_dir}, {combined_viz_dir}, {oriented_rgb_dir}")

    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)

    metadata_records = []
    processed_count = 0
    skipped_count = 0

    landmark_df = pd.DataFrame()
    mask_files_full_paths = {} 
    rgb_files_full_paths = {} 

    for f in (raw_input_dir / "MASKS").glob("*.png"):
        mask_files_full_paths[f.stem.replace('_mask', '')] = f

    for f in (raw_input_dir / "RGB_CROPS").glob("*.png"):
        stem = f.stem.replace('_rgb_crop', '').replace('_crop', '') 
        rgb_files_full_paths[stem] = f

    if landmark_data_file.exists():
        try:
            landmark_df = pd.read_csv(landmark_data_file)
            
            landmark_df['CleanedLabelStem'] = landmark_df['Label'].apply(
                lambda x: Path(x).stem.replace('_overlay', '')
            )
            
            landmark_df_filtered = landmark_df[landmark_df['CleanedLabelStem'].isin(mask_files_full_paths.keys())]
            
            num_common_leaves = landmark_df_filtered['CleanedLabelStem'].nunique()
            print(f"DEBUG_INFO: Number of unique leaf IDs in landmark data: {len(landmark_df['CleanedLabelStem'].unique())}")
            print(f"DEBUG_INFO: Number of mask files found in directory: {len(mask_files_full_paths)}")
            print(f"DEBUG_INFO: Number of common leaf IDs between landmark data and mask files: {num_common_leaves}")
            print(f"DEBUG_INFO: Number of RGB files found in directory: {len(rgb_files_full_paths)}")

            if num_common_leaves == 0:
                print(f"DEBUG_INFO: No common leaf IDs found. This means none of your landmarked files have a corresponding mask.")
                print("Please ensure your mask files are named consistently with the 'Label' column in 'training_data_results.csv' (e.g., 'NAME_ID.png' for 'NAME_ID_overlay').")
                # Set landmark_df to empty to prevent further errors
                landmark_df = pd.DataFrame() 
            else:
                 landmark_df = landmark_df_filtered # Use the filtered DataFrame for processing

        except Exception as e:
            print(f"Error loading or filtering landmark data from '{landmark_data_file}': {e}. No landmarks will be processed.")
            landmark_df = pd.DataFrame() 
    else:
        print(f"Warning: Landmark data file '{LANDMARK_DATA_FILE}' not found. No landmarks will be processed.")

    leaves_to_process_ids = list(mask_files_full_paths.keys()) # Process all masks found
    if not landmark_df.empty: # If landmark data exists, filter to only those with landmarks
        leaves_to_process_ids = [lid for lid in leaves_to_process_ids if lid in landmark_df['CleanedLabelStem'].unique()]


    total_files = len(leaves_to_process_ids)

    if total_files == 0:
        print(f"No .png mask files found (after filtering by landmark data, if applicable) to process. Exiting.")
        return

    print(f"Found {total_files} .png mask files (to process).")


    for i, leaf_id in enumerate(leaves_to_process_ids):
        mask_file_path = mask_files_full_paths[leaf_id]

        output_image_name = f"{leaf_id}.png"
        mask_path = SHAPE_MASK_DIR / output_image_name
        ect_path = RADIAL_ECT_DIR / output_image_name
        viz_path = COMBINED_VIZ_DIR / output_image_name
        oriented_rgb_path = ORIENTED_RGB_DIR / output_image_name

        print(f"Processing leaf shape {i+1}/{total_files} ({leaf_id})")

        # Initialize variables for current leaf, default to empty/None in case of early errors
        full_leaf_contour_points = None
        raw_blade_pixels = np.array([])
        raw_vein_pixels = np.array([])
        raw_landmark_points = np.array([]) # Initialize as empty array, not None
        G = None
        ect_affine_matrix = None # Initialize this here
        transformed_blade_pixels = np.array([])
        transformed_vein_pixels = np.array([])
        transformed_landmark_points = np.array([])

        try:
            # --- Load Mask and Extract Contours/Pixels ---
            mask_img_cv = cv2.imread(str(mask_file_path), cv2.IMREAD_GRAYSCALE)
            if mask_img_cv is None:
                raise ValueError(f"Could not load mask image: {mask_file_path}")
            if mask_img_cv.ndim == 3: 
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

            # --- Load Landmark Data ---
            current_leaf_has_landmarks = False
            if not landmark_df.empty: 
                leaf_landmarks_df = landmark_df[landmark_df['CleanedLabelStem'] == leaf_id]
                if len(leaf_landmarks_df) == 2:
                    leaf_landmarks_df = leaf_landmarks_df.sort_values(by='index')
                    base_row = leaf_landmarks_df.iloc[0]
                    tip_row = leaf_landmarks_df.iloc[1]
                    raw_landmark_points = np.array([[base_row['X'], base_row['Y']],
                                                    [tip_row['X'], tip_row['Y']]], dtype=np.float64)
                    current_leaf_has_landmarks = True
                elif len(leaf_landmarks_df) > 0:
                    raw_landmark_points = np.array([], dtype=np.float64).reshape(0, 2) # Ensure (0, 2) for invalid count
            else:
                raw_landmark_points = np.array([], dtype=np.float64).reshape(0, 2) # Ensure (0, 2) for no landmark data

            
            # --- ECT Processing & Affine Matrix Derivation ---
            G = EmbeddedGraph()
            G.add_cycle(full_leaf_contour_points)

            original_G_coord_matrix = np.copy(G.coord_matrix)

            G.center_coordinates(center_type="origin")
            G.transform_coordinates() # This is where the internal ECT alignment happens (e.g., to positive Y-axis)
            G.scale_coordinates(BOUND_RADIUS)

            try:
                ect_affine_matrix = find_robust_affine_transformation_matrix(original_G_coord_matrix, G.coord_matrix)
                print(f"DEBUG_ECT_AFFINE: ect_affine_matrix shape: {ect_affine_matrix.shape}") # Debug print
            except ValueError as e:
                raise RuntimeError(f"Could not derive affine transformation matrix for ECT: {e}")
            
            # --- CRITICAL DEBUGGING: Check affine_matrix shape BEFORE applying to pixels/landmarks ---
            print(f"DEBUG_ECT_AFFINE_BEFORE_APPLYING: ect_affine_matrix shape immediately before transformations: {ect_affine_matrix.shape}")

            # Transform blade, veins, and landmarks using the ECT's derived affine matrix
            if raw_blade_pixels.size > 0: # Use .size for empty check consistent with reshape(0,2)
                transformed_blade_pixels = apply_transformation_with_affine_matrix(
                    raw_blade_pixels, ect_affine_matrix
                )
            if raw_vein_pixels.size > 0:
                transformed_vein_pixels = apply_transformation_with_affine_matrix(
                    raw_vein_pixels, ect_affine_matrix
                )
            if raw_landmark_points.size > 0:
                transformed_landmark_points = apply_transformation_with_affine_matrix(
                    raw_landmark_points, ect_affine_matrix
                )

            if np.all(G.coord_matrix == 0):
                raise ValueError("Degenerate full leaf shape (all points at origin after scaling)")
            if G.coord_matrix.shape[0] < 3:
                raise ValueError(f"Processed full leaf shape has too few points ({G.coord_matrix.shape[0]}) to form a valid polygon.")

            ect_result = ect_calculator.calculate(G) # Calculate ECT using the ECT-aligned G.coord_matrix


            # --- Process RGB Image ---
            rgb_image_path = rgb_files_full_paths.get(leaf_id)
            if rgb_image_path and rgb_image_path.exists():
                rgb_img_cv = cv2.imread(str(rgb_image_path))
                if rgb_img_cv is not None:
                    # Define the ECT-to-Output-Pixel matrix (maps ECT coords to display coords)
                    
                    scale_to_pixels = IMAGE_SIZE[0] / (2 * BOUND_RADIUS)
                    offset_x_pixels = IMAGE_SIZE[0] / 2
                    offset_y_pixels = IMAGE_SIZE[1] / 2
                    
                    # This matrix performs the same transformation as ect_coords_to_pixels
                    # Conceptually: (x_ect, y_ect) -> (y_ect, x_ect) then map to Y-down pixel system
                    # pixel_x = (y_ect * scale) + offset_x
                    # pixel_y = (-x_ect * scale) + offset_y (due to Y-down pixel system)

                    ect_space_to_output_pixel_matrix_2x3 = np.array([
                        [0, scale_to_pixels, offset_x_pixels],  # ECT_Y -> pixel_X
                        [-scale_to_pixels, 0, offset_y_pixels]   # -ECT_X -> pixel_Y (due to Y-down pixel system)
                    ], dtype=np.float32)

                    # Ensure ect_affine_matrix is still 3x3 before stacking and multiplying for RGB
                    if ect_affine_matrix.shape != (3,3):
                         raise ValueError(f"ect_affine_matrix changed shape to {ect_affine_matrix.shape} before RGB transformation!")

                    ect_space_to_output_pixel_matrix_3x3 = np.vstack([ect_space_to_output_pixel_matrix_2x3, [0, 0, 1]])

                    # M_final = M_display @ M_ect
                    final_transformation_matrix_3x3 = ect_space_to_output_pixel_matrix_3x3 @ ect_affine_matrix

                    M_2x3_for_warp = final_transformation_matrix_3x3[:2, :] 
                    
                    oriented_rgb_img = cv2.warpAffine(rgb_img_cv, M_2x3_for_warp, IMAGE_SIZE, flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
                    cv2.imwrite(str(oriented_rgb_path), oriented_rgb_img)
                else:
                    oriented_rgb_path = None
            else:
                oriented_rgb_path = None

            # Nested try for image saving (errors in saving shouldn't stop overall ECT processing)
            try:
                save_grayscale_shape_mask(transformed_blade_pixels, transformed_vein_pixels, mask_path)
                save_grayscale_radial_ect(ect_result, ect_path) 

                create_combined_viz_from_images(ect_path, viz_path,
                                                blade_color=(255, 255, 255),
                                                blade_alpha=0.3,
                                                landmark_points_transformed=transformed_landmark_points, 
                                                landmark_color=(255, 255, 0),
                                                vein_points_transformed=transformed_vein_pixels, 
                                                vein_color=(255, 255, 255),
                                                transformed_blade_pixels=transformed_blade_pixels) 

            except Exception as e:
                print(f"  Error saving images for '{leaf_id}': {e}. Marking as invalid.")
                skipped_count += 1
                metadata_records.append({
                    "leaf_id": leaf_id,
                    "raw_file_path": str(mask_file_path),
                    "is_processed_valid": False,
                    "reason_skipped": f"Image saving failed: {e}",
                    "num_raw_points_full_leaf": 0,
                    "num_processed_points_full_leaf": 0,
                    "file_shape_mask": str(mask_path.relative_to(output_base_dir)) if mask_path.exists() else "",
                    "file_radial_ect": str(ect_path.relative_to(output_base_dir)) if ect_path.exists() else "",
                    "file_combined_viz": str(viz_path.relative_to(output_base_dir)) if viz_path.exists() else "",
                    "file_oriented_rgb": "",
                    "has_landmarks": False, "has_veins": False, "has_blade": False,
                    "landmark_base_x": np.nan, "landmark_base_y": np.nan,
                    "landmark_tip_x": np.nan, "landmark_tip_y": np.nan
                })
                continue 
            
            base_x, base_y, tip_x, tip_y = np.nan, np.nan, np.nan, np.nan
            has_landmarks_flag = False
            if transformed_landmark_points is not None and transformed_landmark_points.size > 0 and len(transformed_landmark_points) == 2:
                base_x, base_y = transformed_landmark_points[0]
                tip_x, tip_y = transformed_landmark_points[1]
                has_landmarks_flag = True

            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(mask_file_path),
                "is_processed_valid": True,
                "reason_skipped": "",
                "num_raw_points_full_leaf": full_leaf_contour_points.shape[0],
                "num_processed_points_full_leaf": G.coord_matrix.shape[0], 
                "file_shape_mask": str(mask_path.relative_to(output_base_dir)) if isinstance(mask_path, Path) else "",
                "file_radial_ect": str(ect_path.relative_to(output_base_dir)) if isinstance(ect_path, Path) else "",
                "file_combined_viz": str(viz_path.relative_to(output_base_dir)) if isinstance(viz_path, Path) else "",
                "file_oriented_rgb": str(oriented_rgb_path.relative_to(output_base_dir)) if isinstance(oriented_rgb_path, Path) and oriented_rgb_path.exists() else "",
                "has_landmarks": has_landmarks_flag,
                "has_veins": True if raw_vein_pixels.size > 0 else False,
                "has_blade": True if raw_blade_pixels.size > 0 else False,
                "landmark_base_x": base_x,
                "landmark_base_y": base_y,
                "landmark_tip_x": tip_x,
                "landmark_tip_y": tip_y
            })
            processed_count += 1

        except Exception as e: 
            num_raw_full_leaf_pts = 0
            num_proc_full_leaf_pts = 0
            if full_leaf_contour_points is not None:
                num_raw_full_leaf_pts = full_leaf_contour_points.shape[0]
            if G is not None and G.coord_matrix is not None:
                num_proc_full_leaf_pts = G.coord_matrix.shape[0]

            print(f"  Skipped processing '{leaf_id}' due to error: {e}")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(mask_file_path),
                "is_processed_valid": False,
                "reason_skipped": str(e),
                "num_raw_points_full_leaf": num_raw_full_leaf_pts,
                "num_processed_points_full_leaf": num_proc_full_leaf_pts,
                "file_shape_mask": str(mask_path.relative_to(output_base_dir)) if mask_path.exists() else "",
                "file_radial_ect": str(ect_path.relative_to(output_base_dir)) if ect_path.exists() else "",
                "file_combined_viz": str(viz_path.relative_to(output_base_dir)) if viz_path.exists() else "",
                "file_oriented_rgb": "",
                "has_landmarks": False, "has_veins": False, "has_blade": False,
                "landmark_base_x": np.nan,
                "landmark_base_y": np.nan,
                "landmark_tip_x": tip_x,
                "landmark_tip_y": tip_y
            })
            continue 

    metadata_df = pd.DataFrame(metadata_records)
    print(f"DEBUG_FINAL: Metadata records count before saving: {len(metadata_records)}") 
    metadata_df.to_csv(metadata_file, index=False)

    print(f"\n--- Processing Complete ---")
    print(f"Total files considered: {total_files}")
    print(f"Shapes successfully processed and saved: {processed_count}")
    print(f"Shapes skipped (invalid processing/saving): {skipped_count}")
    print(f"Metadata saved to: {metadata_file}") 
    print(f"Processed images saved in: {output_base_dir}/{{shape_masks, radial_ects, combined_viz, oriented_rgb}}")

if __name__ == "__main__":
    RAW_LEAF_SHAPES_DIR_WITH_MASKS = Path("FINAL_ALIGNED_LEAVES_512X512/")

    if not (RAW_LEAF_SHAPES_DIR_WITH_MASKS / "MASKS").exists():
        print(f"Error: Input directory for masks '{RAW_LEAF_SHAPES_DIR_WITH_MASKS / 'MASKS'}' not found.")
        print("Please ensure your mask PNGs are in this directory.")
        sys.exit(1)
    
    if not (RAW_LEAF_SHAPES_DIR_WITH_MASKS / "RGB_CROPS").exists():
        print(f"Error: Input directory for RGB crops '{RAW_LEAF_SHAPES_DIR_WITH_MASKS / 'RGB_CROPS'}' not found.")
        print("Please ensure your RGB crop PNGs are in this directory, named consistently (e.g., 'NAME_ID.png' or 'NAME_ID_rgb_crop.png').")
        sys.exit(1)


    if not LANDMARK_DATA_FILE.exists():
        print(f"Error: Landmark data file '{LANDMARK_DATA_FILE}' not found.")
        print("Please ensure 'training_data_results.csv' is in the current working directory.")
        sys.exit(1)

    process_raw_leaf_shapes(RAW_LEAF_SHAPES_DIR_WITH_MASKS, PROCESSED_DATA_OUTPUT_DIR, LANDMARK_DATA_FILE)