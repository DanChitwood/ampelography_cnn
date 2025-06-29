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
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS) # This also adapts to NUM_ECT_DIRECTIONS
IMAGE_SIZE = (256, 256) # Output size for all images (masks, ECT, RGB)

PROCESSED_DATA_OUTPUT_DIR = Path("processed_leaf_data/")
SHAPE_MASK_DIR = PROCESSED_DATA_OUTPUT_DIR / "shape_masks"
RADIAL_ECT_DIR = PROCESSED_DATA_OUTPUT_DIR / "radial_ects"
COMBINED_VIZ_DIR = PROCESSED_DATA_OUTPUT_DIR / "combined_viz"
ORIENTED_RGB_DIR = PROCESSED_DATA_OUTPUT_DIR / "oriented_rgb" # NEW DIRECTORY FOR ORIENTED RGB
METADATA_FILE = PROCESSED_DATA_OUTPUT_DIR / "metadata.csv"
LANDMARK_DATA_FILE = Path("training_data_results.csv") # This file should be in the same directory as your script

BACKGROUND_PIXEL = 0
BLADE_PIXEL = 1
VEIN_PIXEL = 2

# Define grayscale values for the output mask file
MASK_BACKGROUND_GRAY = 0    # Black
MASK_BLADE_GRAY = 128       # Gray
MASK_VEIN_GRAY = 255        # White


# --- Helper Functions for Transformations ---

def apply_transformation_with_affine_matrix(points: np.ndarray, affine_matrix: np.ndarray):
    if len(points) == 0:
        return points
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_homogeneous = points_homogeneous @ affine_matrix.T
    transformed_points = transformed_homogeneous[:, :2]
    return transformed_points

def find_robust_affine_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray):
    """
    Finds a robust affine transformation matrix between source and destination points.
    Uses three representative points if available, otherwise falls back to first three.
    Ensures non-collinear points for cv2.getAffineTransform.
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        raise ValueError("Need at least 3 points to compute affine transformation.")

    x_min_idx = np.argmin(src_points[:, 0])
    x_max_idx = np.argmax(src_points[:, 0])
    y_min_idx = np.argmin(src_points[:, 1])
    y_max_idx = np.argmax(src_points[:, 1])

    candidate_indices = [x_min_idx, x_max_idx, y_min_idx, y_max_idx]
    unique_indices = list(dict.fromkeys(candidate_indices))
    
    if len(unique_indices) < 3:
        for i in range(len(src_points)):
            if i not in unique_indices:
                unique_indices.append(i)
            if len(unique_indices) == 3:
                break
    
    if len(unique_indices) < 3:
        raise ValueError("Not enough distinct points to find 3 non-collinear points for affine transformation.")

    src_pts_for_transform = np.float32([src_points[i] for i in unique_indices[:3]])
    dst_pts_for_transform = np.float32([dst_points[i] for i in unique_indices[:3]])
    
    area = 0.5 * np.abs(src_pts_for_transform[0,0]*(src_pts_for_transform[1,1]-src_pts_for_transform[2,1]) +
                        src_pts_for_transform[1,0]*(src_pts_for_transform[2,1]-src_pts_for_transform[0,1]) +
                        src_pts_for_transform[2,0]*(src_pts_for_transform[0,1]-src_pts_for_transform[1,1]))
    
    if area < 1e-6:
        raise ValueError("Could not find 3 non-collinear points for affine transformation. Shape is likely degenerate or a line.")

    M_2x3 = cv2.getAffineTransform(src_pts_for_transform, dst_pts_for_transform)
    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    
    return affine_matrix_3x3

# --- Visualization Functions ---

def save_grayscale_shape_mask(transformed_blade_pixels: np.ndarray, transformed_vein_pixels: np.ndarray, save_path: Path):
    """
    Saves a grayscale image representing the transformed leaf mask.
    Pixels are filled based on transformed_blade_pixels and transformed_vein_pixels.
    """
    img = Image.new("L", IMAGE_SIZE, MASK_BACKGROUND_GRAY) # Initialize with background color

    def ect_coords_to_pixels(coords_ect: np.ndarray):
        if len(coords_ect) == 0:
            return np.array([])
        
        transformed_coords_for_plot = np.array([coords_ect[:, 1], -coords_ect[:, 0]]).T

        scale = IMAGE_SIZE[0] / (2 * BOUND_RADIUS)
        offset_x = IMAGE_SIZE[0] / 2
        offset_y = IMAGE_SIZE[1] / 2 

        pixel_x = (transformed_coords_for_plot[:, 0] * scale + offset_x).astype(int)
        pixel_y = (transformed_coords_for_plot[:, 1] * scale + offset_y).astype(int)
        
        return np.column_stack((pixel_x, pixel_y))

    # Fill blade pixels
    if transformed_blade_pixels is not None and len(transformed_blade_pixels) > 0:
        blade_pixel_coords = ect_coords_to_pixels(transformed_blade_pixels)
        for x, y in blade_pixel_coords:
            if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]:
                img.putpixel((x, y), MASK_BLADE_GRAY)

    # Fill vein pixels (overwriting blade pixels if they overlap, as veins are foreground)
    if transformed_vein_pixels is not None and len(transformed_vein_pixels) > 0:
        vein_pixel_coords = ect_coords_to_pixels(transformed_vein_pixels)
        for x, y in vein_pixel_coords:
            if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]:
                img.putpixel((x, y), MASK_VEIN_GRAY)
    
    img.save(save_path)

def save_grayscale_radial_ect(ect_result, save_path: Path):
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap="inferno")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # Clockwise
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
                                     transformed_blade_pixels: np.ndarray = None,
                                     main_contour_points: np.ndarray = None):
    try:
        ect_img = Image.open(ect_image_path).convert("RGBA")
        img_width, img_height = ect_img.size

        def ect_to_pixel(coords: np.ndarray, size: tuple, radius: float):
            if len(coords) == 0:
                return np.array([])
            
            transformed_coords_for_plot = np.array([coords[:, 1], -coords[:, 0]]).T

            scale = size[0] / (2 * radius)
            offset_x = size[0] / 2
            offset_y = size[1] / 2 

            pixel_x = (transformed_coords_for_plot[:, 0] * scale + offset_x).astype(int)
            pixel_y = (transformed_coords_for_plot[:, 1] * scale + offset_y).astype(int)
            
            return np.column_stack((pixel_x, pixel_y))

        composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw_composite = ImageDraw.Draw(composite_overlay)

        if transformed_blade_pixels is not None and len(transformed_blade_pixels) > 0:
            blade_pixel_coords = ect_to_pixel(transformed_blade_pixels, (img_width, img_height), BOUND_RADIUS)
            for x, y in blade_pixel_coords:
                draw_composite.rectangle([x, y, x, y], fill=(blade_color[0], blade_color[1], blade_color[2], int(255 * blade_alpha)))
        elif main_contour_points is not None and len(main_contour_points) > 0:
            contour_pixel_coords = ect_to_pixel(main_contour_points, (img_width, img_height), BOUND_RADIUS).tolist()
            contour_tuples = [tuple(p) for p in contour_pixel_coords]
            if len(contour_tuples) > 2:
                draw_composite.polygon(contour_tuples, fill=(blade_color[0], blade_color[1], blade_color[2], int(255 * blade_alpha)))

        if vein_points_transformed is not None and len(vein_points_transformed) > 0:
            vein_pixel_coords = ect_to_pixel(vein_points_transformed, (img_width, img_height), BOUND_RADIUS)
            vein_fill_color = (vein_color[0], vein_color[1], vein_color[2], int(255 * blade_alpha))
            for x, y in vein_pixel_coords:
                draw_composite.ellipse([x - vein_size//2, y - vein_size//2,
                                        x + vein_size//2, y + vein_size//2],
                                        fill=vein_fill_color)

        if landmark_points_transformed is not None and len(landmark_points_transformed) > 0:
            landmark_pixel_coords = ect_to_pixel(landmark_points_transformed, (img_width, img_height), BOUND_RADIUS)
            for x, y in landmark_pixel_coords:
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
    oriented_rgb_dir = output_base_dir / "oriented_rgb" # NEW
    metadata_file = output_base_dir / "metadata.csv"

    shape_mask_dir.mkdir(parents=True, exist_ok=True)
    radial_ect_dir.mkdir(parents=True, exist_ok=True)
    combined_viz_dir.mkdir(parents=True, exist_ok=True)
    oriented_rgb_dir.mkdir(parents=True, exist_ok=True) # NEW
    print(f"Created output directories: {shape_mask_dir}, {radial_ect_dir}, {combined_viz_dir}, {oriented_rgb_dir}") # NEW

    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)

    metadata_records = []
    processed_count = 0
    skipped_count = 0

    landmark_df = pd.DataFrame()
    mask_files_full_paths = {} # Dictionary to map leaf_id to full mask file path for efficient lookup
    rgb_files_full_paths = {} # NEW: Dictionary to map leaf_id to full RGB file path

    # First, load all mask file paths for quick lookup
    for f in (raw_input_dir / "MASKS").glob("*.png"):
        mask_files_full_paths[f.stem.replace('_mask', '')] = f

    # NEW: Load all RGB crop file paths
    for f in (raw_input_dir / "RGB_CROPS").glob("*.png"): # Assuming RGB_CROPS are also PNGs
        rgb_files_full_paths[f.stem.replace('_rgb_crop', '')] = f # Assuming naming convention '_rgb_crop'

    if landmark_data_file.exists():
        try:
            landmark_df = pd.read_csv(landmark_data_file)
            
            # Create a 'CleanedLabelStem' column that matches the mask file stems (e.g., 'ALGERIA_Ahmeur Bou Ahmeur_2')
            landmark_df['CleanedLabelStem'] = landmark_df['Label'].apply(
                lambda x: Path(x).stem.replace('_overlay', '')
            )
            
            # Filter landmark_df to only include leaves for which we have actual mask files
            original_num_landmarks = len(landmark_df['CleanedLabelStem'].unique())
            landmark_df_filtered = landmark_df[landmark_df['CleanedLabelStem'].isin(mask_files_full_paths.keys())]
            
            num_common_leaves = landmark_df_filtered['CleanedLabelStem'].nunique()
            print(f"DEBUG_INFO: Number of unique leaf IDs in landmark data: {original_num_landmarks}")
            print(f"DEBUG_INFO: Number of mask files found in directory: {len(mask_files_full_paths)}")
            print(f"DEBUG_INFO: Number of common leaf IDs between landmark data and mask files: {num_common_leaves}")
            print(f"DEBUG_INFO: Number of RGB files found in directory: {len(rgb_files_full_paths)}") # NEW

            if num_common_leaves > 0:
                print(f"DEBUG_INFO: Sample of common leaf IDs (first 5): {list(landmark_df_filtered['CleanedLabelStem'].unique())[:5]}")
            else:
                print(f"DEBUG_INFO: No common leaf IDs found. This means none of your landmarked files have a corresponding mask.")
                print("Please ensure your mask files are named consistently with the 'Label' column in 'training_data_results.csv' (e.g., 'NAME_ID_mask.png' for 'NAME_ID_overlay').")

            landmark_df = landmark_df_filtered # Use the filtered DataFrame for processing

        except Exception as e:
            print(f"Error loading or filtering landmark data from '{landmark_data_file}': {e}. No landmarks will be processed.")
            # Set landmark_df to empty to prevent further errors if loading failed
            landmark_df = pd.DataFrame() 
    else:
        print(f"Warning: Landmark data file '{LANDMARK_DATA_FILE}' not found. No landmarks will be processed.")

    # Determine which specific mask files to process based on landmark data
    leaves_to_process_ids = landmark_df['CleanedLabelStem'].unique()
    
    mask_files_to_process = []
    for leaf_id in leaves_to_process_ids:
        if leaf_id in mask_files_full_paths:
            mask_files_to_process.append(mask_files_full_paths[leaf_id])
        else:
            # This case should ideally not happen if landmark_df was filtered correctly
            print(f"Internal Warning: Mask file for landmarked ID '{leaf_id}' not found. Skipping.")

    total_files = len(mask_files_to_process)

    if total_files == 0:
        print(f"No .png mask files found (after filtering by landmark data) to process. Exiting.")
        return

    print(f"Found {total_files} .png mask files (filtered by landmark data) to process.")


    for i, mask_file_path in enumerate(mask_files_to_process):
        leaf_id = mask_file_path.stem.replace('_mask', '') # Also strip _mask here for consistency

        # Define paths here, ALWAYS, before any try/except that might skip
        output_image_name = f"{leaf_id}.png"
        mask_path = shape_mask_dir / output_image_name
        ect_path = radial_ect_dir / output_image_name
        viz_path = combined_viz_dir / output_image_name
        oriented_rgb_path = oriented_rgb_dir / output_image_name # NEW RGB PATH

        # Limit debug prints to the first few leaves, as it will be very verbose otherwise
        if i < 10 or (i + 1) % 100 == 0 or (i + 1) == total_files: # Show first 10, then every 100, and last one
            print(f"Processing leaf shape {i+1}/{total_files} ({leaf_id})")
            debug_this_leaf = True
        else:
            debug_this_leaf = False

        full_leaf_contour_points = None
        raw_blade_pixels = np.array([])
        raw_vein_pixels = np.array([])
        raw_landmark_points = None
        G = None
        ect_affine_matrix = None

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

            if full_leaf_contour_points.ndim == 1:
                if full_leaf_contour_points.shape[0] == 2:
                    full_leaf_contour_points = full_leaf_contour_points.reshape(1, 2)
                else:
                    raise ValueError(f"Unexpected full_leaf_contour_points dimension after squeeze: {full_leaf_contour_points.shape}")
            if full_leaf_contour_points.shape[0] < 3:
                raise ValueError(f"Full leaf shape has too few points ({full_leaf_contour_points.shape[0]}) to form a valid polygon.")

            blade_y, blade_x = np.where(mask_img_cv == BLADE_PIXEL)
            if len(blade_x) > 0:
                raw_blade_pixels = np.column_stack((blade_x, blade_y)).astype(np.float64)

            vein_y, vein_x = np.where(mask_img_cv == VEIN_PIXEL)
            if len(vein_x) > 0:
                raw_vein_pixels = np.column_stack((vein_x, vein_y)).astype(np.float64)

            # --- Load Landmark Data ---
            if not landmark_df.empty: 
                leaf_landmarks_df = landmark_df[landmark_df['CleanedLabelStem'] == leaf_id]
                if len(leaf_landmarks_df) == 2:
                    leaf_landmarks_df = leaf_landmarks_df.sort_values(by='index')
                    base_row = leaf_landmarks_df.iloc[0]
                    tip_row = leaf_landmarks_df.iloc[1]
                    raw_landmark_points = np.array([[base_row['X'], base_row['Y']],
                                                    [tip_row['X'], tip_row['Y']]], dtype=np.float64)
                elif len(leaf_landmarks_df) > 0:
                    if debug_this_leaf: print(f"DEBUG_LOOP: Warning: Found {len(leaf_landmarks_df)} landmark entries for {leaf_id}. Expected 2. Skipping landmarks for this leaf.")
                    raw_landmark_points = None
            
            # --- ECT Processing & Affine Matrix Derivation ---
            G = EmbeddedGraph()
            G.add_cycle(full_leaf_contour_points)

            original_G_coord_matrix = np.copy(G.coord_matrix)

            G.center_coordinates(center_type="origin")
            G.transform_coordinates()
            G.scale_coordinates(BOUND_RADIUS)

            if hasattr(G, '_affine_matrix') and G._affine_matrix is not None:
                ect_affine_matrix = G._affine_matrix
            else:
                try:
                    ect_affine_matrix = find_robust_affine_transformation_matrix(original_G_coord_matrix, G.coord_matrix)
                except ValueError as e:
                    raise RuntimeError(f"Could not derive affine transformation matrix for ECT: {e}")

            if np.all(G.coord_matrix == 0):
                raise ValueError("Degenerate full leaf shape (all points at origin after scaling)")
            if G.coord_matrix.shape[0] < 3:
                raise ValueError(f"Processed full leaf shape has too few points ({G.coord_matrix.shape[0]}) to form a valid polygon.")

            ect_result = ect_calculator.calculate(G)

            # --- Apply the chosen affine matrix to all other components ---
            if raw_blade_pixels is not None and len(raw_blade_pixels) > 0:
                transformed_blade_pixels = apply_transformation_with_affine_matrix(
                    raw_blade_pixels, ect_affine_matrix
                )

            if raw_vein_pixels is not None and len(raw_vein_pixels) > 0:
                transformed_vein_pixels = apply_transformation_with_affine_matrix(
                    raw_vein_pixels, ect_affine_matrix
                )

            if raw_landmark_points is not None and len(raw_landmark_points) > 0:
                transformed_landmark_points = apply_transformation_with_affine_matrix(
                    raw_landmark_points, ect_affine_matrix
                )
            
            # --- NEW: Process RGB Image ---
            rgb_image_path = rgb_files_full_paths.get(leaf_id)
            if rgb_image_path and rgb_image_path.exists():
                rgb_img_cv = cv2.imread(str(rgb_image_path))
                if rgb_img_cv is not None:
                    # Apply the same affine transformation to the RGB image
                    # M_2x3 is the 2x3 part of the 3x3 affine_matrix
                    M_2x3 = ect_affine_matrix[:2, :] 
                    
                    # Ensure the output size matches IMAGE_SIZE
                    oriented_rgb_img = cv2.warpAffine(rgb_img_cv, M_2x3, IMAGE_SIZE, flags=cv2.INTER_LINEAR, borderValue=(0,0,0)) # Fill black background
                    cv2.imwrite(str(oriented_rgb_path), oriented_rgb_img)
                else:
                    if debug_this_leaf: print(f"DEBUG_LOOP: Warning: Could not load RGB image for {leaf_id} at {rgb_image_path}")
                    oriented_rgb_path = None # Indicate failure to load/process RGB
            else:
                if debug_this_leaf: print(f"DEBUG_LOOP: Warning: RGB image not found for {leaf_id} at expected path: {rgb_image_path}")
                oriented_rgb_path = None # Indicate missing RGB

            # Nested try for image saving, using already defined paths
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
                                                transformed_blade_pixels=transformed_blade_pixels,
                                                main_contour_points=G.coord_matrix)

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
                    "file_oriented_rgb": "", # NEW
                    "has_landmarks": False, "has_veins": False, "has_blade": False,
                    "landmark_base_x": np.nan, "landmark_base_y": np.nan,
                    "landmark_tip_x": np.nan, "landmark_tip_y": np.nan
                })
                continue # Skip to next leaf if image saving fails


        except Exception as e: # This is the outer processing error handler
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
                "file_oriented_rgb": "", # NEW
                "has_landmarks": False, "has_veins": False, "has_blade": False,
                "landmark_base_x": np.nan, "landmark_base_y": np.nan,
                "landmark_tip_x": np.nan, "landmark_tip_y": np.nan
            })
            continue # Skip to next leaf if processing fails

        # Determine landmark coordinates to save
        base_x, base_y, tip_x, tip_y = np.nan, np.nan, np.nan, np.nan
        has_landmarks_flag = False
        if transformed_landmark_points is not None and len(transformed_landmark_points) == 2:
            base_x, base_y = transformed_landmark_points[0]
            tip_x, tip_y = transformed_landmark_points[1]
            has_landmarks_flag = True

        # This block is only reached if NO exceptions occurred in EITHER try block
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
            "file_oriented_rgb": str(oriented_rgb_path.relative_to(output_base_dir)) if isinstance(oriented_rgb_path, Path) and oriented_rgb_path.exists() else "", # NEW
            "has_landmarks": has_landmarks_flag,
            "has_veins": True if raw_vein_pixels is not None and len(raw_vein_pixels) > 0 else False,
            "has_blade": True if raw_blade_pixels is not None and len(raw_blade_pixels) > 0 else False,
            "landmark_base_x": base_x,
            "landmark_base_y": base_y,
            "landmark_tip_x": tip_x,
            "landmark_tip_y": tip_y
        })
        processed_count += 1

    metadata_df = pd.DataFrame(metadata_records)
    print(f"DEBUG_FINAL: Metadata records count before saving: {len(metadata_records)}") 
    metadata_df.to_csv(metadata_file, index=False)

    print(f"\n--- Processing Complete ---")
    print(f"Total files considered: {total_files}")
    print(f"Shapes successfully processed and saved: {processed_count}")
    print(f"Shapes skipped (invalid processing/saving): {skipped_count}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Processed images saved in: {output_base_dir}/{{shape_masks, radial_ects, combined_viz, oriented_rgb}}") # NEW

if __name__ == "__main__":
    RAW_LEAF_SHAPES_DIR_WITH_MASKS = Path("FINAL_ALIGNED_LEAVES_512X512/")

    if not (RAW_LEAF_SHAPES_DIR_WITH_MASKS / "MASKS").exists():
        print(f"Error: Input directory for masks '{RAW_LEAF_SHAPES_DIR_WITH_MASKS / 'MASKS'}' not found.")
        print("Please ensure your mask PNGs are in this directory.")
        sys.exit(1)
    
    # NEW: Check for RGB_CROPS directory
    if not (RAW_LEAF_SHAPES_DIR_WITH_MASKS / "RGB_CROPS").exists():
        print(f"Error: Input directory for RGB crops '{RAW_LEAF_SHAPES_DIR_WITH_MASKS / 'RGB_CROPS'}' not found.")
        print("Please ensure your RGB crop PNGs are in this directory, named consistently (e.g., 'NAME_ID.png' or 'NAME_ID_rgb_crop.png').")
        sys.exit(1)


    if not LANDMARK_DATA_FILE.exists():
        print(f"Error: Landmark data file '{LANDMARK_DATA_FILE}' not found.")
        print("Please ensure 'training_data_results.csv' is in the current working directory.")
        sys.exit(1)

    process_raw_leaf_shapes(RAW_LEAF_SHAPES_DIR_WITH_MASKS, PROCESSED_DATA_OUTPUT_DIR, LANDMARK_DATA_FILE)