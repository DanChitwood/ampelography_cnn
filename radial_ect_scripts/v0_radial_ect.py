import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image # For image loading and manipulation
from PIL import ImageDraw # For drawing landmarks
import sys
import shutil
import cv2 # For contour extraction

# Ensure the ect library is installed and accessible
try:
    from ect import ECT, EmbeddedGraph
except ImportError:
    print("Error: The 'ect' library is not found. Please ensure it's installed and accessible.")
    print("Add its directory to PYTHONPATH or install it correctly (e.g., pip install ect-morphology).")
    sys.exit(1)

# --- Configuration Parameters (Consolidated from generate_superformula_data.py and process_leaf_shapes.py) ---
BOUND_RADIUS = 1  # Max radius for ECT calculation (shapes scaled to fit this)
NUM_ECT_DIRECTIONS = 90
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS)
IMAGE_SIZE = (256, 256) # Desired pixel size for output images (width, height)

# Input/Output Directories for processing real leaf shapes
# RAW_LEAF_SHAPES_DIR = Path("raw_leaf_shapes/") # Original, now replaced by the one below
PROCESSED_DATA_OUTPUT_DIR = Path("processed_leaf_data/") # Output root directory for all processed data

# Subdirectories for different image types (relative to PROCESSED_DATA_OUTPUT_DIR)
SHAPE_MASK_DIR = PROCESSED_DATA_OUTPUT_DIR / "shape_masks"
RADIAL_ECT_DIR = PROCESSED_DATA_OUTPUT_DIR / "radial_ects"
COMBINED_VIZ_DIR = PROCESSED_DATA_OUTPUT_DIR / "combined_viz"
METADATA_FILE = PROCESSED_DATA_OUTPUT_DIR / "metadata.csv"
LANDMARK_DATA_FILE = Path("training_data_results.csv") # Path to your landmark CSV

# --- Helper Functions for Transformations (New/Modified) ---

def find_transformation_parameters(original_points: np.ndarray, transformed_points: np.ndarray):
    """
    Estimates the centroid, scale, and rotation matrix that transforms
    original_points to transformed_points. This uses an affine transformation
    estimation (specifically partial affine for rotation, scale, translation without shear).

    Args:
        original_points (np.ndarray): N x 2 array of original coordinates.
        transformed_points (np.ndarray): N x 2 array of transformed coordinates.

    Returns:
        tuple: (original_centroid, scale_factor, rotation_matrix)
    """
    # Ensure points are float32 for OpenCV functions
    original_points_f32 = original_points.astype(np.float32)
    transformed_points_f32 = transformed_points.astype(np.float32)

    # Need at least 3 points for estimateAffinePartial2D
    if len(original_points_f32) < 3 or len(transformed_points_f32) < 3:
        # Fallback for too few points: assume no transformation
        original_centroid = np.mean(original_points_f32, axis=0) if len(original_points_f32) > 0 else np.array([0., 0.])
        return original_centroid, 1.0, np.eye(2)

    try:
        # M is a 2x3 affine transformation matrix: [[R00, R01, Tx], [R10, R11, Ty]]
        # where R is the 2x2 rotation/scale part, and T is the translation part.
        M, _ = cv2.estimateAffinePartial2D(original_points_f32, transformed_points_f32)

        if M is None: # estimateAffinePartial2D can return None if it fails
              raise ValueError("cv2.estimateAffinePartial2D returned None.")

        # Extract rotation matrix (2x2 part) and scale factor
        rotation_matrix = M[:2, :2]
        # The scale factor is the length of the column vectors of the rotation part
        scale_factor = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)

        # Normalize rotation_matrix to be purely rotation (if scale_factor != 0)
        if scale_factor != 0:
              rotation_matrix = rotation_matrix / scale_factor
        else: # Handle degenerate case where scale_factor is zero
              rotation_matrix = np.eye(2)
              scale_factor = 1.0 # Default to no scaling if original extent was zero

        # The translation part M[:, 2] accounts for how the transformed points
        # are translated relative to the original.
        # We need the original centroid to use as a reference point for applying the transformation.
        original_centroid = np.mean(original_points_f32, axis=0)

        return original_centroid, scale_factor, rotation_matrix
    except cv2.error as e:
        print(f"Warning: Could not estimate affine transformation for points ({len(original_points_f32)} points). {e}. Using approximate scale/identity rotation.")
        # Fallback if estimateAffinePartial2D fails (e.g., degenerate points, singular matrix)
        original_centroid = np.mean(original_points_f32, axis=0) if len(original_points_f32) > 0 else np.array([0., 0.])
        # Calculate scale_factor based on extent if affine fails
        if len(original_points_f32) > 1:
            max_orig_extent = np.max(np.abs(original_points_f32 - original_centroid))
            max_trans_extent = np.max(np.abs(transformed_points_f32 - np.mean(transformed_points_f32, axis=0)))
            scale_factor_fallback = max_trans_extent / max_orig_extent if max_orig_extent != 0 else 1.0
        else:
            scale_factor_fallback = 1.0
        return original_centroid, scale_factor_fallback, np.eye(2)
    except ValueError as e: # Catch the case where M is None
        print(f"Warning: Failed to estimate affine transformation. {e}. Using approximate scale/identity rotation.")
        original_centroid = np.mean(original_points_f32, axis=0) if len(original_points_f32) > 0 else np.array([0., 0.])
        if len(original_points_f32) > 1:
            max_orig_extent = np.max(np.abs(original_points_f32 - original_centroid))
            max_trans_extent = np.max(np.abs(transformed_points_f32 - np.mean(transformed_points_f32, axis=0)))
            scale_factor_fallback = max_trans_extent / max_orig_extent if max_orig_extent != 0 else 1.0
        else:
            scale_factor_fallback = 1.0
        return original_centroid, scale_factor_fallback, np.eye(2)


def apply_transformation_to_points(points: np.ndarray, original_centroid: np.ndarray,
                                   scale_factor: float, rotation_matrix: np.ndarray):
    """
    Splies the given transformation parameters (centroid, scale, rotation) to a set of points.
    Note: This specifically applies the transformation to align with the *centered and scaled*
    `G.coord_matrix` space, which is then at the origin.

    Args:
        points (np.ndarray): N x 2 array of points to transform.
        original_centroid (np.ndarray): Centroid of the original reference points.
        scale_factor (float): Scaling factor.
        rotation_matrix (np.ndarray): 2x2 rotation matrix.

    Returns:
        np.ndarray: Transformed points.
    """
    if len(points) == 0:
        return points

    # Centering relative to the original centroid
    centered_points = points - original_centroid

    # Rotation and Scaling
    # (points - original_centroid) @ rotation_matrix.T * scale_factor
    # This aligns the points, but then `G.coord_matrix` is usually centered at (0,0)
    # The affine transform implicitly moves points to the target coordinate system's origin.
    # So we apply rotation and scaling directly after centering.
    transformed_points = (centered_points @ rotation_matrix.T) * scale_factor

    return transformed_points


# --- Visualization Functions ---

def save_grayscale_shape_mask(processed_points: np.ndarray, save_path: Path):
    """
    Saves a grayscale shape mask using Matplotlib.
    The shape is drawn as white on a black background.
    The transformation (flip + rotate) is applied here.
    """
    fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Apply Left/Right flip AND 90-degree clockwise rotation
    # (x_orig, y_orig) -> (-x_orig, y_orig) -> (y_orig, -(-x_orig)) -> (y_orig, x_orig)
    # This specific plotting transformation is applied here, *after* `processed_points`
    # are already in the `ect` library's canonical frame (G.coord_matrix).
    # This means `processed_points` are already centered, oriented, and scaled by ECT logic.
    # The plotting transformation (y_orig, x_orig) is to match the visualization style.
    transformed_x_for_plot = processed_points[:, 1] # This is y_original from the ECT-transformed coordinates
    transformed_y_for_plot = processed_points[:, 0] # This is x_original from the ECT-transformed coordinates

    ax.fill(transformed_x_for_plot, transformed_y_for_plot, color='white')

    ax.set_xlim([-BOUND_RADIUS, BOUND_RADIUS])
    ax.set_ylim([-BOUND_RADIUS, BOUND_RADIUS])

    ax.set_aspect('equal', adjustable='box')

    ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)

def save_grayscale_radial_ect(ect_result, save_path: Path):
    """Saves a grayscale radial ECT image using Matplotlib."""
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap="gray")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # Clockwise
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, mask_image_path: Path, save_path: Path,
                                    mask_color=(255, 0, 255), mask_alpha=0.8,
                                    landmark_points_transformed: np.ndarray = None, # New argument
                                    landmark_color=(255, 255, 0), landmark_size=10): # New arguments
    """
    Combines a grayscale ECT image and a grayscale shape mask into a single RGB visualization.
    The mask is overlaid in a specified color with transparency.
    Optionally plots transformed landmark points.
    """
    try:
        ect_img = Image.open(ect_image_path).convert("RGB")
        mask_img = Image.open(mask_image_path).convert("L") # Ensure mask is grayscale (Luminance)

        ect_np = np.array(ect_img)
        mask_np = np.array(mask_img) # This is 0 (black) or 255 (white)

        overlay_color_np = np.array(mask_color, dtype=np.uint8)
        colored_overlay = np.zeros_like(ect_np)
        colored_overlay[mask_np == 255] = overlay_color_np

        alpha_val = int(mask_alpha * 255)
        blend_alpha = np.zeros_like(mask_np)
        blend_alpha[mask_np == 255] = alpha_val
        blend_alpha_img = Image.fromarray(blend_alpha, mode='L')

        # Combine ECT and mask first
        combined_img_pil = Image.composite(Image.fromarray(colored_overlay), ect_img, blend_alpha_img)

        # Create a blank image to draw landmarks on if they exist
        landmark_overlay_pil = Image.new("RGBA", ect_img.size, (0, 0, 0, 0))
        if landmark_points_transformed is not None and len(landmark_points_transformed) > 0:
            # We need to map the transformed points (in [-BOUND_RADIUS, BOUND_RADIUS] space)
            # to pixel coordinates (0 to IMAGE_SIZE).
            # The plotting for ECT in save_grayscale_radial_ect is polar, but the final
            # saved image is a Cartesian pixel grid. The key is to match the coordinate
            # system used by save_grayscale_shape_mask for the mask overlay.
            # save_grayscale_shape_mask uses:
            # transformed_x = processed_points[:, 1]
            # transformed_y = processed_points[:, 0]
            # Followed by xlim/ylim for [-BOUND_RADIUS, BOUND_RADIUS]
            
            landmark_points_for_plot_viz = np.array([landmark_points_transformed[:,1], landmark_points_transformed[:,0]]).T

            # Normalize to [0, 1] range based on BOUND_RADIUS and then scale to image size
            # x_pixel = (x_plot + BOUND_RADIUS) / (2 * BOUND_RADIUS) * IMAGE_WIDTH
            # y_pixel = (y_plot + BOUND_RADIUS) / (2 * BOUND_RADIUS) * IMAGE_HEIGHT
            
            pixel_x = ((landmark_points_for_plot_viz[:, 0] + BOUND_RADIUS) / (2 * BOUND_RADIUS) * ect_img.size[0]).astype(int)
            pixel_y = ((landmark_points_for_plot_viz[:, 1] + BOUND_RADIUS) / (2 * BOUND_RADIUS) * ect_img.size[1]).astype(int)

            draw = ImageDraw.Draw(landmark_overlay_pil)
            for x, y in zip(pixel_x, pixel_y):
                # Draw a circle for each landmark
                # (x0, y0, x1, y1) bounding box for ellipse
                draw.ellipse([x - landmark_size//2, y - landmark_size//2,
                              x + landmark_size//2, y + landmark_size//2],
                              fill=landmark_color)

        # Overlay landmarks onto the combined image
        final_combined_img = Image.alpha_composite(combined_img_pil.convert("RGBA"), landmark_overlay_pil).convert("RGB")

        final_combined_img.save(save_path)

    except FileNotFoundError:
        print(f"Error: One or both image files not found: {ect_image_path}, {mask_image_path}")
    except Exception as e:
        print(f"Error combining images {ect_image_path} and {mask_image_path}: {e}")

# --- Main Data Processing Logic ---

def process_raw_leaf_shapes(raw_input_dir: Path, output_base_dir: Path, landmark_data_file: Path, clear_existing_data: bool = True):
    """
    Processes raw leaf shape .npy files (or masks), calculates ECTs, saves images/metadata,
    and includes transformed landmark points.
    """
    print(f"Starting processing of raw leaf shapes from: {raw_input_dir}")
    print(f"Output will be saved to: {output_base_dir}")

    # Setup output directories
    if clear_existing_data and output_base_dir.exists():
        print(f"Clearing existing output directory: {output_base_dir}")
        shutil.rmtree(output_base_dir)

    shape_mask_dir = output_base_dir / "shape_masks"
    radial_ect_dir = output_base_dir / "radial_ects"
    combined_viz_dir = output_base_dir / "combined_viz"
    metadata_file = output_base_dir / "metadata.csv"

    shape_mask_dir.mkdir(parents=True, exist_ok=True)
    radial_ect_dir.mkdir(parents=True, exist_ok=True)
    combined_viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories: {shape_mask_dir}, {radial_ect_dir}, {combined_viz_dir}")

    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)

    metadata_records = []
    processed_count = 0
    skipped_count = 0

    # Load landmark data
    landmark_df = pd.DataFrame() # Initialize as empty
    if landmark_data_file.exists():
        try:
            landmark_df = pd.read_csv(landmark_data_file)
            print(f"Loaded landmark data from: {landmark_data_file}")
        except Exception as e:
            print(f"Error loading landmark data from '{landmark_data_file}': {e}. No landmarks will be processed.")
    else:
        print(f"Warning: Landmark data file '{landmark_data_file}' not found. No landmarks will be processed.")

    # Identify image files to process (assuming MASKS folder contains PNGs now)
    mask_files = list((raw_input_dir / "MASKS").glob("*.png")) # Assuming masks are in a 'MASKS' subdirectory
    total_files = len(mask_files)

    if total_files == 0:
        print(f"No .png mask files found in {raw_input_dir / 'MASKS'}. Exiting.")
        return

    print(f"Found {total_files} .png mask files to process.")

    for i, mask_file_path in enumerate(mask_files):
        leaf_id = mask_file_path.stem # Filename without extension

        if (i + 1) % 100 == 0 or (i + 1) == total_files or (i + 1) == 1:
            print(f"Processing leaf shape {i+1}/{total_files} ({leaf_id})")

        raw_shape_points = None
        raw_landmark_points = None # Will store [base_x, base_y], [tip_x, tip_y]
        G = None
        transformed_landmark_points = np.array([]) # Initialize as empty array

        try:
            # --- 1. Extract raw_shape_points from mask image ---
            mask_img_cv = cv2.imread(str(mask_file_path), cv2.IMREAD_GRAYSCALE)
            if mask_img_cv is None:
                raise ValueError(f"Could not load mask image: {mask_file_path}")
            if mask_img_cv.ndim == 3: # Ensure it's grayscale if it loaded as BGR/RGB
                mask_img_cv = cv2.cvtColor(mask_img_cv, cv2.COLOR_BGR2GRAY)

            # Find contours: RETR_EXTERNAL for outer contours, CHAIN_APPROX_NONE for all points
            # Adding an explicit check for empty contours, as findContours can return empty list
            contours, _ = cv2.findContours(mask_img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                raise ValueError("No contours found in the mask image.")

            # Assume the largest contour is the main leaf shape
            main_contour = max(contours, key=cv2.contourArea)
            # Reshape to (N, 2) from (N, 1, 2)
            raw_shape_points = main_contour.squeeze().astype(np.float64)

            # Handle cases where squeeze() might result in a 1D array for 1-2 points
            if raw_shape_points.ndim == 1:
                if raw_shape_points.shape[0] == 2: # Single point, e.g., (x, y)
                    raw_shape_points = raw_shape_points.reshape(1, 2)
                else: # Should not happen if contour has > 1 point
                    raise ValueError(f"Unexpected raw_shape_points dimension after squeeze: {raw_shape_points.shape}")
            if raw_shape_points.shape[0] < 3: # Need at least 3 points for a valid polygon and affine estimation
                raise ValueError(f"Shape has too few points ({raw_shape_points.shape[0]}) to form a valid polygon or estimate transformation.")


            # --- 2. Extract raw_landmark_points ---
            if not landmark_df.empty:
                leaf_landmarks_df = landmark_df[landmark_df['Label'] == f"{leaf_id}.png"] # Match full filename
                if len(leaf_landmarks_df) == 2:
                    # Sort by index to get base then tip
                    leaf_landmarks_df = leaf_landmarks_df.sort_values(by='index')
                    base_row = leaf_landmarks_df.iloc[0]
                    tip_row = leaf_landmarks_df.iloc[1]
                    raw_landmark_points = np.array([[base_row['X'], base_row['Y']],
                                                     [tip_row['X'], tip_row['Y']]], dtype=np.float64)
                elif len(leaf_landmarks_df) > 0:
                    print(f"Warning: Found {len(leaf_landmarks_df)} landmark entries for {leaf_id}. Expected 2. Skipping landmarks for this leaf.")
                    raw_landmark_points = None
                # else: raw_landmark_points remains None

            # --- 3. Process shape with ECT to get its transformed coordinates ---
            G = EmbeddedGraph()
            G.add_cycle(raw_shape_points)

            # Store the *initial* coordinates of G. This is our `original_points` for `find_transformation_parameters`.
            # We must copy here because G.center_coordinates etc. modify G.coord_matrix in-place.
            initial_G_coord_matrix = np.copy(G.coord_matrix)

            G.center_coordinates(center_type="origin")
            G.transform_coordinates() # This is the "magic" step we need to replicate for landmarks
            G.scale_coordinates(BOUND_RADIUS)

            # G.coord_matrix now contains the fully transformed coordinates used for ECT calculation.
            # This is our `transformed_points` for `find_transformation_parameters`.

            if np.all(G.coord_matrix == 0):
                raise ValueError("Degenerate shape (all points at origin after scaling)")
            if G.coord_matrix.shape[0] < 3:
                raise ValueError(f"Processed shape has too few points ({G.coord_matrix.shape[0]}) to form a valid polygon.")

            ect_result = ect_calculator.calculate(G)

            # --- 4. Apply the SAME transformations to landmark points ---
            if raw_landmark_points is not None and len(raw_landmark_points) > 0:
                # Find the transformation from `initial_G_coord_matrix` to `G.coord_matrix`
                # which are the shape points before and after ECT's internal transformations.
                original_centroid, scale_factor, rotation_matrix = \
                    find_transformation_parameters(initial_G_coord_matrix, G.coord_matrix)

                # Now apply these parameters to the raw landmark points
                transformed_landmark_points = apply_transformation_to_points(
                    raw_landmark_points, original_centroid, scale_factor, rotation_matrix
                )
                if transformed_landmark_points.shape[0] != raw_landmark_points.shape[0]:
                    print(f"Warning: Landmark transformation for {leaf_id} resulted in unexpected shape. Skipping landmarks.")
                    transformed_landmark_points = np.array([]) # Reset if something went wrong

        except Exception as e:
            num_raw_pts = raw_shape_points.shape[0] if raw_shape_points is not None else 0
            num_proc_pts = G.coord_matrix.shape[0] if G is not None and G.coord_matrix is not None else 0

            print(f"  Skipped processing '{leaf_id}' due to error: {e}")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(mask_file_path),
                "is_processed_valid": False,
                "reason_skipped": str(e),
                "num_raw_points": num_raw_pts,
                "num_processed_points": num_proc_pts,
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": "",
                "has_landmarks": False
            })
            continue

        output_image_name = f"{leaf_id}.png"
        mask_path = shape_mask_dir / output_image_name
        ect_path = radial_ect_dir / output_image_name
        viz_path = combined_viz_dir / output_image_name

        try:
            # Save individual image components
            save_grayscale_shape_mask(G.coord_matrix, mask_path)
            save_grayscale_radial_ect(ect_result, ect_path)

            # Create combined visualization FROM THE SAVED IMAGES, with landmarks
            create_combined_viz_from_images(ect_path, mask_path, viz_path,
                                             landmark_points_transformed=transformed_landmark_points)

        except Exception as e:
            print(f"  Error saving images for '{leaf_id}': {e}. Marking as invalid.")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(mask_file_path),
                "is_processed_valid": False,
                "reason_skipped": f"Image saving failed: {e}",
                "num_raw_points": raw_shape_points.shape[0],
                "num_processed_points": G.coord_matrix.shape[0],
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": "",
                "has_landmarks": False
            })
            continue

        metadata_records.append({
            "leaf_id": leaf_id,
            "raw_file_path": str(mask_file_path),
            "is_processed_valid": True,
            "reason_skipped": "",
            "num_raw_points": raw_shape_points.shape[0],
            "num_processed_points": G.coord_matrix.shape[0],
            "file_shape_mask": str(mask_path.relative_to(output_base_dir)),
            "file_radial_ect": str(ect_path.relative_to(output_base_dir)),
            "file_combined_viz": str(viz_path.relative_to(output_base_dir)),
            "has_landmarks": True if raw_landmark_points is not None and len(raw_landmark_points) > 0 else False
        })
        processed_count += 1

    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(metadata_file, index=False)

    print(f"\n--- Processing Complete ---")
    print(f"Total files considered: {total_files}")
    print(f"Shapes successfully processed and saved: {processed_count}")
    print(f"Shapes skipped (invalid processing/saving): {skipped_count}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Processed images saved in: {output_base_dir}/{{shape_masks, radial_ects, combined_viz}}")

if __name__ == "__main__":
    # Ensure the RAW_LEAF_SHAPES_DIR now points to the parent of the MASKS folder
    # For example, if MASKS is in FINAL_ALIGNED_LEAVES_512X512, then:
    RAW_LEAF_SHAPES_DIR_WITH_MASKS = Path("FINAL_ALIGNED_LEAVES_512X512/") # Adjusted input dir

    if not (RAW_LEAF_SHAPES_DIR_WITH_MASKS / "MASKS").exists():
        print(f"Error: Input directory for masks '{RAW_LEAF_SHAPES_DIR_WITH_MASKS / 'MASKS'}' not found.")
        print("Please ensure your mask PNGs are in this directory.")
        sys.exit(1)

    if not LANDMARK_DATA_FILE.exists():
        print(f"Error: Landmark data file '{LANDMARK_DATA_FILE}' not found.")
        print("Please ensure 'training_data_results.csv' is in the current working directory.")
        sys.exit(1)

    process_raw_leaf_shapes(RAW_LEAF_SHAPES_DIR_WITH_MASKS, PROCESSED_DATA_OUTPUT_DIR, LANDMARK_DATA_FILE)