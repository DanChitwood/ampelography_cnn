import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
import cv2

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
IMAGE_SIZE = (256, 256)

PROCESSED_DATA_OUTPUT_DIR = Path("processed_leaf_data/")
SHAPE_MASK_DIR = PROCESSED_DATA_OUTPUT_DIR / "shape_masks"
RADIAL_ECT_DIR = PROCESSED_DATA_OUTPUT_DIR / "radial_ects"
COMBINED_VIZ_DIR = PROCESSED_DATA_OUTPUT_DIR / "combined_viz"
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
    if len(points) == 0:
        return points
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # Use .T for opencv affine matrix, as discussed
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

    # Get bounds for selecting diverse points
    x_min_idx = np.argmin(src_points[:, 0])
    x_max_idx = np.argmax(src_points[:, 0])
    y_min_idx = np.argmin(src_points[:, 1])
    y_max_idx = np.argmax(src_points[:, 1])

    # Attempt to use specific points to get 3 non-collinear points
    # Prioritize points at extremes of the shape for better stability
    candidate_indices = [x_min_idx, x_max_idx, y_min_idx, y_max_idx]
    
    # Ensure unique indices and at least 3 points
    unique_indices = list(dict.fromkeys(candidate_indices)) # Preserve order, remove duplicates
    
    # If not enough unique "extreme" points, or if they are collinear, add more points from the start
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
    
    # Check for collinearity using cross product (area of triangle)
    # (x1, y1), (x2, y2), (x3, y3)
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    area = 0.5 * np.abs(src_pts_for_transform[0,0]*(src_pts_for_transform[1,1]-src_pts_for_transform[2,1]) +
                        src_pts_for_transform[1,0]*(src_pts_for_transform[2,1]-src_pts_for_transform[0,1]) +
                        src_pts_for_transform[2,0]*(src_pts_for_transform[0,1]-src_pts_for_transform[1,1]))
    
    if area < 1e-6: # Threshold for "nearly collinear"
        # If the chosen points are collinear, this is a degenerate case.
        raise ValueError("Could not find 3 non-collinear points for affine transformation. Shape is likely degenerate or a line.")

    M_2x3 = cv2.getAffineTransform(src_pts_for_transform, dst_pts_for_transform)
    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    
    return affine_matrix_3x3

# --- Visualization Functions (MODIFIED save_grayscale_shape_mask and create_combined_viz_from_images) ---

def save_grayscale_shape_mask(main_contour_points: np.ndarray, blade_pixels: np.ndarray, vein_pixels: np.ndarray, save_path: Path):
    """
    Saves a grayscale shape mask including blade and vein pixels, using distinct gray values.
    The main leaf contour is used to define the overall shape extent.
    `main_contour_points` are the ECT-transformed contour points.
    `blade_pixels` and `vein_pixels` are the ECT-transformed individual pixel coordinates.
    """
    # Create a blank black image
    img = Image.new("L", IMAGE_SIZE, MASK_BACKGROUND_GRAY) # "L" for grayscale
    draw = ImageDraw.Draw(img)

    # Convert ECT coordinates [-BOUND_RADIUS, BOUND_RADIUS] to pixel coordinates [0, IMAGE_SIZE]
    # Apply the same coordinate swap and inversion as in combined_viz for consistency
    # ECT's (x, y) becomes (y, -x) for plotting orientation, then scaled to pixels

    # Define the transformation function for internal use
    def ect_coords_to_pixels(coords_ect: np.ndarray):
        if len(coords_ect) == 0:
            return np.array([])
        
        # Apply the same coordinate swap and inversion for consistent plotting
        transformed_coords_for_plot = np.array([coords_ect[:, 1], -coords_ect[:, 0]]).T

        # Scale from [-BOUND_RADIUS, BOUND_RADIUS] to pixel range [0, IMAGE_SIZE]
        # pixel_coord = (ect_coord / BOUND_RADIUS * (IMAGE_SIZE/2)) + (IMAGE_SIZE/2)
        # Assuming IMAGE_SIZE is square (width == height) for simplicity with BOUND_RADIUS
        scale = IMAGE_SIZE[0] / (2 * BOUND_RADIUS)
        offset_x = IMAGE_SIZE[0] / 2
        offset_y = IMAGE_SIZE[1] / 2 # Using IMAGE_SIZE[1] for y offset if not square

        pixel_x = (transformed_coords_for_plot[:, 0] * scale + offset_x).astype(int)
        pixel_y = (transformed_coords_for_plot[:, 1] * scale + offset_y).astype(int)
        
        return np.column_stack((pixel_x, pixel_y))

    # 1. Draw the blade pixels
    if blade_pixels is not None and len(blade_pixels) > 0:
        blade_pixel_coords = ect_coords_to_pixels(blade_pixels)
        for x, y in blade_pixel_coords:
            # Ensure pixels are within image bounds before drawing
            if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]:
                img.putpixel((x, y), MASK_BLADE_GRAY)

    # 2. Draw the vein pixels (on top of blade, potentially overwriting if they overlap)
    if vein_pixels is not None and len(vein_pixels) > 0:
        vein_pixel_coords = ect_coords_to_pixels(vein_pixels)
        for x, y in vein_pixel_coords:
            if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]:
                img.putpixel((x, y), MASK_VEIN_GRAY)
    
    # Optionally, if you want the main contour outline explicitly drawn on top of pixels
    # This might be redundant if pixels fill the shape well, but ensures the boundary
    if main_contour_points is not None and len(main_contour_points) > 0:
        contour_pixel_coords = ect_coords_to_pixels(main_contour_points).tolist()
        # Convert to tuple of (x,y) pairs for ImageDraw.polygon
        # If your contour is closed, make sure the first point is repeated at the end.
        contour_tuples = [tuple(p) for p in contour_pixel_coords]
        if len(contour_tuples) > 2: # Need at least 3 points for a polygon
            draw.polygon(contour_tuples, outline=MASK_BLADE_GRAY) # Outline the shape
            # If you want to fill the contour, but we're already filling with pixels above
            # draw.polygon(contour_tuples, fill=MASK_BLADE_GRAY) # This would fill the shape

    img.save(save_path)

def save_grayscale_radial_ect(ect_result, save_path: Path):
    """Saves a grayscale radial ECT image using Matplotlib."""
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    # Changed cmap to 'inferno'
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap="inferno") # Changed from "gray" to "inferno"
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # Clockwise
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, save_path: Path,
                                    blade_color=(255, 255, 255), blade_alpha=0.3, # Blade is white, 30% transparency
                                    landmark_points_transformed: np.ndarray = None,
                                    landmark_color=(255, 255, 0), landmark_size=5, # Landmarks are yellow
                                    vein_points_transformed: np.ndarray = None,
                                    vein_color=(255, 255, 255), vein_size=1, # Veins are white
                                    transformed_blade_pixels: np.ndarray = None, # Transformed blade pixels
                                    main_contour_points: np.ndarray = None): # Main leaf contour to draw blade shape
    """
    Combines a grayscale ECT image, a transformed blade overlay, transformed vein pixels,
    and transformed landmark points into a single RGB visualization.
    The main contour is now drawn as the blade shape, and then blade/vein pixels are drawn on top.
    """
    try:
        # 1. Load the ECT image (background) and ensure it's RGBA
        ect_img = Image.open(ect_image_path).convert("RGBA")
        img_width, img_height = ect_img.size

        # A utility function to convert ECT coordinates to image pixel coordinates
        def ect_to_pixel(coords: np.ndarray, size: tuple, radius: float):
            if len(coords) == 0:
                return np.array([])
            
            # Apply the same coordinate swap and inversion for consistent plotting
            transformed_coords_for_plot = np.array([coords[:, 1], -coords[:, 0]]).T

            # DEFINE SCALE AND OFFSETS LOCALLY WITHIN THIS FUNCTION
            scale = size[0] / (2 * radius) # Use size[0] (width) for scale as image is square
            offset_x = size[0] / 2
            offset_y = size[1] / 2 

            pixel_x = (transformed_coords_for_plot[:, 0] * scale + offset_x).astype(int)
            pixel_y = (transformed_coords_for_plot[:, 1] * scale + offset_y).astype(int)
            
            return np.column_stack((pixel_x, pixel_y))

        # Create a base image for all overlays (initially transparent)
        composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw_composite = ImageDraw.Draw(composite_overlay)

        # 2. Draw the main leaf blade shape (using transformed_blade_pixels or main_contour_points)
        # If transformed_blade_pixels are available and preferred for filling the blade area
        if transformed_blade_pixels is not None and len(transformed_blade_pixels) > 0:
            blade_pixel_coords = ect_to_pixel(transformed_blade_pixels, (img_width, img_height), BOUND_RADIUS)
            for x, y in blade_pixel_coords:
                draw_composite.rectangle([x, y, x, y], fill=(blade_color[0], blade_color[1], blade_color[2], int(255 * blade_alpha)))
        elif main_contour_points is not None and len(main_contour_points) > 0:
            # Fallback to filling the entire contour if individual blade pixels are not desired for fill
            contour_pixel_coords = ect_to_pixel(main_contour_points, (img_width, img_height), BOUND_RADIUS).tolist()
            contour_tuples = [tuple(p) for p in contour_pixel_coords]
            if len(contour_tuples) > 2:
                draw_composite.polygon(contour_tuples, fill=(blade_color[0], blade_color[1], blade_color[2], int(255 * blade_alpha)))

        # 3. Draw Transformed Vein Points onto the composite overlay
        if vein_points_transformed is not None and len(vein_points_transformed) > 0:
            vein_pixel_coords = ect_to_pixel(vein_points_transformed, (img_width, img_height), BOUND_RADIUS)
            # Use the specified vein_color and alpha for transparency
            vein_fill_color = (vein_color[0], vein_color[1], vein_color[2], int(255 * blade_alpha)) # Using blade_alpha for veins too
            for x, y in vein_pixel_coords:
                draw_composite.ellipse([x - vein_size//2, y - vein_size//2,
                                  x + vein_size//2, y + vein_size//2],
                                  fill=vein_fill_color)

        # 4. Draw Transformed Landmark Points onto the same composite overlay
        if landmark_points_transformed is not None and len(landmark_points_transformed) > 0:
            landmark_pixel_coords = ect_to_pixel(landmark_points_transformed, (img_width, img_height), BOUND_RADIUS)
            for x, y in landmark_pixel_coords:
                # Landmark color is RGB, alpha is implied 255 for solid landmarks
                draw_composite.ellipse([x - landmark_size//2, y - landmark_size//2,
                                  x + landmark_size//2, y + landmark_size//2],
                                  fill=landmark_color)
        
        # 5. Composite the combined overlay onto the ECT image
        final_combined_img = Image.alpha_composite(ect_img, composite_overlay).convert("RGB")
        final_combined_img.save(save_path)

    except FileNotFoundError:
        print(f"Error: ECT image file not found: {ect_image_path}")
    except Exception as e:
        print(f"Error creating combined visualization for {ect_image_path.stem}: {e}")
        # For debugging, you can re-raise: raise


# --- Main Data Processing Logic (MODIFIED to pass blade/vein pixels to save_grayscale_shape_mask) ---

def process_raw_leaf_shapes(raw_input_dir: Path, output_base_dir: Path, landmark_data_file: Path, clear_existing_data: bool = True):
    print(f"Starting processing of raw leaf shapes from: {raw_input_dir}")
    print(f"Output will be saved to: {output_base_dir}")

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

    landmark_df = pd.DataFrame()
    if landmark_data_file.exists():
        try:
            landmark_df = pd.read_csv(landmark_data_file)
            print(f"Loaded landmark data from: {landmark_data_file}")
        except Exception as e:
            print(f"Error loading landmark data from '{landmark_data_file}': {e}. No landmarks will be processed.")
    else:
        print(f"Warning: Landmark data file '{LANDMARK_DATA_FILE}' not found. No landmarks will be processed.")

    mask_files = list((raw_input_dir / "MASKS").glob("*.png"))
    total_files = len(mask_files)

    if total_files == 0:
        print(f"No .png mask files found in {raw_input_dir / 'MASKS'}. Exiting.")
        return

    print(f"Found {total_files} .png mask files to process.")

    for i, mask_file_path in enumerate(mask_files):
        leaf_id = mask_file_path.stem

        if (i + 1) % 100 == 0 or (i + 1) == total_files or (i + 1) == 1:
            print(f"Processing leaf shape {i+1}/{total_files} ({leaf_id})")

        full_leaf_contour_points = None
        raw_blade_pixels = np.array([])
        raw_vein_pixels = np.array([])
        raw_landmark_points = None
        G = None
        ect_affine_matrix = None

        # Transformed points for visualization and combined mask
        transformed_blade_pixels = np.array([])
        transformed_vein_pixels = np.array([])
        transformed_landmark_points = np.array([])

        try:
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

            if not landmark_df.empty:
                leaf_landmarks_df = landmark_df[landmark_df['Label'] == f"{leaf_id}.png"]
                if len(leaf_landmarks_df) == 2:
                    leaf_landmarks_df = leaf_landmarks_df.sort_values(by='index')
                    base_row = leaf_landmarks_df.iloc[0]
                    tip_row = leaf_landmarks_df.iloc[1]
                    raw_landmark_points = np.array([[base_row['X'], base_row['Y']],
                                                    [tip_row['X'], tip_row['Y']]], dtype=np.float64)
                elif len(leaf_landmarks_df) > 0:
                    print(f"Warning: Found {len(leaf_landmarks_df)} landmark entries for {leaf_id}. Expected 2. Skipping landmarks for this leaf.")
                    raw_landmark_points = None

            # --- ECT Processing ---
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
            if raw_landmark_points is not None and len(raw_landmark_points) > 0:
                transformed_landmark_points = apply_transformation_with_affine_matrix(
                    raw_landmark_points, ect_affine_matrix
                )
                if transformed_landmark_points.shape[0] != raw_landmark_points.shape[0]:
                    print(f"Warning: Landmark transformation for {leaf_id} resulted in unexpected shape. Skipping landmarks.")
                    transformed_landmark_points = np.array([])

            if raw_blade_pixels is not None and len(raw_blade_pixels) > 0:
                transformed_blade_pixels = apply_transformation_with_affine_matrix(
                    raw_blade_pixels, ect_affine_matrix
                )
            
            if raw_vein_pixels is not None and len(raw_vein_pixels) > 0:
                transformed_vein_pixels = apply_transformation_with_affine_matrix(
                    raw_vein_pixels, ect_affine_matrix
                )

        except Exception as e:
            num_raw_full_leaf_pts = full_leaf_contour_points.shape[0] if full_leaf_contour_points is not None else 0
            num_proc_full_leaf_pts = G.coord_matrix.shape[0] if G is not None and G.coord_matrix is not None else 0

            print(f"  Skipped processing '{leaf_id}' due to error: {e}")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(mask_file_path),
                "is_processed_valid": False,
                "reason_skipped": str(e),
                "num_raw_points_full_leaf": num_raw_full_leaf_pts,
                "num_processed_points_full_leaf": num_proc_full_leaf_pts,
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": "",
                "has_landmarks": False, "has_veins": False, "has_blade": False
            })
            continue

        output_image_name = f"{leaf_id}.png"
        mask_path = shape_mask_dir / output_image_name
        ect_path = radial_ect_dir / output_image_name
        viz_path = combined_viz_dir / output_image_name

        try:
            # Save individual ECT-transformed full leaf shape mask with blade and vein pixels
            save_grayscale_shape_mask(G.coord_matrix, transformed_blade_pixels, transformed_vein_pixels, mask_path)
            # ECT saved with 'inferno' colormap
            save_grayscale_radial_ect(ect_result, ect_path)

            # Create combined visualization with all transformed components
            create_combined_viz_from_images(ect_path, viz_path,
                                            blade_color=(255, 255, 255), # White for blade
                                            blade_alpha=0.3,             # 30% transparency for blade
                                            landmark_points_transformed=transformed_landmark_points,
                                            landmark_color=(255, 255, 0), # Yellow for landmarks
                                            vein_points_transformed=transformed_vein_pixels,
                                            vein_color=(255, 255, 255),  # White for veins
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
                "num_raw_points_full_leaf": full_leaf_contour_points.shape[0],
                "num_processed_points_full_leaf": G.coord_matrix.shape[0],
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": "",
                "has_landmarks": False, "has_veins": False, "has_blade": False
            })
            continue

        metadata_records.append({
            "leaf_id": leaf_id,
            "raw_file_path": str(mask_file_path),
            "is_processed_valid": True,
            "reason_skipped": "",
            "num_raw_points_full_leaf": full_leaf_contour_points.shape[0],
            "num_processed_points_full_leaf": G.coord_matrix.shape[0],
            "file_shape_mask": str(mask_path.relative_to(output_base_dir)),
            "file_radial_ect": str(ect_path.relative_to(output_base_dir)),
            "file_combined_viz": str(viz_path.relative_to(output_base_dir)),
            "has_landmarks": True if raw_landmark_points is not None and len(raw_landmark_points) > 0 else False,
            "has_veins": True if raw_vein_pixels is not None and len(raw_vein_pixels) > 0 else False,
            "has_blade": True if raw_blade_pixels is not None and len(raw_blade_pixels) > 0 else False
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
    RAW_LEAF_SHAPES_DIR_WITH_MASKS = Path("FINAL_ALIGNED_LEAVES_512X512/")

    if not (RAW_LEAF_SHAPES_DIR_WITH_MASKS / "MASKS").exists():
        print(f"Error: Input directory for masks '{RAW_LEAF_SHAPES_DIR_WITH_MASKS / 'MASKS'}' not found.")
        print("Please ensure your mask PNGs are in this directory.")
        sys.exit(1)

    if not LANDMARK_DATA_FILE.exists():
        print(f"Error: Landmark data file '{LANDMARK_DATA_FILE}' not found.")
        print("Please ensure 'training_data_results.csv' is in the current working directory.")
        sys.exit(1)

    process_raw_leaf_shapes(RAW_LEAF_SHAPES_DIR_WITH_MASKS, PROCESSED_DATA_OUTPUT_DIR, LANDMARK_DATA_FILE)