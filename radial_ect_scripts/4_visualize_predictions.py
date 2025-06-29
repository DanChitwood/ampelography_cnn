import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
from tqdm.auto import tqdm # For smart progress bars

# --- Configuration Parameters (Consistent with previous scripts) ---
BOUND_RADIUS = 1
IMAGE_SIZE = (256, 256) # Output size for all images (masks, ECT, RGB)

# Input Directories/Files
MODEL_INPUTS_BASE_DIR = Path("model_inputs")
PREDICTIONS_CSV_FILE = Path("predictions.csv") # Output from 3_predict_landmarks.py

# Output Directory
MODEL_OUTPUTS_BASE_DIR = Path("model_outputs")

# Subdirectory for visualizations with predictions
PREDICTED_VIZ_DIR = MODEL_OUTPUTS_BASE_DIR / "combined_viz_with_predictions"

# Output file for predictions in pixel coordinates
PREDICTIONS_PIXEL_CSV_FILE = MODEL_OUTPUTS_BASE_DIR / "predictions_pixel_coords.csv"

# Landmark Drawing Parameters
LANDMARK_COLOR = (255, 0, 0) # Red for predicted landmarks (RGB)
LANDMARK_RADIUS = 3 # Radius for drawing landmarks

# --- Helper Function for Coordinate Transformation (from previous scripts) ---

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
    
    # Ensure coords_ect is float64 for consistent calculations
    coords_ect = coords_ect.astype(np.float64)

    # After 90-degree CCW rotation and reflection, the mapping is:
    # ECT's Y-coordinate becomes the conceptual X-coordinate.
    # ECT's X-coordinate becomes the conceptual Y-coordinate.
    display_x_conceptual = coords_ect[:, 1]  # ECT Y maps to conceptual X
    display_y_conceptual = coords_ect[:, 0]  # ECT X maps to conceptual Y

    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2 

    # Map to pixel coordinates. Remember image Y-axis is typically "down".
    # Negate display_y_conceptual for Y-down mapping
    pixel_x = (display_x_conceptual * scale_factor + offset_x)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y)
    
    # Ensure pixel coordinates are within image bounds [0, IMAGE_SIZE-1]
    pixel_x = np.clip(pixel_x, 0, image_size[0] - 1).astype(int)
    pixel_y = np.clip(pixel_y, 0, image_size[1] - 1).astype(int)
    
    return np.column_stack((pixel_x, pixel_y))

# --- Main Visualization Logic ---
def visualize_predicted_landmarks():
    print(f"Starting visualization of predicted landmarks...")
    print(f"Reading predictions from: {PREDICTIONS_CSV_FILE}")
    print(f"Input visualizations from: {MODEL_INPUTS_BASE_DIR / 'combined_viz'}")
    print(f"Saving outputs to: {MODEL_OUTPUTS_BASE_DIR}")

    # --- 1. Setup Output Directories ---
    if MODEL_OUTPUTS_BASE_DIR.exists():
        print(f"Clearing existing output directory: {MODEL_OUTPUTS_BASE_DIR}")
        shutil.rmtree(MODEL_OUTPUTS_BASE_DIR)
    
    PREDICTED_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {PREDICTED_VIZ_DIR}")

    # --- 2. Load Data ---
    if not PREDICTIONS_CSV_FILE.exists():
        print(f"Error: Predicted landmarks CSV not found at {PREDICTIONS_CSV_FILE}")
        sys.exit(1)
    predictions_df = pd.read_csv(PREDICTIONS_CSV_FILE)

    model_inputs_metadata_file = MODEL_INPUTS_BASE_DIR / "metadata.csv"
    if not model_inputs_metadata_file.exists():
        print(f"Error: Model inputs metadata CSV not found at {model_inputs_metadata_file}")
        sys.exit(1)
    model_inputs_metadata_df = pd.read_csv(model_inputs_metadata_file)

    # Merge predictions with model_inputs_metadata to get combined_viz paths
    # Use a suffix to distinguish original columns if leaf_id is not unique (though it should be here)
    merged_df = pd.merge(predictions_df, model_inputs_metadata_df, on='leaf_id', how='inner')

    if merged_df.empty:
        print("Error: No common leaf_ids found between predictions.csv and model_inputs/metadata.csv. Exiting.")
        sys.exit(1)

    print(f"Found {len(merged_df)} predictions to visualize.")

    # --- 3. Process and Visualize Each Leaf ---
    pixel_predictions_records = []
    
    for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Visualizing Predictions"):
        leaf_id = row['leaf_id']
        combined_viz_relative_path = row['file_combined_viz']
        
        # Construct full path to the combined_viz image
        combined_viz_path = MODEL_INPUTS_BASE_DIR / combined_viz_relative_path
        
        if not combined_viz_path.exists():
            print(f"Warning: Combined visualization image not found for {leaf_id} at {combined_viz_path}. Skipping.")
            continue

        # Get predicted landmarks in normalized (-1 to 1) space
        predicted_base_x_norm = row['predicted_base_x']
        predicted_base_y_norm = row['predicted_base_y']
        predicted_tip_x_norm = row['predicted_tip_x']
        predicted_tip_y_norm = row['predicted_tip_y']

        predicted_landmarks_norm = np.array([
            [predicted_base_x_norm, predicted_base_y_norm],
            [predicted_tip_x_norm, predicted_tip_y_norm]
        ])

        # Transform to pixel coordinates (256x256)
        predicted_landmarks_pixel = ect_coords_to_pixels(predicted_landmarks_norm, IMAGE_SIZE, BOUND_RADIUS)
        
        predicted_base_x_pixel, predicted_base_y_pixel = predicted_landmarks_pixel[0]
        predicted_tip_x_pixel, predicted_tip_y_pixel = predicted_landmarks_pixel[1]

        # Add to pixel predictions record
        pixel_predictions_records.append({
            'leaf_id': leaf_id,
            'predicted_base_x_pixel': predicted_base_x_pixel,
            'predicted_base_y_pixel': predicted_base_y_pixel,
            'predicted_tip_x_pixel': predicted_tip_x_pixel,
            'predicted_tip_y_pixel': predicted_tip_y_pixel
        })

        # Load the image
        img = Image.open(combined_viz_path).convert("RGB") # Ensure it's RGB for drawing red marks
        draw = ImageDraw.Draw(img)

        # Draw predicted base landmark
        draw.ellipse([
            predicted_base_x_pixel - LANDMARK_RADIUS, predicted_base_y_pixel - LANDMARK_RADIUS,
            predicted_base_x_pixel + LANDMARK_RADIUS, predicted_base_y_pixel + LANDMARK_RADIUS
        ], fill=LANDMARK_COLOR, outline=LANDMARK_COLOR)

        # Draw predicted tip landmark
        draw.ellipse([
            predicted_tip_x_pixel - LANDMARK_RADIUS, predicted_tip_y_pixel - LANDMARK_RADIUS,
            predicted_tip_x_pixel + LANDMARK_RADIUS, predicted_tip_y_pixel + LANDMARK_RADIUS
        ], fill=LANDMARK_COLOR, outline=LANDMARK_COLOR)

        # Save the visualized image
        output_image_path = PREDICTED_VIZ_DIR / f"{leaf_id}.png"
        img.save(output_image_path)

    # --- 4. Save Pixel-Transformed Predictions CSV ---
    if pixel_predictions_records:
        pixel_predictions_df = pd.DataFrame(pixel_predictions_records)
        pixel_predictions_df.to_csv(PREDICTIONS_PIXEL_CSV_FILE, index=False)
        print(f"\nPixel-transformed predictions saved to: {PREDICTIONS_PIXEL_CSV_FILE.resolve()}")
    else:
        print("\nNo pixel-transformed predictions to save (no valid images processed).")

    print("\nLandmark visualization script complete!")

if __name__ == "__main__": # Corrected line!
    visualize_predicted_landmarks()