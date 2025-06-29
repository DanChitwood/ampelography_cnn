import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import sys
import shutil
from tqdm.auto import tqdm
from skimage.transform import AffineTransform, warp
from skimage.registration import phase_cross_correlation

# --- Configuration Parameters ---
IMAGE_SIZE = (256, 256) # All images are this size
BOUND_RADIUS = 1 # Consistent with ECT generation

# Input Directories/Files
MODEL_INPUTS_BASE_DIR = Path("model_inputs")

# Output Directory for this script
REGISTRATION_OUTPUT_BASE_DIR = Path("registration_output")

# Subdirectories for aligned outputs
ALIGNED_ECTS_DIR = REGISTRATION_OUTPUT_BASE_DIR / "aligned_ects"
ALIGNED_MASKS_DIR = REGISTRATION_OUTPUT_BASE_DIR / "aligned_masks"
ALIGNED_COMBINED_VIZ_DIR = REGISTRATION_OUTPUT_BASE_DIR / "aligned_combined_viz"

# Output file for alignment transformations
ALIGNMENT_TRANSFORMS_CSV = REGISTRATION_OUTPUT_BASE_DIR / "alignment_transforms.csv"

# --- Alignment Parameters ---
# The ID of the leaf whose SHAPE MASK will serve as the reference for alignment.
# By default, it will pick the first leaf found in metadata.csv.
# You can set it to a specific leaf_id, e.g., REFERENCE_LEAF_ID = "Plant001_Leaf01"
REFERENCE_LEAF_ID = None 

# An additional, constant rotation applied to ALL aligned images (in degrees).
# Use this to globally orient all leaves to a desired canonical direction (e.g., tip up).
# You might need to run the script once with 0, inspect results in aligned_masks,
# and then adjust this value (e.g., if aligned leaves consistently point right, set to -90).
CANONICAL_OFFSET_ANGLE_DEG = 0 # Example: 0 for no extra rotation. Adjust after first run.

# --- Helper Functions ---

def load_image_as_float(path: Path):
    """
    Loads an image for phase_cross_correlation. Converts it to grayscale ('L') if necessary
    and then normalizes to float [0, 1].
    """
    img = Image.open(path)
    if img.mode != 'L':
        # Convert to grayscale for numerical correlation, suppress print
        img = img.convert('L') 
    return np.array(img).astype(float) / 255.0 # Normalize to [0, 1]

def apply_transformation_and_save(img_path: Path, rotation_deg: float, translation_x: float, translation_y: float, output_dir: Path, filename: str, interp_order: int):
    """
    Applies rotation (around center) and translation to an image using skimage.transform.warp
    and saves it. Handles image mode conversion explicitly.
    
    Args:
        img_path (Path): Path to the input image.
        rotation_deg (float): Rotation angle in degrees (CCW for skimage).
        translation_x (float): Translation in x (columns/width) in pixels.
        translation_y (float): Translation in y (rows/height) in pixels.
        output_dir (Path): Directory to save the transformed image.
        filename (str): Name of the output file (e.g., "leaf_id.png").
        interp_order (int): Interpolation order (0 for nearest, 1 for linear, 3 for cubic).
                           For masks, use 0 (nearest) to preserve discrete values.
    """
    img_pil = Image.open(img_path)
    
    # Determine the target mode based on the output directory
    # and convert the PIL image to that mode FIRST.
    if output_dir == ALIGNED_ECTS_DIR or output_dir == ALIGNED_MASKS_DIR:
        if img_pil.mode != 'L':
            # Convert to 'L' silently
            img_pil = img_pil.convert('L')
    elif output_dir == ALIGNED_COMBINED_VIZ_DIR:
        if img_pil.mode != 'RGB':
            # Convert to 'RGB' silently
            img_pil = img_pil.convert('RGB')
    else:
        # This case should ideally not be reached if output_dir is one of the predefined ones.
        raise ValueError(f"Unknown output directory type: {output_dir} for {img_path}. Cannot process.")

    # Now, img_pil is guaranteed to be in 'L' or 'RGB' mode.
    if img_pil.mode == 'L':
        # For masks (interp_order=0), keep original uint8 range to preserve 0,1,2 values
        if interp_order == 0:
            np_img = np.array(img_pil, dtype=np.uint8)
        else: # For ECTs, convert to float 0-1 range for warp
            np_img = np.array(img_pil, dtype=float) / 255.0
            
    elif img_pil.mode == 'RGB': # Combined Viz, convert to float 0-1 range for warp
        np_img = np.array(img_pil, dtype=float) / 255.0
    else:
        # This 'else' block should technically be unreachable if conversions above are robust.
        raise RuntimeError(f"Unexpected image mode {img_pil.mode} after conversion attempts for {img_path}")


    # Calculate image center (for rotation around center)
    center_y, center_x = np_img.shape[0] / 2 - 0.5, np_img.shape[1] / 2 - 0.5
    
    rotation_radians = np.deg2rad(rotation_deg)

    # Define the affine transformation.
    # It first translates the image so its center is at the origin,
    # then rotates, then translates it back to its original position,
    # and finally adds the calculated pixel translation.
    tform = AffineTransform(
        translation=[-center_x, -center_y]  # 1. Move center to origin
    ) + AffineTransform(
        rotation=rotation_radians           # 2. Rotate
    ) + AffineTransform(
        translation=[center_x + translation_x, center_y + translation_y] # 3. Move back + apply translation
    )

    # Apply the inverse transformation to the output grid to sample from the input image.
    transformed_img = warp(np_img, inverse_map=tform.inverse,
                           output_shape=np_img.shape[:2], # Ensure 2D output shape even for RGB (warp handles channels)
                           mode='constant', cval=0.0, order=interp_order,
                           preserve_range=True) # Keep original range of values

    # Convert back to PIL Image and save
    if img_pil.mode == 'L':
        if interp_order == 0: # For masks, round to nearest integer and ensure uint8
            final_img_array = np.round(transformed_img).astype(np.uint8)
        else: # For ECTs, scale back from 0-1 float to 0-255 uint8
            final_img_array = np.clip(transformed_img * 255.0, 0, 255).astype(np.uint8)
        final_img = Image.fromarray(final_img_array, mode='L')
    elif img_pil.mode == 'RGB':
        # Scale back from 0-1 float to 0-255 uint8, clip values
        final_img_array = np.clip(transformed_img * 255.0, 0, 255).astype(np.uint8)
        final_img = Image.fromarray(final_img_array, mode='RGB')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    final_img.save(output_dir / filename)


# --- Main Alignment Logic ---
def align_leaves():
    print(f"Starting leaf alignment process...")
    print(f"Input images from: {MODEL_INPUTS_BASE_DIR}")
    print(f"Saving aligned outputs to: {REGISTRATION_OUTPUT_BASE_DIR}")

    # --- 1. Setup Output Directories ---
    if REGISTRATION_OUTPUT_BASE_DIR.exists():
        print(f"Clearing existing output directory: {REGISTRATION_OUTPUT_BASE_DIR}")
        shutil.rmtree(REGISTRATION_OUTPUT_BASE_DIR)
    
    ALIGNED_ECTS_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNED_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNED_COMBINED_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories.")

    # --- 2. Load Metadata and Select Reference Leaf ---
    metadata_file = MODEL_INPUTS_BASE_DIR / "metadata.csv"
    if not metadata_file.exists():
        print(f"Error: Metadata CSV not found at {metadata_file}. Exiting.")
        sys.exit(1)
    metadata_df = pd.read_csv(metadata_file)

    if REFERENCE_LEAF_ID is None:
        if not metadata_df.empty:
            chosen_reference_leaf_id = metadata_df['leaf_id'].iloc[0] # Pick the first one
            print(f"No REFERENCE_LEAF_ID specified. Using first leaf found: {chosen_reference_leaf_id}")
        else:
            print("Error: Metadata is empty, cannot select a reference leaf. Exiting.")
            sys.exit(1)
    else:
        if REFERENCE_LEAF_ID not in metadata_df['leaf_id'].values:
            print(f"Error: Specified REFERENCE_LEAF_ID '{REFERENCE_LEAF_ID}' not found in metadata. Exiting.")
            sys.exit(1)
        chosen_reference_leaf_id = REFERENCE_LEAF_ID
    
    reference_ect_row = metadata_df[metadata_df['leaf_id'] == chosen_reference_leaf_id].iloc[0]
    
    # --- CHANGE START ---
    # Load the SHAPE MASK of the reference leaf for alignment
    reference_mask_path = MODEL_INPUTS_BASE_DIR / reference_ect_row['file_shape_mask']

    if not reference_mask_path.exists():
        print(f"Error: Reference Shape Mask not found at {reference_mask_path}. Exiting.")
        sys.exit(1)
    
    reference_image = load_image_as_float(reference_mask_path)
    print(f"Using {chosen_reference_leaf_id}'s SHAPE MASK as the reference for alignment.")
    # --- CHANGE END ---

    # --- 3. Perform Alignment for Each Leaf ---
    alignment_records = []

    print("\nCalculating alignment transformations...")
    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Aligning Leaves"):
        leaf_id = row['leaf_id']
        
        # --- CHANGE START ---
        # Load the SHAPE MASK of the current leaf for alignment
        mask_path = MODEL_INPUTS_BASE_DIR / row['file_shape_mask']

        if not mask_path.exists():
            print(f"Warning: Shape Mask not found for {leaf_id} at {mask_path}. Skipping.")
            continue
        
        moving_image = load_image_as_float(mask_path)
        # --- CHANGE END ---

        # Calculate phase_cross_correlation. This returns shifts (row, col) and rotation_angle_deg.
        # It handles both translation and rotation efficiently.
        # shift is (row_shift, col_shift), which is (y_shift, x_shift)
        # diffphase is the rotation angle in degrees (CCW) required to transform the moving image to the reference.
        shift, error, diffphase = phase_cross_correlation(
            reference_image, moving_image, 
            upsample_factor=10, # Sub-pixel accuracy
            space="real" # Perform in real space for initial guess, then refine
        )
        
        translation_x_pixels = shift[1] # col_shift
        translation_y_pixels = shift[0] # row_shift
        
        rotation_angle_deg = -diffphase # The angle needed to align moving to reference

        alignment_records.append({
            'leaf_id': leaf_id,
            'reference_leaf_id': chosen_reference_leaf_id,
            'rotation_to_reference_deg': rotation_angle_deg,
            'translation_x_to_reference_pixels': translation_x_pixels,
            'translation_y_to_reference_pixels': translation_y_pixels
        })
    
    # Save the raw alignment records
    pd.DataFrame(alignment_records).to_csv(ALIGNMENT_TRANSFORMS_CSV, index=False)
    print(f"\nRaw alignment transformations saved to: {ALIGNMENT_TRANSFORMS_CSV.resolve()}")

    # --- 4. Apply Transformations and Save Aligned Images ---
    print(f"\nApplying transformations and saving aligned images (with canonical offset: {CANONICAL_OFFSET_ANGLE_DEG} deg)...")
    
    # Create a DataFrame from alignment_records for easier lookup
    alignment_df = pd.DataFrame(alignment_records)

    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Saving Aligned Images"):
        leaf_id = row['leaf_id']
        
        # Retrieve the calculated transformation for this leaf
        transform_data = alignment_df[alignment_df['leaf_id'] == leaf_id]
        if transform_data.empty:
            print(f"Warning: No alignment data found for {leaf_id}. Skipping aligned save.")
            continue
        transform_data = transform_data.iloc[0]
        
        individual_rotation_deg = transform_data['rotation_to_reference_deg']
        individual_translation_x = transform_data['translation_x_to_reference_pixels']
        individual_translation_y = transform_data['translation_y_to_reference_pixels']

        # Total rotation includes individual alignment + canonical offset
        total_rotation_deg = individual_rotation_deg + CANONICAL_OFFSET_ANGLE_DEG

        # File paths for original images
        ect_orig_path = MODEL_INPUTS_BASE_DIR / row['file_radial_ect']
        mask_orig_path = MODEL_INPUTS_BASE_DIR / row['file_shape_mask']
        combined_viz_orig_path = MODEL_INPUTS_BASE_DIR / row['file_combined_viz']

        if not all(p.exists() for p in [ect_orig_path, mask_orig_path, combined_viz_orig_path]):
            print(f"Warning: Original image files not found for {leaf_id}. Skipping aligned save.")
            continue

        # Apply and save for ECT (bilinear interpolation for smooth grayscale)
        apply_transformation_and_save(
            ect_orig_path, total_rotation_deg, individual_translation_x, individual_translation_y,
            ALIGNED_ECTS_DIR, f"{leaf_id}.png", interp_order=1 # Bilinear
        )

        # Apply and save for Mask (nearest neighbor interpolation to preserve 0,1,2 values)
        apply_transformation_and_save(
            mask_orig_path, total_rotation_deg, individual_translation_x, individual_translation_y,
            ALIGNED_MASKS_DIR, f"{leaf_id}.png", interp_order=0 # Nearest
        )
        
        # Apply and save for Combined Viz (bilinear interpolation for smoother colors/gradients)
        apply_transformation_and_save(
            combined_viz_orig_path, total_rotation_deg, individual_translation_x, individual_translation_y,
            ALIGNED_COMBINED_VIZ_DIR, f"{leaf_id}.png", interp_order=1 # Bilinear
        )

    print("\nLeaf alignment script complete!")

if __name__ == "__main__":
    align_leaves()