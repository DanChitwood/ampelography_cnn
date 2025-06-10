import os
import numpy as np
from PIL import Image
import shutil
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk # For associating veins/background with blade

def prepare_prediction_data(
    input_folder: str = "OUTPUTS_MASKS_SEGMENTED",
    output_masks_folder: str = "PREDICTION_MASKS",
    output_rgb_folder: str = "PREDICTION_RGB_IMAGES",
    resolution: int = 512 # Extracted from your training script
):
    """
    Prepares segmented mask images for model prediction by isolating the largest
    connected blade component, including associated veins and background, and
    applying the same scaling/translation transformations as the training data.

    Args:
        input_folder (str): Path to the folder containing sub-folders of segmented masks.
                            Default: "OUTPUTS_MASKS_SEGMENTED".
        output_masks_folder (str): Folder to save the prepared grayscale masks (PNG).
                                   Default: "PREDICTION_MASKS".
        output_rgb_folder (str): Folder to save the RGB visualizations of the prepared masks (PNG).
                                 Default: "PREDICTION_RGB_IMAGES".
        resolution (int): Desired height and width of the output square images.
                          Must match the resolution used for training.
    """
    print(f"--- Starting preparation of prediction data from '{input_folder}' ---")
    print(f"  Output Resolution: {resolution}x{resolution}")

    # Clear existing output folders
    if os.path.exists(output_masks_folder):
        print(f"Clearing existing output masks folder: {output_masks_folder}...")
        shutil.rmtree(output_masks_folder)
    os.makedirs(output_masks_folder, exist_ok=True)

    if os.path.exists(output_rgb_folder):
        print(f"Clearing existing output RGB folder: {output_rgb_folder}...")
        shutil.rmtree(output_rgb_folder)
    os.makedirs(output_rgb_folder, exist_ok=True)

    processed_count = 0

    # Walk through the input folder and its sub-folders
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")): # Assuming .jpg as per your description, but can handle .png
                file_path = os.path.join(root, file)
                original_filename_base, _ = os.path.splitext(file) # Get filename without extension

                print(f"Processing: {file_path}")

                try:
                    # Load the image as a NumPy array (grayscale)
                    mask_img_pil = Image.open(file_path).convert("L")
                    mask_np = np.array(mask_img_pil)

                    # --- Step 1: Isolate the largest connected blade component ---
                    blade_mask = (mask_np == 1).astype(np.uint8) # Binary mask for blade pixels
                    if np.sum(blade_mask) == 0:
                        print(f"  Warning: No blade pixels found in {file}. Skipping.")
                        continue

                    labeled_blade = label(blade_mask, connectivity=2) # Find connected components
                    regions = regionprops(labeled_blade)

                    if not regions:
                        print(f"  Warning: No connected blade components found in {file}. Skipping.")
                        continue

                    # Find the largest blade component
                    largest_region = max(regions, key=lambda r: r.area)
                    largest_blade_mask = (labeled_blade == largest_region.label).astype(np.uint8)

                    # Expand the largest blade component to include associated veins and background
                    # We create a mask for the entire leaf (blade + veins) from the original image,
                    # then mask it with the bounding box of the largest blade component to implicitly
                    # include its associated veins and background.
                    # A small dilation ensures we capture the boundary accurately for the bounding box.
                    dilated_largest_blade = binary_dilation(largest_blade_mask, footprint=disk(3)) # Dilate slightly for robustness

                    # Find the bounding box of the dilated largest blade component
                    rows, cols = np.where(dilated_largest_blade == 1)
                    if len(rows) == 0: # Should not happen if largest_region was found
                        print(f"  Warning: Dilated blade has no pixels for {file}. Skipping.")
                        continue

                    min_r, max_r = np.min(rows), np.max(rows)
                    min_c, max_c = np.min(cols), np.max(cols)

                    # Create a new mask containing only the relevant region (largest blade + its associated pixels)
                    isolated_leaf_mask_np = np.zeros_like(mask_np)
                    # Use the bounding box from the dilated largest blade to crop the original mask
                    isolated_leaf_mask_np[min_r:max_r+1, min_c:max_c+1] = \
                        mask_np[min_r:max_r+1, min_c:max_c+1]

                    # Ensure that only the largest blade component and pixels within its *original* extent
                    # that were originally blade or vein are kept. Background (0) is kept for non-blade/vein areas.
                    # We specifically want to keep the blade (1) and vein (2) pixels that are
                    # part of the largest connected component's original footprint.
                    # A more precise way: identify all pixels (0, 1, 2) that are part of the original large component's bounding box
                    # AND where the blade pixel is 1. We also include veins (2) and background (0) in that same bounding box.
                    
                    # Create a mask where only pixels inside the bounding box of the largest component are kept
                    temp_mask = np.zeros_like(mask_np)
                    temp_mask[min_r:max_r+1, min_c:max_c+1] = mask_np[min_r:max_r+1, min_c:max_c+1]

                    # Now, clean up any blade/vein pixels *outside* the largest connected component
                    # by setting them to background (0)
                    final_processed_mask_np = np.copy(temp_mask)
                    
                    # All blade pixels that are NOT part of the largest_blade_mask should be background
                    final_processed_mask_np[np.logical_and(mask_np == 1, labeled_blade != largest_region.label)] = 0
                    
                    # All vein pixels that are NOT within the bounding box of the largest component should be background
                    # This implies: if a vein pixel is *outside* the bounding box derived from the largest blade component,
                    # it should be set to background. This logic is largely handled by temp_mask.
                    # What if a vein is inside the bounding box but not "connected" to the largest blade component?
                    # The instruction was "vein or background pixels associated with the largest single connected component of blade will be included".
                    # This is best interpreted as "any blade=1, vein=2, or background=0 pixel that falls within the *bounding box* of the largest connected
                    # component of blade pixels". The previous temp_mask creation does this effectively.
                    
                    # We just need to make sure any stray blade pixels that *aren't* part of the largest component are removed.
                    # The `labeled_blade != largest_region.label` check ensures this for blade pixels.
                    
                    # For veins: if they are inside the bounding box of the largest blade component, they are kept.
                    # If they are outside, they are already 0 in `temp_mask`. This seems correct.

                    # --- Step 2: Apply the same transformations (scaling and translation) ---
                    # The training script uses all_coords_original.min/max(axis=0) to determine scaling.
                    # Here, we use the min/max of the *isolated* blade and vein pixels.
                    
                    # Get coordinates of all non-background pixels in the isolated mask
                    rows_final, cols_final = np.where(final_processed_mask_np != 0)
                    if len(rows_final) == 0:
                        print(f"  Warning: No foreground pixels remaining after isolation for {file}. Skipping.")
                        continue

                    # Create a dummy array of coordinates for scaling
                    current_coords = np.vstack([cols_final, rows_final]).T # PIL (x,y) -> NumPy (col, row)
                    
                    min_xy_current = current_coords.min(axis=0)
                    max_xy_current = current_coords.max(axis=0)

                    range_xy_current = max_xy_current - min_xy_current
                    max_range_current = max(np.max(range_xy_current) if np.max(range_xy_current) > 0 else 1, 1e-6)
                    scale = resolution * 0.9 / max_range_current # 0.9 for a border

                    scaled_coords = (current_coords - min_xy_current) * scale
                    offset = (resolution - (max_xy_current - min_xy_current) * scale) / 2
                    final_transformed_coords = np.round(scaled_coords + offset).astype(int)

                    # Create a new blank mask and draw the transformed pixels
                    transformed_mask_np = np.zeros((resolution, resolution), dtype=np.uint8)
                    
                    # Map the original pixel values (0, 1, 2) to their new positions
                    for original_idx, (c, r) in enumerate(current_coords):
                        new_c, new_r = final_transformed_coords[original_idx]
                        if 0 <= new_r < resolution and 0 <= new_c < resolution:
                            transformed_mask_np[new_r, new_c] = final_processed_mask_np[r, c]

                    # --- Save the processed mask and RGB visualization ---
                    
                    # Save mask for model input (grayscale PNG)
                    mask_output_path = os.path.join(output_masks_folder, f"{original_filename_base}.png")
                    Image.fromarray(transformed_mask_np, mode="L").save(mask_output_path)

                    # Create and save RGB visualization
                    rgb_viz_np = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                    # Blade: Orange (255, 140, 0)
                    rgb_viz_np[transformed_mask_np == 1] = [255, 140, 0]
                    # Vein: Magenta (255, 0, 255)
                    rgb_viz_np[transformed_mask_np == 2] = [255, 0, 255]
                    
                    rgb_output_path = os.path.join(output_rgb_folder, f"{original_filename_base}.png")
                    Image.fromarray(rgb_viz_np, mode="RGB").save(rgb_output_path)
                    
                    processed_count += 1
                    print(f"  Saved transformed mask and RGB for '{original_filename_base}.png'")

                except Exception as e:
                    print(f"  ERROR processing {file_path}: {e}. Skipping this file.")
                    continue

    print(f"\n--- Finished preparing {processed_count} prediction images. ---")
    print(f"Prepared masks saved to: '{output_masks_folder}'")
    print(f"RGB visualizations saved to: '{output_rgb_folder}'")

# Example usage (run this after defining the function)
if __name__ == "__main__":
    # Ensure you have scikit-image installed: pip install scikit-image
    
    # Create dummy input structure for testing if it doesn't exist
    if not os.path.exists("OUTPUTS_MASKS_SEGMENTED"):
        print("Creating dummy input data for testing...")
        os.makedirs("OUTPUTS_MASKS_SEGMENTED/AHMEUR_BOU_AHMEUR_ab", exist_ok=True)
        os.makedirs("OUTPUTS_MASKS_SEGMENTED/LOULI_ab", exist_ok=True)

        # Create dummy images (a simple square for blade, a line for vein)
        def create_dummy_mask(path, blade_val, vein_val, background_val):
            img_np = np.zeros((100, 100), dtype=np.uint8)
            # Blade (a square)
            img_np[20:80, 20:80] = blade_val
            # Veins (a line)
            img_np[30:70, 45:55] = vein_val
            
            # Add a second, smaller "blade" component to test largest connected component logic
            img_np[5:10, 5:10] = blade_val

            Image.fromarray(img_np, mode="L").save(path)

        create_dummy_mask("OUTPUTS_MASKS_SEGMENTED/AHMEUR_BOU_AHMEUR_ab/AHMEUR_BOU_AHMEUR_ab_1.jpg", 1, 2, 0)
        create_dummy_mask("OUTPUTS_MASKS_SEGMENTED/AHMEUR_BOU_AHMEUR_ab/AHMEUR_BOU_AHMEUR_ab_2.jpg", 1, 2, 0)
        create_dummy_mask("OUTPUTS_MASKS_SEGMENTED/LOULI_ab/LOULI_ab_1.jpg", 1, 2, 0)
        print("Dummy data created.")

    # Run the preparation script
    prepare_prediction_data()