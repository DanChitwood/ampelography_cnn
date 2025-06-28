import pandas as pd
from pathlib import Path
import shutil
from PIL import Image
import re
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = Path(".") # Current working directory (where this script is run from)

TRUE_LEAVES_FILTERED_ROOT = BASE_DIR / "TRUE_LEAVES_FILTERED"
BAD_LEAVES_DIR = BASE_DIR / "bad_leaves"

FINAL_ALIGNED_LEAVES_ROOT = BASE_DIR / "FINAL_ALIGNED_LEAVES_512x512" # New folder name to reflect new dimensions
FINAL_MASKS_DIR = FINAL_ALIGNED_LEAVES_ROOT / "MASKS"
FINAL_OVERLAYS_DIR = FINAL_ALIGNED_LEAVES_ROOT / "OVERLAYS"
FINAL_RGB_CROPS_DIR = FINAL_ALIGNED_LEAVES_ROOT / "RGB_CROPS"

ALL_FINAL_DEST_DIRS = [FINAL_MASKS_DIR, FINAL_OVERLAYS_DIR, FINAL_RGB_CROPS_DIR]

LOG_FILE = BASE_DIR / "image_alignment_log_512x512.txt" # New log file name

TARGET_IMAGE_DIM = 512 # New universal target dimension for both width and height

# --- Helper Functions ---

def log_and_print(message, is_error=False, log_list=None):
    """Prints a message and optionally appends it to a log list."""
    if log_list is not None:
        log_list.append(message)
    if is_error:
        print(f"ERROR: {message}")
    else:
        print(message)

def extract_component_name(filename):
    """
    Extracts the base component name from a filename.
    Assumes suffix is _mask.png, _overlay.png, or _rgb_crop.png.
    """
    if filename.endswith("_mask.png"):
        return filename[:-len("_mask.png")]
    elif filename.endswith("_overlay.png"):
        return filename[:-len("_overlay.png")]
    elif filename.endswith("_rgb_crop.png"):
        return filename[:-len("_rgb_crop.png")]
    
    return filename # Fallback if suffix doesn't match

def resize_and_pad_image(img, target_dim):
    """
    Resizes an image to fit within target_dim x target_dim while preserving aspect ratio,
    then pads it to exactly target_dim x target_dim with a black background.
    Handles different image modes (RGB, L).
    """
    original_width, original_height = img.size

    # Calculate scale factor to fit the image within the target_dim x target_dim square
    # The image will be scaled so its largest side matches target_dim
    scale = min(target_dim / original_width, target_dim / original_height)
    
    # Calculate new dimensions after scaling
    scaled_width = int(original_width * scale)
    scaled_height = int(original_height * scale)

    # Resize the image using a high-quality resampling filter (LANCZOS is good for both up/downsampling)
    img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

    # Determine background image mode and color based on original image mode
    # For masks (L or P), ensure black is 0. For RGB, it's (0,0,0).
    if img.mode == 'L' or img.mode == 'P':
        new_img = Image.new('L', (target_dim, target_dim), 0) # Black for grayscale/palette
    elif img.mode == 'RGB':
        new_img = Image.new('RGB', (target_dim, target_dim), (0, 0, 0)) # Black for RGB
    else:
        # For other modes, convert to RGB for consistency.
        log_and_print(f"  INFO: Image mode '{img.mode}' not directly handled for padding, converting to RGB for consistency.", is_error=False)
        img = img.convert('RGB')
        new_img = Image.new('RGB', (target_dim, target_dim), (0, 0, 0))

    # Calculate paste position to center the scaled image on the new canvas
    paste_x = (target_dim - scaled_width) // 2
    paste_y = (target_dim - scaled_height) // 2

    new_img.paste(img, (paste_x, paste_y))
    return new_img

# --- Main Logic ---
def main():
    log_messages = []

    log_and_print(f"--- Starting Image Alignment and Filtering Process (512x512) ---", log_list=log_messages)
    log_and_print(f"Source root: {TRUE_LEAVES_FILTERED_ROOT}", log_list=log_messages)
    log_and_print(f"Bad leaves dir: {BAD_LEAVES_DIR}", log_list=log_messages)
    log_and_print(f"Final destination root: {FINAL_ALIGNED_LEAVES_ROOT}\n", log_list=log_messages)
    log_and_print(f"Target image dimensions: {TARGET_IMAGE_DIM}x{TARGET_IMAGE_DIM}\n", log_list=log_messages)

    # 1. Prepare exclusion lists based on bad_leaves
    bad_component_names = set()
    if BAD_LEAVES_DIR.exists():
        log_and_print(f"Scanning bad_leaves folder for exclusions...", log_list=log_messages)
        for f in tqdm(list(BAD_LEAVES_DIR.iterdir()), desc="Reading bad_leaves"):
            if f.is_file() and f.suffix == '.png':
                comp_name = extract_component_name(f.name)
                bad_component_names.add(comp_name)
        log_and_print(f"Found {len(bad_component_names)} components to exclude from bad_leaves.", log_list=log_messages)
    else:
        log_and_print(f"WARNING: '{BAD_LEAVES_DIR}' not found. No bad leaves will be excluded based on this folder.", is_error=True, log_list=log_messages)

    # 2. Create new destination directories
    log_and_print(f"\nCreating final destination directories...", log_list=log_messages)
    for d in ALL_FINAL_DEST_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        log_and_print(f"  Created/Ensured: {d}", log_list=log_messages)
    log_and_print(f"Final destination directories ready.\n", log_list=log_messages)

    # 3. Iterate, filter, resize, and copy
    total_copied_count = 0
    total_skipped_count = 0

    # We iterate through the OVERLAYS in TRUE_LEAVES_FILTERED as the primary source of component names.
    source_overlay_dir = TRUE_LEAVES_FILTERED_ROOT / "OVERLAYS"
    if not source_overlay_dir.exists():
        log_and_print(f"ERROR: Source OVERLAYS directory '{source_overlay_dir}' not found. Cannot copy files. Exiting.", is_error=True, log_list=log_messages)
        with open(LOG_FILE, 'w') as f: f.writelines(line + '\n' for line in log_messages)
        return

    log_and_print(f"Processing and copying files to '{FINAL_ALIGNED_LEAVES_ROOT}'...", log_list=log_messages)
    # Ensure iterdir() is converted to a list for tqdm to get total count
    for overlay_file_path in tqdm(list(source_overlay_dir.iterdir()), desc="Copying filtered and aligned leaves"):
        if not overlay_file_path.is_file() or overlay_file_path.suffix != '.png':
            continue

        comp_name = extract_component_name(overlay_file_path.name)

        # Apply exclusions
        if comp_name in bad_component_names:
            log_and_print(f"  Skipped '{comp_name}': Found in bad_leaves.", log_list=log_messages)
            total_skipped_count += 1
            continue
        if comp_name.startswith("CROSSES_"):
            log_and_print(f"  Skipped '{comp_name}': Belongs to CROSSES_ dataset.", log_list=log_messages)
            total_skipped_count += 1
            continue

        # Define names and paths for the three file types for the current component
        mask_file_name = comp_name + "_mask.png"
        rgb_crop_file_name = comp_name + "_rgb_crop.png"
        overlay_file_name = comp_name + "_overlay.png" # This is the current overlay_file_path.name

        source_mask_path = TRUE_LEAVES_FILTERED_ROOT / "MASKS" / mask_file_name
        source_rgb_crop_path = TRUE_LEAVES_FILTERED_ROOT / "RGB_CROPS" / rgb_crop_file_name
        # The overlay_file_path variable already holds the correct path for the source overlay

        dest_mask_path = FINAL_MASKS_DIR / mask_file_name
        dest_rgb_crop_path = FINAL_RGB_CROPS_DIR / rgb_crop_file_name
        dest_overlay_path = FINAL_OVERLAYS_DIR / overlay_file_name

        # Collect paths to process and check their existence
        current_component_paths_to_process = [
            (source_mask_path, dest_mask_path),
            (source_rgb_crop_path, dest_rgb_crop_path),
            (overlay_file_path, dest_overlay_path)
        ]

        all_source_files_present = True
        for src_path, _ in current_component_paths_to_process:
            if not src_path.exists():
                log_and_print(f"  WARNING: Missing source file for {comp_name}: {src_path.name}. Skipping this component.", is_error=True, log_list=log_messages)
                all_source_files_present = False
                break
            # Check if destination already exists (to prevent re-copying if script is run multiple times)
            if _.exists():
                log_and_print(f"  INFO: Destination file '{_.name}' already exists for '{comp_name}'. Skipping copy for this component to avoid duplicates.", log_list=log_messages)
                all_source_files_present = False # If any dest exists, assume component was copied
                break

        if not all_source_files_present:
            total_skipped_count += 1
            continue

        # Process and copy each file type
        try:
            for src_path, dest_path in current_component_paths_to_process:
                with Image.open(src_path) as img:
                    # Apply resize and pad
                    processed_img = resize_and_pad_image(img, TARGET_IMAGE_DIM)
                    processed_img.save(dest_path)
            total_copied_count += 1
        except Exception as e:
            log_and_print(f"  ERROR processing and copying {comp_name}: {e}. Skipping.", is_error=True, log_list=log_messages)
            total_skipped_count += 1


    log_and_print(f"\n--- Process Complete ---", log_list=log_messages)
    log_and_print(f"Total components copied to '{FINAL_ALIGNED_LEAVES_ROOT}': {total_copied_count}", log_list=log_messages)
    log_and_print(f"Total components skipped (bad, CROSSES_, or errors): {total_skipped_count}", log_list=log_messages)
    log_and_print(f"Detailed log saved to: {LOG_FILE}", log_list=log_messages)

    # Save the full log to file
    with open(LOG_FILE, 'w') as f:
        f.writelines(line + '\n' for line in log_messages)

if __name__ == "__main__":
    main()