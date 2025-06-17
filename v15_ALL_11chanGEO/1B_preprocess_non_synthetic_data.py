import os
import numpy as np
from PIL import Image, ImageDraw
import json
from tqdm import tqdm
from pathlib import Path
import re
import shutil # Still useful for potential cleanup, but dummy data generation removed

# --- CONFIGURATION ---
BASE_DIR = Path(os.getcwd()) # Assumes script is run from the directory containing non_synthetic_data

# Input directories for your original non-synthetic data
INPUT_NON_SYNTHETIC_ROOT = BASE_DIR / "non_synthetic_data"
INPUT_MULTIPLE_LEAVES_DIR = INPUT_NON_SYNTHETIC_ROOT / "multiple_leaves"
INPUT_SINGLE_LEAF_DIR = INPUT_NON_SYNTHETIC_ROOT / "single_leaf"

# Output directory for processed non-synthetic data, mimicking synthetic_dataset structure
OUTPUT_PROCESSED_NON_SYNTHETIC_DIR = BASE_DIR / "non_synthetic_dataset"
OUTPUT_PROCESSED_NON_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

# Target dimensions for all images (must match your other scripts)
TARGET_WIDTH = 2048
TARGET_HEIGHT = 2040
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# Image extensions to look for
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".JPG", ".JPEG"] # Assuming only these for original data

# --- HELPER FUNCTIONS ---

def find_image_path(base_dir, image_filename_no_ext):
    """Searches for an image file with various extensions."""
    for ext in IMAGE_EXTENSIONS:
        full_path = base_dir / (image_filename_no_ext + ext)
        if full_path.exists():
            return full_path
    return None

def rescale_and_pad_image(image_pil, target_size, fill_color=(255, 255, 255)):
    """
    Rescales an image to fit within target_size while maintaining aspect ratio,
    then pads with a specified fill_color to reach target_size.
    Returns the padded image, the bounding box (paste_x, paste_y, new_width, new_height)
    of the *scaled content* within the padded image, AND the scaling factor applied.
    """
    original_width, original_height = image_pil.size
    target_width, target_height = target_size
    
    if original_width == 0 or original_height == 0:
        return Image.new("RGB", target_size, fill_color), (0, 0, 0, 0), 1.0

    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale_factor = min(scale_w, scale_h)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    scaled_img = image_pil.resize((new_width, new_height), Image.LANCZOS)

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    padded_img = Image.new("RGB", target_size, fill_color)
    padded_img.paste(scaled_img, (paste_x, paste_y))
    
    return padded_img, (paste_x, paste_y, new_width, new_height), scale_factor

def read_coords(path):
    """Reads coordinate data from a text file. Returns empty array if file not found or empty."""
    if not path.exists():
        return np.array([])
    try:
        # Use dtype=float because scaling can result in non-integer values before rounding
        coords = np.loadtxt(path, dtype=np.float32) 
        if coords.ndim == 1: # If only one point, make it 2D (1, 2)
            coords = coords.reshape(1, -1)
        return coords
    except ValueError: # Empty file or malformed
        return np.array([])
    except Exception as e:
        print(f"Error reading coordinates from {path}: {e}")
        return np.array([])

def calculate_geodesic_origin(vein_coords_array):
    """
    Calculates the geodesic origin as the average of the first and last
    coordinates of the vein trace.
    Args:
        vein_coords_array (np.array): Nx2 array of (x, y) coordinates for the vein trace.
                                      These should already be scaled and offset.
    Returns:
        tuple: (x, y) coordinates of the geodesic origin, or None if not enough points.
    """
    if vein_coords_array.shape[0] < 2:
        return None # Need at least two points to define start and end
    
    first_point = vein_coords_array[0]
    last_point = vein_coords_array[-1]
    
    origin_x = int(round((first_point[0] + last_point[0]) / 2))
    origin_y = int(round((first_point[1] + last_point[1]) / 2))
    
    return (origin_x, origin_y)

# --- MAIN PROCESSING FUNCTION ---

def process_non_synthetic_data():
    """
    Processes raw non-synthetic images and their traces into the
    'synthetic_dataset' mimic format, including standardizing image size,
    extracting individual leaf coordinates, and calculating geodesic origins.
    """
    print(f"--- Starting Non-Synthetic Data Preprocessing ---")
    print(f"Input root: {INPUT_NON_SYNTHETIC_ROOT}")
    print(f"Output root: {OUTPUT_PROCESSED_NON_SYNTHETIC_DIR}")

    global_image_counter = 0 # To assign unique non_synthetic_XXXXX IDs

    # Process 'multiple_leaves' folder first
    print(f"\nProcessing images from '{INPUT_MULTIPLE_LEAVES_DIR}'...")
    if not INPUT_MULTIPLE_LEAVES_DIR.exists():
        print(f"  WARNING: Input directory not found: {INPUT_MULTIPLE_LEAVES_DIR}. Skipping.")
    else:
        for image_file in tqdm(sorted(list(INPUT_MULTIPLE_LEAVES_DIR.glob("*.jpg")) + list(INPUT_MULTIPLE_LEAVES_DIR.glob("*.jpeg"))), desc="Multiple Leaves"):
            process_single_image_set(image_file, INPUT_MULTIPLE_LEAVES_DIR, global_image_counter)
            global_image_counter += 1

    # Process 'single_leaf' folder
    print(f"\nProcessing images from '{INPUT_SINGLE_LEAF_DIR}'...")
    if not INPUT_SINGLE_LEAF_DIR.exists():
        print(f"  WARNING: Input directory not found: {INPUT_SINGLE_LEAF_DIR}. Skipping.")
    else:
        for image_file in tqdm(sorted(list(INPUT_SINGLE_LEAF_DIR.glob("*.jpg")) + list(INPUT_SINGLE_LEAF_DIR.glob("*.jpeg"))), desc="Single Leaf"):
            process_single_image_set(image_file, INPUT_SINGLE_LEAF_DIR, global_image_counter, is_single_leaf=True)
            global_image_counter += 1

    print(f"\n--- Non-Synthetic Data Preprocessing Complete! ---")
    print(f"Processed {global_image_counter} image sets.")

def process_single_image_set(image_path: Path, current_input_dir: Path, image_id_counter: int, is_single_leaf=False):
    """
    Processes a single image and its associated blade/vein files.
    """
    file_stem = image_path.stem
    output_synthetic_id = f"non_synthetic_{image_id_counter:05d}"
    output_dir = OUTPUT_PROCESSED_NON_SYNTHETIC_DIR / output_synthetic_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Processing {image_path.name} -> {output_synthetic_id}/")

    try:
        # --- 1. Load and Standardize RGB Image ---
        original_img_pil = Image.open(image_path).convert("RGB")
        original_width, original_height = original_img_pil.size # Get original dimensions for scaling factor
        
        # Get processed image, content bbox, and the scaling factor
        processed_rgb_pil, content_bbox, scale_factor = rescale_and_pad_image(original_img_pil, TARGET_SIZE)
        processed_rgb_pil.save(output_dir / f"{output_synthetic_id}_rgb.png")
        print(f"    Saved standardized RGB: {output_synthetic_id}_rgb.png")

        # Get the offset from original image coordinates to padded image coordinates
        pad_x_offset = content_bbox[0]
        pad_y_offset = content_bbox[1]
        
        # --- 2. Gather all leaf indices and process traces ---
        blade_trace_files = []
        vein_trace_files = []

        if is_single_leaf:
            blade_trace_files.append(current_input_dir / f"{file_stem}_blade.txt")
            vein_trace_files.append(current_input_dir / f"{file_stem}_veins.txt")
        else:
            # For multiple leaves, find all numbered _bladeX.txt and _veinsX.txt
            for f in current_input_dir.iterdir():
                if f.stem.startswith(file_stem) and f.suffix == '.txt':
                    blade_match = re.match(rf"{re.escape(file_stem)}_blade(\d+)\.txt", f.name)
                    vein_match = re.match(rf"{re.escape(file_stem)}_veins(\d+)\.txt", f.name)
                    if blade_match:
                        blade_trace_files.append(f)
                    elif vein_match:
                        vein_trace_files.append(f)
            # Sort by leaf index to ensure consistent processing order
            blade_trace_files.sort(key=lambda p: int(re.search(r'(\d+)\.txt', p.name).group(1)) if re.search(r'(\d+)\.txt', p.name) else 0)
            vein_trace_files.sort(key=lambda p: int(re.search(r'(\d+)\.txt', p.name).group(1)) if re.search(r'(\d+)\.txt', p.name) else 0)

        synthetic_blade_coords_output = []
        synthetic_vein_coords_output = []
        synthetic_geodesic_origins_output = []
        
        processed_leaf_counter = 0

        # Process each blade/vein pair
        leaf_file_map = {}
        for f in blade_trace_files + vein_trace_files:
            if is_single_leaf:
                leaf_idx = 0 # Assign 0 for single leaf cases for consistency
            else:
                try:
                    leaf_idx = int(re.search(r'(\d+)\.txt', f.name).group(1))
                except AttributeError:
                    print(f"      Could not parse leaf_idx from {f.name}. Skipping this trace.")
                    continue
            
            if leaf_idx not in leaf_file_map:
                leaf_file_map[leaf_idx] = {"blade": None, "vein": None}
            
            if "blade" in f.name:
                leaf_file_map[leaf_idx]["blade"] = f
            elif "veins" in f.name:
                leaf_file_map[leaf_idx]["vein"] = f

        sorted_leaf_indices = sorted(leaf_file_map.keys())

        for leaf_idx in sorted_leaf_indices:
            blade_path = leaf_file_map[leaf_idx].get("blade")
            vein_path = leaf_file_map[leaf_idx].get("vein")

            if not blade_path and not vein_path:
                print(f"      Skipping leaf {leaf_idx} for {file_stem}: No blade or vein file found.")
                continue

            # Read original coordinates
            original_blade_coords = read_coords(blade_path) if blade_path else np.array([])
            original_vein_coords = read_coords(vein_path) if vein_path else np.array([])

            # Apply scaling THEN padding offset to coordinates
            # Coordinates are (x, y)
            scaled_blade_coords = (original_blade_coords * scale_factor) if original_blade_coords.size > 0 else np.array([])
            scaled_vein_coords = (original_vein_coords * scale_factor) if original_vein_coords.size > 0 else np.array([])

            # Apply padding offset (only if coords exist after scaling)
            current_leaf_blade_coords = (scaled_blade_coords + np.array([pad_x_offset, pad_y_offset])).tolist() if scaled_blade_coords.size > 0 else []
            current_leaf_vein_coords = (scaled_vein_coords + np.array([pad_x_offset, pad_y_offset])).tolist() if scaled_vein_coords.size > 0 else []

            # Calculate geodesic origin using the scaled and offset vein coordinates
            geodesic_origin_xy = calculate_geodesic_origin(scaled_vein_coords + np.array([pad_x_offset, pad_y_offset])) if scaled_vein_coords.size > 0 else None
            
            # Save individual leaf coordinates (round to int for pixel coordinates)
            if current_leaf_blade_coords:
                coords_str = "\n".join([f"{int(round(x))} {int(round(y))}" for x, y in current_leaf_blade_coords])
                with open(output_dir / f"{output_synthetic_id}_blade_{processed_leaf_counter}.txt", 'w') as f:
                    f.write(coords_str)
            
            if current_leaf_vein_coords:
                coords_str = "\n".join([f"{int(round(x))} {int(round(y))}" for x, y in current_leaf_vein_coords])
                with open(output_dir / f"{output_synthetic_id}_vein_{processed_leaf_counter}.txt", 'w') as f:
                    f.write(coords_str)
            
            # Collect geodesic origins for the JSON file
            if geodesic_origin_xy:
                synthetic_geodesic_origins_output.append({"leaf_idx": processed_leaf_counter, "x": geodesic_origin_xy[0], "y": geodesic_origin_xy[1]})
            
            processed_leaf_counter += 1
        
        # Save geodesic origins for the entire image (all leaves)
        if synthetic_geodesic_origins_output:
            with open(output_dir / f"{output_synthetic_id}_geodesic_origins.json", 'w') as f:
                json.dump(synthetic_geodesic_origins_output, f, indent=4)
            print(f"    Saved geodesic origins: {output_synthetic_id}_geodesic_origins.json")
        else:
            print(f"    No valid geodesic origins found for {output_synthetic_id}.")

        # --- Generate Combined Mask (0=bg, 1=blade, 2=vein) for visualization/completeness ---
        combined_mask = Image.new('L', TARGET_SIZE, 0) # 'L' mode for 8-bit grayscale
        
        output_blade_files_for_image = sorted(list(output_dir.glob(f"{output_synthetic_id}_blade_*.txt")))
        output_vein_files_for_image = sorted(list(output_dir.glob(f"{output_synthetic_id}_vein_*.txt")))

        temp_blade_mask_np = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8) # H, W
        temp_vein_mask_np = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8) # H, W

        # Draw blades first
        for blade_fpath in output_blade_files_for_image:
            coords = read_coords(blade_fpath)
            if coords.size > 0:
                mask_for_leaf = Image.new('L', TARGET_SIZE, 0)
                draw_leaf_mask = ImageDraw.Draw(mask_for_leaf)
                # Ensure coordinates are integer for drawing
                draw_leaf_mask.polygon([tuple(map(int, map(round, p))) for p in coords], fill=1)
                temp_blade_mask_np = np.logical_or(temp_blade_mask_np, np.array(mask_for_leaf)).astype(np.uint8)
        
        # Draw veins second, overwriting blade areas
        for vein_fpath in output_vein_files_for_image:
            coords = read_coords(vein_fpath)
            if coords.size > 0:
                mask_for_leaf = Image.new('L', TARGET_SIZE, 0)
                draw_leaf_mask = ImageDraw.Draw(mask_for_leaf)
                # Ensure coordinates are integer for drawing
                draw_leaf_mask.polygon([tuple(map(int, map(round, p))) for p in coords], fill=1)
                temp_vein_mask_np = np.logical_or(temp_vein_mask_np, np.array(mask_for_leaf)).astype(np.uint8)

        # Combine: blade=1, vein=2. Vein pixels overwrite blade where they overlap.
        combined_mask_np = np.zeros(TARGET_SIZE[::-1], dtype=np.uint8)
        combined_mask_np[temp_blade_mask_np == 1] = 1 # Set blade areas to 1
        combined_mask_np[temp_vein_mask_np == 1] = 2 # Set vein areas to 2 (overwriting 1 if overlap)

        combined_mask = Image.fromarray(combined_mask_np, mode='L')
        combined_mask.save(output_dir / f"{output_synthetic_id}_combined_mask.png")
        print(f"    Saved combined mask: {output_synthetic_id}_combined_mask.png")

        # --- Save Overlay Image for visual check (mimicking 1_generate_synthetic_images.py output) ---
        overlay_img = processed_rgb_pil.convert("RGBA")
        draw_overlay = ImageDraw.Draw(overlay_img)
        
        # Blade overlay (e.g., green, semi-transparent)
        blade_pixels_y, blade_pixels_x = np.where(combined_mask_np == 1)
        for px, py in zip(blade_pixels_x, blade_pixels_y):
            overlay_img.putpixel((px, py), (0, 255, 0, 128)) # Green with 50% transparency

        # Vein overlay (e.g., red, more opaque)
        vein_pixels_y, vein_pixels_x = np.where(combined_mask_np == 2)
        for px, py in zip(vein_pixels_x, vein_pixels_y):
            overlay_img.putpixel((px, py), (255, 0, 0, 192)) # Red with higher transparency
        
        # Overlay geodesic origin points (e.g., blue circles)
        for origin_data in synthetic_geodesic_origins_output:
            ox, oy = origin_data["x"], origin_data["y"]
            draw_overlay.ellipse((ox-5, oy-5, ox+5, oy+5), fill=(0, 0, 255, 255)) # Blue, opaque

        overlay_img.save(output_dir / f"{output_synthetic_id}_overlay.png")
        print(f"    Saved overlay image: {output_synthetic_id}_overlay.png")

    except FileNotFoundError as e:
        print(f"  Error processing {image_path.name}: File not found - {e}. Skipping.")
    except Exception as e:
        print(f"  An unexpected error occurred while processing {image_path.name}: {e}. Skipping.")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # --- IMPORTANT: The dummy data generation has been removed from here. ---
    # Ensure your 'non_synthetic_data' directory with 'multiple_leaves' and 'single_leaf'
    # subfolders, containing your actual image and trace files, is set up correctly.

    process_non_synthetic_data()

    # Optional: You can uncomment the cleanup lines if you want to remove the output
    # folder after testing, but be careful not to delete your actual data!
    # print("\nOptional: Cleaning up output dummy data...")
    # shutil.rmtree(OUTPUT_PROCESSED_NON_SYNTHETIC_DIR, ignore_errors=True)
    # print("Cleanup complete.")