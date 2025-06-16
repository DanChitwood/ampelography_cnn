import os
import numpy as np
from PIL import Image, ImageDraw
import json
import random
from tqdm import tqdm
import re

# --- CONFIGURATION ---
BASE_DIR = os.getcwd() # Assumes script is run from the directory containing foreground_library and background_library

# Input directories
FOREGROUND_LIBRARY_DIR = os.path.join(BASE_DIR, "foreground_library")
BACKGROUND_LIBRARY_DIR = os.path.join(BASE_DIR, "background_library")

# Output directory for synthetic images
OUTPUT_SYNTHETIC_DIR = os.path.join(BASE_DIR, "synthetic_dataset")
os.makedirs(OUTPUT_SYNTHETIC_DIR, exist_ok=True)

# Target dimensions for all synthetic images (and preprocessed backgrounds)
TARGET_WIDTH = 2048
TARGET_HEIGHT = 2040
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# Background image extensions
BACKGROUND_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG", ".JPEG", ".TIF", ".TIFF", ".PNG"]

# Number of synthetic images to generate
NUM_SYNTHETIC_IMAGES = 300 # You can adjust this as needed

# Number of leaves to place per synthetic image
MIN_LEAVES_PER_IMAGE = 1
MAX_LEAVES_PER_IMAGE = 5

# Maximum attempts to place a leaf without overlap
MAX_PLACEMENT_ATTEMPTS = 50

# --- HELPER FUNCTIONS ---

def find_image_path(base_dir, image_filename_no_ext):
    """Searches for an image file with various extensions."""
    for ext in BACKGROUND_IMAGE_EXTENSIONS:
        full_path = os.path.join(base_dir, image_filename_no_ext + ext)
        if os.path.exists(full_path):
            return full_path
        full_path_lower = os.path.join(base_dir, image_filename_no_ext + ext.lower())
        if os.path.exists(full_path_lower): # Check for lowercase extension too
            return full_path_lower
    return None

def rotate_to_wide(image_pil):
    """Rotates an image so its width is greater than its height."""
    width, height = image_pil.size
    if height > width:
        image_pil = image_pil.transpose(Image.Transpose.ROTATE_270)
    return image_pil

def rescale_and_pad_image(image_pil, target_size, fill_color=(255, 255, 255)):
    """
    Rescales an image to fit within target_size while maintaining aspect ratio,
    then pads with a specified fill_color to reach target_size.
    Returns the padded image, and the bounding box (min_x, min_y, max_x, max_y)
    of the *actual content* within the padded image.
    """
    original_width, original_height = image_pil.size
    target_width, target_height = target_size
    
    if original_width == 0 or original_height == 0:
        return Image.new("RGB", target_size, fill_color), (0, 0, target_width, target_height) # Full padding

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
    
    # Return the padded image and the bounding box of the original content
    return padded_img, (paste_x, paste_y, paste_x + new_width, paste_y + new_height)

def get_foreground_assets():
    """Loads all foreground assets (RGB, masks, metadata) from the foreground_library."""
    foreground_assets = []
    for dataset_name in os.listdir(FOREGROUND_LIBRARY_DIR):
        dataset_path = os.path.join(FOREGROUND_LIBRARY_DIR, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        sample_ids = set()
        for f in os.listdir(dataset_path):
            # Extract base_fid more robustly by looking for patterns like _foreground, _blade_mask, etc.
            match = re.match(r"(.+?)_(foreground|blade_mask|vein_mask|metadata)\.json|\.png", f)
            if match:
                sample_id = match.group(1)
                sample_ids.add(sample_id)
            else:
                # Fallback for unexpected naming, but previous regex should cover most
                if '_' in f:
                    parts = f.split('_')
                    if len(parts) > 1 and (parts[-1].startswith('blade_mask') or \
                                           parts[-1].startswith('vein_mask') or \
                                           parts[-1].startswith('foreground') or \
                                           parts[-1].startswith('metadata')):
                        sample_id = '_'.join(parts[:-1])
                        sample_ids.add(sample_id)


        for base_fid in sample_ids:
            try:
                fg_path = os.path.join(dataset_path, f"{base_fid}_foreground.png")
                blade_mask_path = os.path.join(dataset_path, f"{base_fid}_blade_mask.png")
                vein_mask_path = os.path.join(dataset_path, f"{base_fid}_vein_mask.png")
                metadata_path = os.path.join(dataset_path, f"{base_fid}_metadata.json")

                if not (os.path.exists(fg_path) and os.path.exists(blade_mask_path) and
                        os.path.exists(vein_mask_path) and os.path.exists(metadata_path)):
                    print(f"  WARNING: Incomplete asset set for {base_fid} in {dataset_name}. Skipping.")
                    continue

                fg_img = Image.open(fg_path).convert("RGBA")
                blade_mask = Image.open(blade_mask_path).convert("L")
                vein_mask = Image.open(vein_mask_path).convert("L")
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Ensure required metadata exists
                if not all(k in metadata for k in ["cropped_width", "cropped_height", 
                                                    "transformed_blade_coords", "transformed_vein_coords",
                                                    "crop_bbox_min_x", "crop_bbox_min_y"]):
                    print(f"  WARNING: Missing critical metadata for {base_fid}. Skipping.")
                    continue

                foreground_assets.append({
                    "id": base_fid,
                    "dataset": dataset_name,
                    "foreground_img": fg_img,
                    "blade_mask": blade_mask,
                    "vein_mask": vein_mask,
                    "metadata": metadata
                })
            except Exception as e:
                print(f"  ERROR loading asset {base_fid} from {dataset_name}: {e}. Skipping.")
    print(f"Loaded {len(foreground_assets)} foreground assets.")
    return foreground_assets

def get_background_images():
    """Loads all background images."""
    background_images = []
    for f in os.listdir(BACKGROUND_LIBRARY_DIR):
        file_no_ext = os.path.splitext(f)[0]
        full_path = find_image_path(BACKGROUND_LIBRARY_DIR, file_no_ext)
        if full_path:
            try:
                bg_img = Image.open(full_path).convert("RGB")
                background_images.append(bg_img)
            except Exception as e:
                print(f"  ERROR loading background image {f}: {e}. Skipping.")
    print(f"Loaded {len(background_images)} background images.")
    return background_images

def check_overlap(new_bbox, existing_bboxes):
    """Checks if a new bounding box overlaps with any existing bounding boxes."""
    nx1, ny1, nx2, ny2 = new_bbox
    for ex1, ey1, ex2, ey2 in existing_bboxes:
        if not (nx2 <= ex1 or nx1 >= ex2 or ny2 <= ey1 or ny1 >= ey2):
            return True # Overlap detected
    return False

# --- MAIN SYNTHETIC IMAGE GENERATION ---

print("Loading foreground and background assets...")
foreground_assets = get_foreground_assets()
background_images = get_background_images()

if not foreground_assets:
    print("No foreground assets found. Please run 1_generate_foreground_assets.py first and ensure data is present.")
    exit()
if not background_images:
    print("No background images found in background_library. Please add images to proceed.")
    exit()

print("\nStarting synthetic image generation...")

# Determine background images to use (sampling without replacement if possible)
if NUM_SYNTHETIC_IMAGES <= len(background_images):
    selected_background_indices = random.sample(range(len(background_images)), NUM_SYNTHETIC_IMAGES)
else:
    # If not enough unique backgrounds, use with replacement but still randomize order
    print(f"  WARNING: NUM_SYNTHETIC_IMAGES ({NUM_SYNTHETIC_IMAGES}) is greater than available background images ({len(background_images)}). Backgrounds will be sampled with replacement.")
    selected_background_indices = random.choices(range(len(background_images)), k=NUM_SYNTHETIC_IMAGES)

for i in tqdm(range(NUM_SYNTHETIC_IMAGES), desc="Generating Synthetic Images"):
    # 1. Select and preprocess a background image
    background_idx = selected_background_indices[i]
    base_background_img = background_images[background_idx]
    
    # Get the bounding box of the actual image content within the padded background
    processed_background, bg_content_bbox = rescale_and_pad_image(base_background_img.copy(), TARGET_SIZE, fill_color=(255, 255, 255))
    
    synthetic_rgb = processed_background.copy()
    combined_mask = Image.new('L', TARGET_SIZE, 0) 
    placed_bboxes = [] # To keep track of bounding boxes of placed leaves
    
    synthetic_blade_coords_output = []
    synthetic_vein_coords_output = []
    synthetic_geodesic_origins_output = []

    num_leaves_to_place = random.randint(MIN_LEAVES_PER_IMAGE, MAX_LEAVES_PER_IMAGE)
    
    # Select leaves for this image without replacement from the available foreground assets
    if num_leaves_to_place > len(foreground_assets):
        # If trying to place more leaves than available unique assets, use all unique assets
        # and then sample with replacement for the remaining
        leaves_for_this_image = random.sample(foreground_assets, len(foreground_assets))
        remaining_leaves_to_add = num_leaves_to_place - len(foreground_assets)
        if remaining_leaves_to_add > 0:
            leaves_for_this_image.extend(random.choices(foreground_assets, k=remaining_leaves_to_add))
    else:
        leaves_for_this_image = random.sample(foreground_assets, num_leaves_to_place)

    leaf_counter = 0 # To label multiple leaves in output files
    
    # Define the bounding box for placing leaves (the non-padded area of the background)
    bg_min_x, bg_min_y, bg_max_x, bg_max_y = bg_content_bbox

    for selected_asset in leaves_for_this_image:
        fg_img = selected_asset["foreground_img"]
        blade_mask_cropped = selected_asset["blade_mask"]
        vein_mask_cropped = selected_asset["vein_mask"]
        metadata = selected_asset["metadata"]

        cropped_width = metadata["cropped_width"]
        cropped_height = metadata["cropped_height"]
        
        # Get original transformed coords and geodesic origin from metadata
        original_transformed_blade_coords = np.array(metadata["transformed_blade_coords"])
        original_transformed_vein_coords = np.array(metadata["transformed_vein_coords"])
        original_geodesic_origin = np.array(metadata["geodesic_origin_coords"])
        
        crop_bbox_min_x = metadata["crop_bbox_min_x"]
        crop_bbox_min_y = metadata["crop_bbox_min_y"]

        placed = False
        for attempt in range(MAX_PLACEMENT_ATTEMPTS):
            # Calculate placement range within the background's content area
            # A leaf's top-left corner (paste_x, paste_y) must be within this range:
            # min_allowed_paste_x = bg_min_x
            # max_allowed_paste_x = bg_max_x - cropped_width
            # min_allowed_paste_y = bg_min_y
            # max_allowed_paste_y = bg_max_y - cropped_height

            if cropped_width > (bg_max_x - bg_min_x) or cropped_height > (bg_max_y - bg_min_y):
                # Leaf is larger than the actual background content area, cannot be placed realistically
                # print(f"  DEBUG: Leaf {selected_asset['id']} too large for background content area. Skipping.")
                break # Break from attempts loop

            # Ensure random.randint bounds are valid (max >= min)
            min_paste_x = bg_min_x
            max_paste_x = max(bg_min_x, bg_max_x - cropped_width) # Ensure max is at least min
            
            min_paste_y = bg_min_y
            max_paste_y = max(bg_min_y, bg_max_y - cropped_height) # Ensure max is at least min
            
            # If max_paste_x or max_paste_y are less than min_paste_x or min_paste_y, it means
            # the leaf is too big for the *content area*, but not necessarily the full image.
            # We've already checked if it's too big for the content area above.
            
            if min_paste_x > max_paste_x or min_paste_y > max_paste_y: # This should theoretically be caught by the size check, but as a safeguard
                 # print(f"  DEBUG: Placement range invalid for leaf {selected_asset['id']}. Skipping.")
                 break

            paste_x = random.randint(min_paste_x, max_paste_x)
            paste_y = random.randint(min_paste_y, max_paste_y)

            # Calculate the bounding box of the leaf if placed at (paste_x, paste_y)
            new_leaf_bbox = (paste_x, paste_y, paste_x + cropped_width, paste_y + cropped_height)

            if not check_overlap(new_leaf_bbox, placed_bboxes):
                synthetic_rgb.paste(fg_img, (paste_x, paste_y), fg_img)
                
                combined_mask_np = np.array(combined_mask)
                blade_mask_cropped_np = np.array(blade_mask_cropped)
                vein_mask_cropped_np = np.array(vein_mask_cropped)

                # Place blade pixels (value 1)
                combined_mask_np[paste_y : paste_y + cropped_height, paste_x : paste_x + cropped_width][blade_mask_cropped_np > 0] = 1
                # Place vein pixels (value 2), overwriting blade pixels where veins exist
                combined_mask_np[paste_y : paste_y + cropped_height, paste_x : paste_x + cropped_width][vein_mask_cropped_np > 0] = 2
                
                combined_mask = Image.fromarray(combined_mask_np, mode='L')
                
                placed_bboxes.append(new_leaf_bbox)
                
                offset_x = paste_x - crop_bbox_min_x
                offset_y = paste_y - crop_bbox_min_y

                current_leaf_blade_coords = (original_transformed_blade_coords + np.array([offset_x, offset_y])).tolist()
                current_leaf_vein_coords = (original_transformed_vein_coords + np.array([offset_x, offset_y])).tolist()
                current_geodesic_origin = (original_geodesic_origin + np.array([offset_x, offset_y])).tolist()

                synthetic_blade_coords_output.append({"leaf_idx": leaf_counter, "coords": current_leaf_blade_coords})
                synthetic_vein_coords_output.append({"leaf_idx": leaf_counter, "coords": current_leaf_vein_coords})
                synthetic_geodesic_origins_output.append({"leaf_idx": leaf_counter, "coords": current_geodesic_origin})
                
                leaf_counter += 1
                placed = True
                break # Leaf placed, move to next leaf
        
        # if not placed:
            # This can happen if the leaf is too big or no non-overlapping spot is found
            # print(f"  WARNING: Could not place leaf {selected_asset['id']} on synthetic image {i} after {MAX_PLACEMENT_ATTEMPTS} attempts.")

    # --- Save Outputs for the current synthetic image ---
    synthetic_id = f"synthetic_{i:05d}"
    output_dir = os.path.join(OUTPUT_SYNTHETIC_DIR, synthetic_id)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save blade and vein trace information
    for blade_data in synthetic_blade_coords_output:
        coords_str = "\n".join([f"{x} {y}" for x, y in blade_data["coords"]])
        with open(os.path.join(output_dir, f"{synthetic_id}_blade_{blade_data['leaf_idx']}.txt"), 'w') as f:
            f.write(coords_str)
    
    for vein_data in synthetic_vein_coords_output:
        coords_str = "\n".join([f"{x} {y}" for x, y in vein_data["coords"]])
        with open(os.path.join(output_dir, f"{synthetic_id}_vein_{vein_data['leaf_idx']}.txt"), 'w') as f:
            f.write(coords_str)
            
    # Also save geodesic origins
    geodesic_origins_data = [{"leaf_idx": go['leaf_idx'], "x": go['coords'][0], "y": go['coords'][1]} for go in synthetic_geodesic_origins_output]
    with open(os.path.join(output_dir, f"{synthetic_id}_geodesic_origins.json"), 'w') as f:
        json.dump(geodesic_origins_data, f, indent=4)

    # 2) Save combined mask (value 0: background, 1: blade, 2: vein)
    combined_mask.save(os.path.join(output_dir, f"{synthetic_id}_combined_mask.png"))

    # 3) Save synthetic RGB image
    synthetic_rgb.save(os.path.join(output_dir, f"{synthetic_id}_rgb.png"))

    # 4) Save overlay image (RGB with mask overlay)
    overlay_img = synthetic_rgb.convert("RGBA")
    draw_overlay = ImageDraw.Draw(overlay_img)
    
    combined_mask_np = np.array(combined_mask)
    
    # Overlay blade pixels (e.g., green)
    blade_pixels_y, blade_pixels_x = np.where(combined_mask_np == 1)
    for px, py in zip(blade_pixels_x, blade_pixels_y):
        overlay_img.putpixel((px, py), (0, 255, 0, 128)) # Green with 50% transparency

    # Overlay vein pixels (e.g., red)
    vein_pixels_y, vein_pixels_x = np.where(combined_mask_np == 2)
    for px, py in zip(vein_pixels_x, vein_pixels_y):
        overlay_img.putpixel((px, py), (255, 0, 0, 192)) # Red with higher transparency
        
    # Overlay geodesic origin points (e.g., blue circles)
    for origin_data in synthetic_geodesic_origins_output:
        ox, oy = origin_data["coords"]
        draw_overlay.ellipse((ox-5, oy-5, ox+5, oy+5), fill=(0, 0, 255, 255)) # Blue, opaque

    overlay_img.save(os.path.join(output_dir, f"{synthetic_id}_overlay.png"))

print("\n--- Synthetic image generation complete! ---")
print(f"Generated {NUM_SYNTHETIC_IMAGES} synthetic images in '{OUTPUT_SYNTHETIC_DIR}'")