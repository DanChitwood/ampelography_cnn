# 3_generate_11channel_inputs.py

# === IMPORTS ===
import os
import numpy as np
from PIL import Image
from skimage import color, filters, exposure, img_as_float
from tqdm import tqdm

# === CONFIGURATION ===
# Input directory for processed RGB images (from 0_main_data_preprocessing.py)
PROCESSED_RGB_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/RGB_IMAGES"

# Output directory for the 11-channel NumPy arrays
ELEVEN_CHANNEL_DATA_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/11CHANNEL_INPUTS"

# Ensure output directory exists
os.makedirs(ELEVEN_CHANNEL_DATA_DIR, exist_ok=True)

# Filter parameters (using more detailed sigmas for potentially better results)
# You can adjust these based on empirical observation or prior knowledge
# These are representative ranges; feel free to fine-tune later if needed.
SIGMA_RANGE_VEINS = range(1, 5) # Smaller sigmas for finer veins
SIGMA_RANGE_LARGER_FEATURES = range(1, 7) # Larger sigmas for broader features like vein thickness variations

# === HELPERS ===
def enhance_contrast(arr, percentile=99.9):
    """
    Clips values above the given percentile and rescales to 0–1.
    Uses a higher percentile for potentially sparse strong responses
    like veins, to avoid clipping too much detail.
    """
    if arr.size == 0:
        return np.array([]) # Handle empty array case
    vmax = np.percentile(arr, percentile)
    if vmax == 0: # Avoid division by zero if all values are zero
        return np.zeros_like(arr)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled

# === MAIN PROCESSING ===
# Get a list of all processed RGB image filenames
image_files = sorted([f for f in os.listdir(PROCESSED_RGB_DIR) if f.lower().endswith(".png")])

print(f"Generating 11-channel NumPy inputs for {len(image_files)} images...")

for fname in tqdm(image_files, desc="Applying filters and stacking channels"):
    base_name = os.path.splitext(fname)[0] # e.g., 'sample_id_1'
    rgb_path = os.path.join(PROCESSED_RGB_DIR, fname)
    output_npy_path = os.path.join(ELEVEN_CHANNEL_DATA_DIR, f"{base_name}.npy")

    # Skip if .npy file already exists
    if os.path.exists(output_npy_path):
        continue

    # Validate existence of RGB image
    if not os.path.exists(rgb_path):
        print(f"⚠️ Warning: RGB image '{rgb_path}' not found, skipping 11-channel input generation for {base_name}.")
        continue

    try:
        # Load image and convert to float (required by scikit-image)
        image_rgb_pil = Image.open(rgb_path).convert("RGB")
        image_rgb_float = np.array(image_rgb_pil).astype(np.float32) / 255.0 # Normalize RGB to 0-1
        
        # 1. RGB Channels (3)
        # Already image_rgb_float
        
        # 2. Grayscale Channel (1)
        gray_image = color.rgb2gray(image_rgb_float)
        gray_channel = np.expand_dims(gray_image, axis=-1).astype(np.float32)

        # 3. Lab Color Space Channels (3)
        # Convert RGB (0-1 float) to Lab
        image_lab_float = color.rgb2lab(image_rgb_float)
        
        # Normalize L channel to 0-1 (typically 0-100)
        l_channel = np.expand_dims(image_lab_float[:,:,0] / 100.0, axis=-1).astype(np.float32)
        
        # Normalize a and b channels to 0-1 (typically -128 to 127, scale to fit)
        # Assuming a range of -128 to 127 for a* and b*, so add 128 and divide by 255
        a_channel = np.expand_dims((image_lab_float[:,:,1] + 128) / 255.0, axis=-1).astype(np.float32)
        b_channel = np.expand_dims((image_lab_float[:,:,2] + 128) / 255.0, axis=-1).astype(np.float32)

        # 4. Ridge Filters (4 channels)
        # Apply ridge filters. Note: black_ridges=False for Sato/Meijering/Frangi for bright ridges
        # black_ridges=True for Hessian for dark ridges. This depends on vein appearance.
        # Use more appropriate sigma ranges
        sato_raw = filters.sato(gray_image, sigmas=SIGMA_RANGE_VEINS, black_ridges=False)
        meijering_raw = filters.meijering(gray_image, sigmas=SIGMA_RANGE_VEINS, black_ridges=False)
        frangi_raw = filters.frangi(gray_image, sigmas=SIGMA_RANGE_VEINS, black_ridges=False)
        hessian_raw = filters.hessian(gray_image, sigmas=SIGMA_RANGE_LARGER_FEATURES, black_ridges=True)

        # Enhance contrast and ensure float32 (0-1) range for all filter outputs
        sato_processed = enhance_contrast(sato_raw).astype(np.float32)
        meijering_processed = enhance_contrast(meijering_raw).astype(np.float32)
        frangi_processed = enhance_contrast(frangi_raw).astype(np.float32)
        hessian_processed = enhance_contrast(hessian_raw).astype(np.float32)

        # Reshape single-channel filter outputs to (H, W, 1) before stacking
        sato_channel = np.expand_dims(sato_processed, axis=-1)
        meijering_channel = np.expand_dims(meijering_processed, axis=-1)
        frangi_channel = np.expand_dims(frangi_processed, axis=-1)
        hessian_channel = np.expand_dims(hessian_processed, axis=-1)

        # Combine all channels: RGB (3) + Grayscale (1) + Lab (3) + Sato (1) + Meijering (1) + Frangi (1) + Hessian (1) = 11 channels
        # Order matters for consistency when loading into the model!
        eleven_channel_input = np.concatenate([
            image_rgb_float,
            gray_channel,
            l_channel,
            a_channel,
            b_channel,
            sato_channel,
            meijering_channel,
            frangi_channel,
            hessian_channel
        ], axis=-1) # Concatenate along the last dimension (channels)

        # Save the combined array
        np.save(output_npy_path, eleven_channel_input)

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

print(f"\n✅ All 11-channel NumPy inputs generated and saved to '{ELEVEN_CHANNEL_DATA_DIR}'.")