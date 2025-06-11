# generate_7channel_inputs.py

# === IMPORTS ===
import os
import numpy as np
from PIL import Image
from skimage import color, filters, exposure, img_as_float
from tqdm import tqdm

# === CONFIGURATION ===
# Input directory for processed RGB images
PROCESSED_RGB_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/RGB_IMAGES"

# Output directory for the 7-channel NumPy arrays
SEVEN_CHANNEL_DATA_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/7CHANNEL_INPUTS"

# Ensure output directory exists
os.makedirs(SEVEN_CHANNEL_DATA_DIR, exist_ok=True)

# Filter parameters (these will be optimized later)
# For now, using range(1,4) as in your original script
SATO_SIGMAS = range(1, 4)
MEIJERING_SIGMAS = range(1, 4)
FRANGI_SIGMAS = range(1, 4)
HESSIAN_SIGMAS = range(1, 4)

# === HELPERS ===
def enhance_contrast(arr, percentile=99.9):
    """
    Clips values above the given percentile and rescales to 0–1.
    Uses a higher percentile (99.9) for potentially sparse strong responses
    like veins, to avoid clipping too much detail.
    """
    vmax = np.percentile(arr, percentile)
    if vmax == 0: # Avoid division by zero if all values are zero
        return np.zeros_like(arr)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled

# === MAIN PROCESSING ===
# Get a list of all processed RGB image filenames
image_files = sorted([f for f in os.listdir(PROCESSED_RGB_DIR) if f.lower().endswith(".png")])

print(f"Generating 7-channel NumPy inputs for {len(image_files)} images...")

for fname in tqdm(image_files, desc="Applying ridge filters and stacking channels"):
    base_name = os.path.splitext(fname)[0] # e.g., 'sample_id_1'
    rgb_path = os.path.join(PROCESSED_RGB_DIR, fname)
    output_npy_path = os.path.join(SEVEN_CHANNEL_DATA_DIR, f"{base_name}.npy")

    # Skip if .npy file already exists
    if os.path.exists(output_npy_path):
        # print(f"  7-channel input for {base_name}.npy already exists, skipping.")
        continue

    # Validate existence of RGB image
    if not os.path.exists(rgb_path):
        print(f"⚠️ Warning: RGB image '{rgb_path}' not found, skipping 7-channel input generation for {base_name}.")
        continue

    try:
        # Load image and convert to float (required by scikit-image filters)
        image_rgb_pil = Image.open(rgb_path).convert("RGB")
        image_rgb_float = np.array(image_rgb_pil).astype(np.float32) / 255.0 # Normalize RGB to 0-1
        
        # Convert to grayscale for ridge filters
        gray_image = color.rgb2gray(image_rgb_float)

        # Apply ridge filters
        # Note: black_ridges=False for Sato/Meijering/Frangi highlights bright ridges (veins in dark image)
        # black_ridges=True for Hessian highlights dark ridges (veins usually darker)
        sato_raw = filters.sato(gray_image, sigmas=SATO_SIGMAS, black_ridges=False)
        meijering_raw = filters.meijering(gray_image, sigmas=MEIJERING_SIGMAS, black_ridges=False)
        frangi_raw = filters.frangi(gray_image, sigmas=FRANGI_SIGMAS, black_ridges=False)
        hessian_raw = filters.hessian(gray_image, sigmas=HESSIAN_SIGMAS, black_ridges=True)

        # Enhance contrast and ensure float32 (0-1) range for all filter outputs
        # This will be used as input to CNN, so 0-1 float is appropriate
        sato_processed = enhance_contrast(sato_raw).astype(np.float32)
        meijering_processed = enhance_contrast(meijering_raw).astype(np.float32)
        frangi_processed = enhance_contrast(frangi_raw).astype(np.float32)
        hessian_processed = enhance_contrast(hessian_raw).astype(np.float32)

        # Stack channels: RGB (3) + Sato (1) + Meijering (1) + Frangi (1) + Hessian (1) = 7 channels
        # Reshape single-channel filter outputs to (H, W, 1) before stacking
        sato_channel = np.expand_dims(sato_processed, axis=-1)
        meijering_channel = np.expand_dims(meijering_processed, axis=-1)
        frangi_channel = np.expand_dims(frangi_processed, axis=-1)
        hessian_channel = np.expand_dims(hessian_processed, axis=-1)

        # Combine all channels. Order matters for consistency!
        # Final shape will be (Height, Width, 7)
        seven_channel_input = np.concatenate([
            image_rgb_float,
            sato_channel,
            meijering_channel,
            frangi_channel,
            hessian_channel
        ], axis=-1) # Concatenate along the last dimension (channels)

        # Save the combined array
        np.save(output_npy_path, seven_channel_input)

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

print(f"\n✅ All 7-channel NumPy inputs generated and saved to '{SEVEN_CHANNEL_DATA_DIR}'.")