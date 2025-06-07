import os
import numpy as np
from PIL import Image, ImageOps
from skimage import filters, color, img_as_ubyte
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_FOLDER = "INPUTS"
TARGET_SIZE = (2048, 1468)  # Default target size (width, height)
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

# Output directories for filtered images
FILTERS = ["rgb", "sato", "meijering", "frangi", "hessian"]
OUTPUT_ROOT = "."

# === Contrast Enhancement Function ===
def enhance_contrast(arr, percentile=99):
    """Clip values above the given percentile and rescale to 0–1."""
    vmax = np.percentile(arr, percentile)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax  # Normalize to 0–1
    return arr_rescaled

# === Output Paths ===
def create_output_paths(input_path):
    rel_path = os.path.relpath(input_path, start=INPUT_FOLDER)
    rel_dir = os.path.dirname(rel_path)
    output_paths = {}
    for filt in FILTERS:
        out_dir = os.path.join(OUTPUT_ROOT, f"images_{filt}", rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        output_paths[filt] = os.path.join(out_dir, os.path.basename(rel_path))
    return output_paths

# === Image Discovery ===
def find_images(root):
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(dirpath, f))
    return image_paths

# === Rotate & Pad ===
def rotate_to_wide(img):
    return img if img.width >= img.height else img.transpose(Image.Transpose.ROTATE_90)

def rescale_and_pad_image(img, target_size):
    img = ImageOps.contain(img, target_size)
    padded = Image.new("RGB", target_size, (255, 255, 255))
    paste_pos = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
    padded.paste(img, paste_pos)
    return padded

# === Filter Application ===
def apply_filters_and_save(img_rgb, output_paths):
    gray = color.rgb2gray(np.array(img_rgb))

    # Apply filters and enhance contrast
    results = {
        "rgb": img_rgb,
        "sato": img_as_ubyte(enhance_contrast(filters.sato(gray, sigmas=range(2, 8), black_ridges=False))),
        "meijering": img_as_ubyte(enhance_contrast(filters.meijering(gray, sigmas=range(2, 8), black_ridges=False))),
        "frangi": img_as_ubyte(enhance_contrast(filters.frangi(gray, sigmas=range(5, 40, 10), black_ridges=False))),
        "hessian": img_as_ubyte(enhance_contrast(filters.hessian(gray, sigmas=range(2, 8), black_ridges=True))),
    }

    # Save all outputs
    for key, img in results.items():
        Image.fromarray(np.array(img)).save(output_paths[key])

# === MAIN EXECUTION ===
image_files = find_images(INPUT_FOLDER)
print(f"🔍 Found {len(image_files)} images to process.")

for img_path in tqdm(image_files, desc="Processing images"):
    try:
        img = Image.open(img_path).convert("RGB")
        img = rotate_to_wide(img)
        img_resized = rescale_and_pad_image(img, TARGET_SIZE)
        output_paths = create_output_paths(img_path)
        apply_filters_and_save(img_resized, output_paths)
    except Exception as e:
        print(f"⚠️ Failed to process {img_path}: {e}")

print("✅ All images processed and saved.")

