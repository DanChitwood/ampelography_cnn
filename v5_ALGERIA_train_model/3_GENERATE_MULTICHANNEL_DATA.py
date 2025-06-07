import os
import numpy as np
from PIL import Image
from skimage import color, filters, exposure, img_as_ubyte
from tqdm import tqdm

# === INPUT AND OUTPUT ===
input_dir = "BACKGROUNDS"
output_dirs = {
    "images": "images_rgb",
    "sato": "images_sato",
    "meijering": "images_meijering",
    "frangi": "images_frangi",
    "hessian": "images_hessian"
}

# === SETUP ===
for folder in output_dirs.values():
    os.makedirs(folder, exist_ok=True)

# Get base names from PNG files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
base_names = [os.path.splitext(f)[0] for f in image_files]

def enhance_contrast(arr, percentile=99):
    """Clip values above the given percentile and rescale to 0–1."""
    vmax = np.percentile(arr, percentile)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax  # Normalize to 0–1
    return arr_rescaled

# === PROCESSING ===
for base in tqdm(base_names, desc="Processing ridge filters"):
    img_path = os.path.join(input_dir, base + ".png")

    if not os.path.exists(img_path):
        print(f"⚠️ Skipping {base}: missing image.")
        continue

    # Load image and convert to grayscale
    image = np.array(Image.open(img_path).convert("RGB"))
    gray_image = color.rgb2gray(image)

    # Apply ridge filters
    sato_raw = filters.sato(gray_image, sigmas=range(1, 4), black_ridges=False)
    meijering_raw = filters.meijering(gray_image, sigmas=range(1, 4), black_ridges=False)
    frangi_raw = filters.frangi(gray_image, sigmas=range(1, 4), black_ridges=False)
    hessian_raw = filters.hessian(gray_image, sigmas=range(1, 4), black_ridges=True)

    # Enhance contrast using clipping and rescaling
    sato = img_as_ubyte(enhance_contrast(sato_raw))
    meijering = img_as_ubyte(enhance_contrast(meijering_raw))
    frangi = img_as_ubyte(enhance_contrast(frangi_raw))
    hessian = img_as_ubyte(enhance_contrast(hessian_raw))

    # Save outputs
    filename = base + ".png"
    Image.fromarray(image).save(os.path.join(output_dirs["images"], filename))
    Image.fromarray(sato).save(os.path.join(output_dirs["sato"], filename))
    Image.fromarray(meijering).save(os.path.join(output_dirs["meijering"], filename))
    Image.fromarray(frangi).save(os.path.join(output_dirs["frangi"], filename))
    Image.fromarray(hessian).save(os.path.join(output_dirs["hessian"], filename))

print("✅ Done generating filtered image channels with enhanced contrast.")
