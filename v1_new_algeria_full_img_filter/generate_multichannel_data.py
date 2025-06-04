import os
import numpy as np
from PIL import Image
from skimage import io, color, filters, img_as_ubyte
from tqdm import tqdm

# Input and output folders
input_dir = "new_algeria_images"
output_dirs = {
    "images": "images_rgb",
    "sato": "images_sato",
    "meijering": "images_meijering",
    "frangi": "images_frangi",
    "hessian": "images_hessian"
}

# Create output folders if they don't exist
for folder in output_dirs.values():
    os.makedirs(folder, exist_ok=True)

# Get all image base names
jpg_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
base_names = [os.path.splitext(f)[0] for f in jpg_files]

for base in tqdm(base_names, desc="Processing ridge filters"):
    img_path = os.path.join(input_dir, base + ".jpg")

    if not os.path.exists(img_path):
        print(f"Skipping {base}: missing image.")
        continue

    # Load image and convert to grayscale
    image = np.array(Image.open(img_path).convert("RGB"))
    gray_image = color.rgb2gray(image)

    # Apply ridge filters
    print(f"Applying filters for: {base}")
    sato = img_as_ubyte(filters.sato(gray_image, sigmas=range(2, 8), black_ridges=False))
    meijering = img_as_ubyte(filters.meijering(gray_image, sigmas=range(2, 8), black_ridges=False))
    frangi = img_as_ubyte(filters.frangi(gray_image, sigmas=range(5, 40, 10), black_ridges=False))
    hessian = img_as_ubyte(filters.hessian(gray_image, sigmas=range(2, 8), black_ridges=True))

    # Save outputs
    filename = base + ".png"
    Image.fromarray(image).save(os.path.join(output_dirs["images"], filename))
    Image.fromarray(sato).save(os.path.join(output_dirs["sato"], filename))
    Image.fromarray(meijering).save(os.path.join(output_dirs["meijering"], filename))
    Image.fromarray(frangi).save(os.path.join(output_dirs["frangi"], filename))
    Image.fromarray(hessian).save(os.path.join(output_dirs["hessian"], filename))

print("✅ Done generating filtered image channels for CNN input.")

