import os
import numpy as np
from PIL import Image
from skimage import io, color, draw, filters, img_as_ubyte
from tqdm import tqdm

# Input and output folders
input_dir = "training_data"
output_dirs = {
    "images": "images",
    "masks": "masks",
    "sato": "sato",
    "meijering": "meijering",
    "frangi": "frangi",
    "hessian": "hessian"
}

# Create output folders if they don't exist
for folder in output_dirs.values():
    os.makedirs(folder, exist_ok=True)

# Get all image base names
jpg_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
base_names = [os.path.splitext(f)[0] for f in jpg_files]

def read_coords(filepath):
    return np.loadtxt(filepath, delimiter=None)

def rasterize_polygon(coords, shape):
    rr, cc = draw.polygon(coords[:,1], coords[:,0], shape=shape)
    return rr, cc

def crop_to_nonzero(img, mask):
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return img, mask  # Avoid crashing if mask is empty
    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()
    return img[min_y:max_y+1, min_x:max_x+1], mask[min_y:max_y+1, min_x:max_x+1]

for base in tqdm(base_names, desc="Processing leaves"):
    img_path = os.path.join(input_dir, base + ".jpg")
    blade_path = os.path.join(input_dir, base + "_blade.txt")
    veins_path = os.path.join(input_dir, base + "_veins.txt")

    if not all(os.path.exists(p) for p in [img_path, blade_path, veins_path]):
        print(f"Skipping {base} due to missing files.")
        continue

    # Load image and outlines
    image = np.array(Image.open(img_path).convert("RGB"))
    blade_coords = read_coords(blade_path)
    vein_coords = read_coords(veins_path)

    # Rasterize
    blade_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    vein_mask = np.zeros_like(blade_mask)

    rr_b, cc_b = rasterize_polygon(blade_coords, image.shape[:2])
    rr_v, cc_v = rasterize_polygon(vein_coords, image.shape[:2])
    blade_mask[rr_b, cc_b] = 1
    vein_mask[rr_v, cc_v] = 1

    # Create final mask (0: background, 1: blade, 2: vein)
    final_mask = np.zeros_like(blade_mask)
    final_mask[blade_mask == 1] = 1
    final_mask[vein_mask == 1] = 2  # vein overwrites blade

    # Crop
    image_cropped, mask_cropped = crop_to_nonzero(image, final_mask)
    gray_cropped = color.rgb2gray(image_cropped)

    # Apply ridge filters
    print(f"Applying filters for: {base}")
    sato = img_as_ubyte(filters.sato(gray_cropped, sigmas=range(2, 8), black_ridges=False))
    meijering = img_as_ubyte(filters.meijering(gray_cropped, sigmas=range(2, 8), black_ridges=False))
    frangi = img_as_ubyte(filters.frangi(gray_cropped, sigmas=range(5, 40, 10), black_ridges=False))
    hessian = img_as_ubyte(filters.hessian(gray_cropped, sigmas=range(2, 8), black_ridges=True))

    # Save outputs
    filename = base + ".png"
    Image.fromarray(image_cropped).save(os.path.join(output_dirs["images"], filename))
    Image.fromarray(mask_cropped).save(os.path.join(output_dirs["masks"], filename))
    Image.fromarray(sato).save(os.path.join(output_dirs["sato"], filename))
    Image.fromarray(meijering).save(os.path.join(output_dirs["meijering"], filename))
    Image.fromarray(frangi).save(os.path.join(output_dirs["frangi"], filename))
    Image.fromarray(hessian).save(os.path.join(output_dirs["hessian"], filename))

print("✅ Done generating all filtered images and masks.")
