# === IMPORTS ===
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm

# === CONFIGURATION ===
blade_color = "orange"
vein_color = "magenta"
overlay_alpha = 0.7
plot_dpi = 100
MAX_DIMENSION = 2048

training_dir = "TRAINING_DATA"
image_extensions = [".jpg", ".jpeg", ".tif", ".tiff"]

output_dirs = {
    "background": "BACKGROUNDS",
    "blade": "BLADE",
    "veins": "VEINS",
    "plots": "PLOTS"
}
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

# === HELPERS ===
def read_coords(path):
    return np.loadtxt(path)

def resolve_image_path(folder, image_value):
    name, ext = os.path.splitext(image_value)
    candidates = [image_value] if ext else [f"{image_value}{e}" for e in image_extensions]
    all_files = os.listdir(folder)
    for candidate in candidates:
        for f in all_files:
            if f.lower() == candidate.lower():
                return os.path.join(folder, f)
    raise FileNotFoundError(f"No image found for base name {image_value} in {folder}")

def rotate_to_wide(img, coords):
    if img.width >= img.height:
        return img, coords
    img_rot = img.transpose(Image.Transpose.ROTATE_90)
    coords_rot = np.copy(coords)
    coords_rot[:, [0, 1]] = coords[:, [1, 0]]
    coords_rot[:, 1] = img.width - coords_rot[:, 1]
    return img_rot, coords_rot

def downscale_to_max_dim(size, max_dim):
    w, h = size
    scale = min(max_dim / max(w, h), 1.0)
    return int(w * scale), int(h * scale), scale

def transform_coords(coords, orig_size, new_size):
    scale = min(new_size[0] / orig_size[0], new_size[1] / orig_size[1])
    coords_scaled = coords * scale
    pad_x = (new_size[0] - orig_size[0] * scale) / 2
    pad_y = (new_size[1] - orig_size[1] * scale) / 2
    coords_trans = coords_scaled + [pad_x, pad_y]
    return coords_trans

def rescale_and_pad_image(img, target_size):
    img = ImageOps.contain(img, target_size)
    padded = Image.new("RGB", target_size, (255, 255, 255))
    paste_pos = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
    padded.paste(img, paste_pos)
    return padded

# === PHASE 1: Determine Target Size ===
info_files = [f for f in os.listdir(training_dir) if f.endswith("_info.csv")]
max_width, max_height = 0, 0

print("🔍 Scanning for maximum dimensions...")
for info_file in tqdm(info_files):
    try:
        fid = info_file.replace("_info.csv", "")
        meta = pd.read_csv(os.path.join(training_dir, info_file))
        imgval = meta.loc[meta["factor"] == "image", "value"].values[0]
        imgpath = resolve_image_path(training_dir, imgval)

        im = Image.open(imgpath).convert("RGB")
        if im.width < im.height:
            im = im.transpose(Image.Transpose.ROTATE_90)

        max_width = max(max_width, im.width)
        max_height = max(max_height, im.height)

    except Exception as e:
        print(f"⚠️ Skipping {info_file}: {e}")

target_width, target_height, _ = downscale_to_max_dim((max_width, max_height), MAX_DIMENSION)
target_size = (target_width, target_height)
print(f"📐 Target image size (max {MAX_DIMENSION}px): {target_size}")

# === PHASE 2: Generate and Save ===
print("🚀 Processing and saving outputs...")
name_counter = {}

for info_file in tqdm(info_files):
    try:
        fid = info_file.replace("_info.csv", "")
        img_info = pd.read_csv(os.path.join(training_dir, info_file))
        imgval = img_info.loc[img_info["factor"] == "image", "value"].values[0]
        imgpath = resolve_image_path(training_dir, imgval)
        blade_path = os.path.join(training_dir, f"{fid}_blade.txt")
        vein_path = os.path.join(training_dir, f"{fid}_veins.txt")

        if not os.path.exists(blade_path) or not os.path.exists(vein_path):
            print(f"⚠️ Skipping {fid}: Missing blade or vein file")
            continue

        name_base = fid.strip()
        count = name_counter.get(name_base, 0) + 1
        name_counter[name_base] = count
        outname = f"{name_base}_{count}" if count > 1 else name_base

        img_orig = Image.open(imgpath).convert("RGB")
        blade = read_coords(blade_path)
        vein = read_coords(vein_path)

        img_rot, blade_rot = rotate_to_wide(img_orig, blade)
        _, vein_rot = rotate_to_wide(img_orig, vein)

        img_trans = rescale_and_pad_image(img_rot, target_size)
        blade_trans = transform_coords(blade_rot, img_rot.size, target_size)
        vein_trans = transform_coords(vein_rot, img_rot.size, target_size)

        img_trans.save(os.path.join(output_dirs["background"], outname + ".png"))
        np.savetxt(os.path.join(output_dirs["blade"], outname + "_blade.txt"), blade_trans, fmt="%.2f")
        np.savetxt(os.path.join(output_dirs["veins"], outname + "_veins.txt"), vein_trans, fmt="%.2f")

        fig, ax = plt.subplots()
        ax.imshow(img_trans)
        ax.add_patch(Polygon(blade_trans, closed=True, color=blade_color, alpha=overlay_alpha))
        ax.add_patch(Polygon(vein_trans, closed=True, color=vein_color, alpha=overlay_alpha))
        ax.axis("off")
        fig.savefig(os.path.join(output_dirs["plots"], outname + ".png"), dpi=plot_dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    except Exception as e:
        print(f"⚠️ Skipping {fid}: {e}")

print("✅ All files processed and saved.")
