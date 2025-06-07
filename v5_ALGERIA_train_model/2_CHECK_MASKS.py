# === IMPORTS ===
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# === CONFIGURATION ===
background_dir = "BACKGROUNDS"
mask_dir = "MASKS"
output_dir = "FINAL_CHECK"
os.makedirs(output_dir, exist_ok=True)

# Colors: 0 = transparent, 1 = orange (blade), 2 = magenta (veins)
cmap = ListedColormap([
    (0, 0, 0, 0),         # background -> fully transparent
    (1.0, 0.55, 0.0, 0.7),  # blade -> orange, semi-transparent
    (1.0, 0.0, 1.0, 0.7),  # veins -> magenta, semi-transparent
])

# === PROCESSING ===
for fname in tqdm(sorted(os.listdir(background_dir))):
    if not fname.lower().endswith(".png"):
        continue

    bg_path = os.path.join(background_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    if not os.path.exists(mask_path):
        print(f"⚠️ Missing mask for {fname}, skipping...")
        continue

    # Load images
    bg_img = Image.open(bg_path).convert("RGB")
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    # Original background
    axs[0].imshow(bg_img)
    axs[0].axis("off")
    #axs[0].set_title("Modified Background")

    # Overlay: show background + mask overlay (with transparent background)
    axs[1].imshow(bg_img)
    axs[1].imshow(mask, cmap=cmap, interpolation="none")
    axs[1].axis("off")
    #axs[1].set_title("Overlay: Blade (orange), Veins (magenta)")

    # Save output
    out_path = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

print("✅ All overlays saved to FINAL_CHECK.")