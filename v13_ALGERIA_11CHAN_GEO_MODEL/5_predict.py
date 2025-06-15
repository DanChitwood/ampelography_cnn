import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import color, filters, exposure, img_as_float # For 11-channel generation

# --- CONFIGURATION ---
NEW_ALGERIA_LEAVES_ROOT = "NEW_ALGERIA_LEAVES" # Directory containing class subfolders of JPGs
BEST_MODEL_PATH = "V1_best_model_vein_dice_0.7697_epoch29.pt" # Update with your best model path
OUTPUT_PREDICTIONS_DIR = "PREDICTIONS" # Parent directory for all outputs
OUTPUT_MASKS_DIR = os.path.join(OUTPUT_PREDICTIONS_DIR, "MASKS")
OUTPUT_OVERLAYS_DIR = os.path.join(OUTPUT_PREDICTIONS_DIR, "OVERLAYS")

# Ensure output directories exist
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
os.makedirs(OUTPUT_OVERLAYS_DIR, exist_ok=True)

# Device configuration (must match training device if not using map_location)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model parameters (must match training parameters)
NUM_SEG_CLASSES = 3 # Background, Blade, Vein
IN_CHANNELS = 11 # As used in your UNet

# Filter parameters for 11-channel generation (must match training preprocessing)
SIGMA_RANGE_VEINS = range(1, 5)
SIGMA_RANGE_LARGER_FEATURES = range(1, 7)

# Overlay colors and transparency
BLADE_COLOR = 'dodgerblue'
BLADE_ALPHA = 0.7
VEIN_COLOR = 'magenta'
VEIN_ALPHA = 1.0

# --- HELPER FUNCTIONS for 11-channel generation (copied from your script) ---
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

def generate_11_channel_input(rgb_image_path):
    """
    Generates an 11-channel input array from an RGB image,
    mimicking the preprocessing from 3_generate_11channel_inputs.py.
    """
    try:
        image_rgb_pil = Image.open(rgb_image_path).convert("RGB")
        image_rgb_float = np.array(image_rgb_pil).astype(np.float32) / 255.0 # Normalize RGB to 0-1

        # 1. RGB Channels (3)
        # Already image_rgb_float

        # 2. Grayscale Channel (1)
        gray_image = color.rgb2gray(image_rgb_float)
        gray_channel = np.expand_dims(gray_image, axis=-1).astype(np.float32)

        # 3. Lab Color Space Channels (3)
        image_lab_float = color.rgb2lab(image_rgb_float)
        l_channel = np.expand_dims(image_lab_float[:,:,0] / 100.0, axis=-1).astype(np.float32)
        a_channel = np.expand_dims((image_lab_float[:,:,1] + 128) / 255.0, axis=-1).astype(np.float32)
        b_channel = np.expand_dims((image_lab_float[:,:,2] + 128) / 255.0, axis=-1).astype(np.float32)

        # 4. Ridge Filters (4 channels)
        sato_raw = filters.sato(gray_image, sigmas=SIGMA_RANGE_VEINS, black_ridges=False)
        meijering_raw = filters.meijering(gray_image, sigmas=SIGMA_RANGE_VEINS, black_ridges=False)
        frangi_raw = filters.frangi(gray_image, sigmas=SIGMA_RANGE_VEINS, black_ridges=False)
        hessian_raw = filters.hessian(gray_image, sigmas=SIGMA_RANGE_LARGER_FEATURES, black_ridges=True)

        sato_processed = enhance_contrast(sato_raw).astype(np.float32)
        meijering_processed = enhance_contrast(meijering_raw).astype(np.float32)
        frangi_processed = enhance_contrast(frangi_raw).astype(np.float32)
        hessian_processed = enhance_contrast(hessian_raw).astype(np.float32)

        sato_channel = np.expand_dims(sato_processed, axis=-1)
        meijering_channel = np.expand_dims(meijering_processed, axis=-1)
        frangi_channel = np.expand_dims(frangi_processed, axis=-1)
        hessian_channel = np.expand_dims(hessian_processed, axis=-1)

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
        ], axis=-1) # (H, W, 11)

        return eleven_channel_input

    except Exception as e:
        print(f"❌ Error generating 11-channel input for {os.path.basename(rgb_image_path)}: {e}")
        return None

# --- UNet Model Definition (Must be identical to your training script) ---
class UNet(nn.Module):
    def __init__(self, in_channels, num_seg_classes):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder path
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        
        self.pool = nn.MaxPool2d(2)

        # Decoder path
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(512, 256), CBR(256, 256))

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        # Output head for Segmentation
        self.seg_out_conv = nn.Conv2d(64, num_seg_classes, 1) # Outputs logits for each class

        # Output head for Geodesic Distance Regression
        self.geo_out_conv = nn.Sequential(
            nn.Conv2d(64, 1, 1), # Output 1 channel
            nn.Sigmoid()         # Constrain output to 0-1 range
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d4 = self.up4(e4)
        if d4.shape != e3.shape:
             d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if d3.shape != e2.shape:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape != e1.shape:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        seg_output = self.seg_out_conv(d2)
        geo_output = self.geo_out_conv(d2)

        return seg_output, geo_output

# --- Main Prediction Loop ---
def main():
    # Load the trained model
    model = UNet(in_channels=IN_CHANNELS, num_seg_classes=NUM_SEG_CLASSES).to(DEVICE)
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading model from {BEST_MODEL_PATH}...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    else:
        print(f"Error: Model checkpoint not found at {BEST_MODEL_PATH}")
        return

    # Iterate through each class subfolder in NEW_ALGERIA_LEAVES
    for class_folder in sorted(os.listdir(NEW_ALGERIA_LEAVES_ROOT)):
        class_folder_path = os.path.join(NEW_ALGERIA_LEAVES_ROOT, class_folder)
        if not os.path.isdir(class_folder_path):
            continue

        print(f"\nProcessing images in class folder: {class_folder}")
        image_files = sorted([f for f in os.listdir(class_folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        if not image_files:
            print(f"No image files found in {class_folder_path}. Skipping.")
            continue

        for fname in tqdm(image_files, desc=f"Predicting on {class_folder}"):
            full_image_path = os.path.join(class_folder_path, fname)
            
            # Extract base FID (filename without extension, which acts as unique ID)
            fid = os.path.splitext(fname)[0]

            # Generate 11-channel input from the original RGB image
            eleven_channel_input_np = generate_11_channel_input(full_image_path)
            
            if eleven_channel_input_np is None:
                continue # Skip if 11-channel generation failed

            # Convert to PyTorch tensor (H, W, C) -> (C, H, W) and add batch dimension
            image_tensor = torch.from_numpy(eleven_channel_input_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                seg_output_logits, geo_output = model(image_tensor)
                
            # Process Segmentation Output
            predicted_seg_mask_tensor = torch.argmax(seg_output_logits.squeeze(0), dim=0) # Remove batch dim, get class with max logit
            predicted_seg_mask_np = predicted_seg_mask_tensor.cpu().numpy() # (H, W)
            
            # Process Geodesic Output
            predicted_geo_map_np = geo_output.squeeze(0).squeeze(0).cpu().numpy() # Remove batch & channel dims (H, W)

            # --- Save Raw Masks ---
            # Segmentation Mask (0: Background, 1: Blade, 2: Vein)
            seg_mask_pil = Image.fromarray(predicted_seg_mask_np.astype(np.uint8), mode='L')
            seg_mask_pil.save(os.path.join(OUTPUT_MASKS_DIR, f"{fid}_segmentation_mask.png"))
            
            # Geodesic Map (save as numpy array for full float precision)
            np.save(os.path.join(OUTPUT_MASKS_DIR, f"{fid}_geodesic_map.npy"), predicted_geo_map_np)

            # --- Generate and Save Overlays ---
            # Load original RGB image for overlay
            original_rgb_pil = Image.open(full_image_path).convert("RGB")
            original_rgb_np = np.array(original_rgb_pil)

            plt.figure(figsize=(original_rgb_np.shape[1]/100, original_rgb_np.shape[0]/100), dpi=100) # Size in inches, dpi for resolution
            plt.imshow(original_rgb_np)
            
            # Overlay Blade (class 1)
            blade_mask = (predicted_seg_mask_np == 1)
            plt.imshow(blade_mask, cmap='Blues', alpha=BLADE_ALPHA * blade_mask, interpolation='nearest') # alpha=BLADE_ALPHA * blade_mask to apply alpha only where mask is True
            
            # Overlay Vein (class 2)
            vein_mask = (predicted_seg_mask_np == 2)
            plt.imshow(vein_mask, cmap='Oranges', alpha=VEIN_ALPHA * vein_mask, interpolation='nearest') # Using Oranges for magenta-like effect, or specify exact RGB/RGBA for magenta

            # If you want exact magenta:
            # magenta_mask = np.zeros((*original_rgb_np.shape[:2], 4)) # H, W, RGBA
            # magenta_mask[vein_mask] = [1, 0, 1, VEIN_ALPHA] # R, G, B, Alpha for magenta
            # plt.imshow(magenta_mask)

            plt.axis('off') # Hide axes
            plt.title(fid, loc='left', color='white', fontsize=10, backgroundcolor='black', pad=-5) # Display FID as a small title
            plt.tight_layout(pad=0) # Remove padding around the image

            overlay_save_path = os.path.join(OUTPUT_OVERLAYS_DIR, f"{fid}_overlay.png")
            plt.savefig(overlay_save_path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close() # Close plot to free memory

    print("\n✅ Prediction complete! Results saved to 'PREDICTIONS' folder.")

if __name__ == "__main__":
    main()