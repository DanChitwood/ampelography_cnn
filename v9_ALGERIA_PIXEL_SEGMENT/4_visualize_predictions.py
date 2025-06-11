# visualize_predictions.py

# === IMPORTS ===
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from skimage import color, filters
import torchvision.transforms.functional as TF # For image transforms
from tqdm import tqdm
import glob # For finding files in subdirectories
import torch.nn as nn # <--- ADD THIS LINE
import torch.nn.functional as F # <--- ADD THIS LINE

# === CONFIGURATION ===
# Path to your best model checkpoint (find this in your 'model_checkpoints' folder)
BEST_MODEL_PATH = "model_checkpoints/best_model_vein_dice_0.7863_epoch38.pt" # <--- REPLACE WITH YOUR ACTUAL BEST MODEL PATH
# Example: "model_checkpoints/best_model_vein_dice_0.7863_epoch38.pt"

# Directories for data (read-only)
PROCESSED_RGB_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/RGB_IMAGES"
SEVEN_CHANNEL_DATA_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/7CHANNEL_INPUTS"
GROUND_TRUTH_MASKS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/GROUND_TRUTH_MASKS"

# Output directory for visualizations
OUTPUT_VIZ_DIR = "PREDICTION_VISUALIZATIONS"
os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)

# Configuration for preprocessing unseen data (MUST MATCH main_data_preprocessing.py)
# You can find the TARGET_SIZE from the first lines of output from main_data_preprocessing.py
# If you don't remember, you can quickly re-run main_data_preprocessing.py for a moment,
# it prints the TARGET_SIZE at the beginning.
TARGET_WIDTH = 2048 # <--- REPLACE WITH YOUR ACTUAL TARGET_WIDTH
TARGET_HEIGHT = 2040 # <--- REPLACE WITH YOUR ACTUAL TARGET_HEIGHT
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# Filter parameters (MUST MATCH generate_7channel_inputs.py initially)
SATO_SIGMAS = range(1, 4)
MEIJERING_SIGMAS = range(1, 4)
FRANGI_SIGMAS = range(1, 4)
HESSIAN_SIGMAS = range(1, 4)

# UNet Model Configuration (must match your trained model)
NUM_INPUT_CHANNELS = 7
NUM_CLASSES = 3 # 0: Background, 1: Blade, 2: Vein

# Visualization colors (RGB for classes 0, 1, 2)
# Background will be transparent or black, Blade will be green, Vein will be red
COLORS = {
    0: (0, 0, 0, 0),    # Transparent for Background (alpha 0)
    1: (0, 255, 0, 128),  # Semi-transparent Green for Blade
    2: (255, 0, 0, 128) # Semi-transparent Red for Vein
}

# Device for inference
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# =============== UNet Model Definition (MUST BE IDENTICAL TO TRAINING) ===============
# Copy the UNet class definition exactly as it was in train_segmentation_model.py
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(512, 256), CBR(256, 256))

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(128, 64), CBR(64, 64)) # Corrected to 64 for second CBR

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

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

        return self.out_conv(d2)

# =============== Preprocessing Functions (Replicated from main_data_preprocessing.py) ===============
# These are needed to preprocess the raw, unseen images for inference.

def rotate_to_wide(image_pil):
    """Rotates an image so its width is >= its height."""
    width, height = image_pil.size
    if height > width:
        image_pil = image_pil.transpose(Image.Transpose.ROTATE_270) # Rotate 90 degrees clockwise
    return image_pil

def rescale_and_pad_image(image_pil, target_size):
    """
    Rescales image to fit within target_size while preserving aspect ratio,
    then pads with white to reach target_size.
    """
    # Use ImageOps.contain to scale down while preserving aspect ratio
    # If the image is smaller than target_size, it will be scaled up.
#    img_contained = Image.thumbnail(image_pil, target_size, Image.LANCZOS) # Use thumbnail for scaling down
    # Thumbnail scales in-place and might not stretch to target_size if aspect ratio differs.
    # ImageOps.contain is better for consistent scaling and padding.
    
    # Correct ImageOps.contain usage:
    img_contained = Image.new("RGB", target_size, (255, 255, 255)) # Create white canvas
    # Calculate paste position for centering
    
    # Calculate scaling factor for ImageOps.contain manually to fit inside TARGET_SIZE
    original_width, original_height = image_pil.size
    target_width, target_height = target_size
    
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale_factor = min(scale_w, scale_h) # Fit within both dimensions

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    scaled_img = image_pil.resize((new_width, new_height), Image.LANCZOS)

    # Calculate paste position to center the scaled image on the target_size canvas
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    padded_img = Image.new("RGB", target_size, (255, 255, 255)) # White background
    padded_img.paste(scaled_img, (paste_x, paste_y))
    
    return padded_img

def enhance_contrast(arr, percentile=99.9):
    """Clips values above the given percentile and rescales to 0–1."""
    vmax = np.percentile(arr, percentile)
    if vmax == 0:
        return np.zeros_like(arr)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled

def create_7channel_input(image_rgb_pil):
    """Applies ridge filters and stacks channels for a single image."""
    image_rgb_float = np.array(image_rgb_pil).astype(np.float32) / 255.0
    gray_image = color.rgb2gray(image_rgb_float)

    sato_raw = filters.sato(gray_image, sigmas=SATO_SIGMAS, black_ridges=False)
    meijering_raw = filters.meijering(gray_image, sigmas=MEIJERING_SIGMAS, black_ridges=False)
    frangi_raw = filters.frangi(gray_image, sigmas=FRANGI_SIGMAS, black_ridges=False)
    hessian_raw = filters.hessian(gray_image, sigmas=HESSIAN_SIGMAS, black_ridges=True)

    sato_processed = enhance_contrast(sato_raw).astype(np.float32)
    meijering_processed = enhance_contrast(meijering_raw).astype(np.float32)
    frangi_processed = enhance_contrast(frangi_raw).astype(np.float32)
    hessian_processed = enhance_contrast(hessian_raw).astype(np.float32)

    sato_channel = np.expand_dims(sato_processed, axis=-1)
    meijering_channel = np.expand_dims(meijering_processed, axis=-1)
    frangi_channel = np.expand_dims(frangi_processed, axis=-1)
    hessian_channel = np.expand_dims(hessian_processed, axis=-1)

    seven_channel_input = np.concatenate([
        image_rgb_float,
        sato_channel,
        meijering_channel,
        frangi_channel,
        hessian_channel
    ], axis=-1)
    
    return torch.from_numpy(seven_channel_input).permute(2, 0, 1) # (C, H, W)

# =============== Visualization Helper ===============
def visualize_segmentation(original_rgb_pil, ground_truth_mask=None, predicted_mask_np=None, output_path="output.png"):
    """
    Creates a visualization of the original image, ground truth, prediction, and an overlay.
    """
    width, height = original_rgb_pil.size
    
    # Create an image for the segmented prediction
    pred_viz = Image.new("RGBA", (width, height), (0, 0, 0, 255)) # Start with black
    if predicted_mask_np is not None:
        for class_id, color_rgba in COLORS.items():
            if class_id == 0: continue # Skip background
            mask_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0)) # Transparent layer
            draw = ImageDraw.Draw(mask_layer)
            # Find pixels belonging to this class and draw them
            pixels_of_class = np.argwhere(predicted_mask_np == class_id)
            for y, x in pixels_of_class:
                draw.point((x, y), fill=color_rgba)
            pred_viz = Image.alpha_composite(pred_viz, mask_layer)
            
    # Create an image for the ground truth visualization (if available)
    gt_viz = Image.new("RGBA", (width, height), (0, 0, 0, 255)) # Start with black
    if ground_truth_mask is not None:
        for class_id, color_rgba in COLORS.items():
            if class_id == 0: continue # Skip background
            mask_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0)) # Transparent layer
            draw = ImageDraw.Draw(mask_layer)
            pixels_of_class = np.argwhere(ground_truth_mask == class_id)
            for y, x in pixels_of_class:
                draw.point((x, y), fill=color_rgba)
            gt_viz = Image.alpha_composite(gt_viz, mask_layer)

    # Create composite overlay
    overlay_viz = original_rgb_pil.copy().convert("RGBA")
    if predicted_mask_np is not None:
        pred_overlay_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(pred_overlay_layer)
        for class_id, color_rgba in COLORS.items():
            if class_id == 0: continue
            pixels_of_class = np.argwhere(predicted_mask_np == class_id)
            for y, x in pixels_of_class:
                draw_overlay.point((x, y), fill=color_rgba)
        overlay_viz = Image.alpha_composite(overlay_viz, pred_overlay_layer)

    # Create a blank image to combine all visualizations
    # 4 images horizontally: Original | GT | Prediction | Overlay
    combined_width = width * 4
    combined_image = Image.new("RGB", (combined_width, height))
    combined_image.paste(original_rgb_pil, (0, 0))
    combined_image.paste(gt_viz.convert("RGB"), (width, 0)) # Convert RGBA to RGB for pasting
    combined_image.paste(pred_viz.convert("RGB"), (width * 2, 0))
    combined_image.paste(overlay_viz.convert("RGB"), (width * 3, 0))
    
    # Save the combined image
    combined_image.save(output_path)


# =============== Main Prediction & Visualization Logic ===============
def run_visualization(model, num_viz_validation=5, num_viz_unseen=5, unseen_data_root_dir=None):
    model.eval() # Set model to evaluation mode
    print(f"Loading model from: {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    print("Model loaded successfully.")

    # --- 1. Visualize from Validation Set ---
    print(f"\n--- Visualizing {num_viz_validation} samples from Validation Set ---")
    all_processed_images = sorted([f for f in os.listdir(SEVEN_CHANNEL_DATA_DIR) if f.endswith(".npy")])
    if num_viz_validation > len(all_processed_images):
        print(f"Warning: Requested {num_viz_validation} validation images, but only {len(all_processed_images)} available. Visualizing all available.")
        num_viz_validation = len(all_processed_images)

    # Select random samples from the processed data
    selected_indices = np.random.choice(len(all_processed_images), num_viz_validation, replace=False)

    for i, idx in enumerate(tqdm(selected_indices, desc="Validation Viz")):
        base_name = os.path.splitext(all_processed_images[idx])[0]
        
        # Load processed 7-channel input and original RGB
        seven_channel_input = np.load(os.path.join(SEVEN_CHANNEL_DATA_DIR, all_processed_images[idx]))
        original_rgb_pil = Image.open(os.path.join(PROCESSED_RGB_DIR, base_name + ".png")).convert("RGB")
        
        # Load ground truth mask
        ground_truth_mask = np.array(Image.open(os.path.join(GROUND_TRUTH_MASKS_DIR, base_name + ".png")).convert('L'))

        # Prepare input for model
        input_tensor = torch.from_numpy(seven_channel_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE) # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # (H, W)

        # Visualize and save
        output_path = os.path.join(OUTPUT_VIZ_DIR, f"val_prediction_{base_name}.png")
        visualize_segmentation(original_rgb_pil, ground_truth_mask, predicted_mask, output_path)

    # --- 2. Visualize from Unseen Data (if provided) ---
    if unseen_data_root_dir and os.path.exists(unseen_data_root_dir):
        print(f"\n--- Visualizing {num_viz_unseen} samples from Unseen Data ({unseen_data_root_dir}) ---")
        # Find all .jpg or .png images recursively
        unseen_image_paths = sorted(glob.glob(os.path.join(unseen_data_root_dir, '**', '*.png'), recursive=True) +
                                    glob.glob(os.path.join(unseen_data_root_dir, '**', '*.jpg'), recursive=True))
        
        if num_viz_unseen > len(unseen_image_paths):
            print(f"Warning: Requested {num_viz_unseen} unseen images, but only {len(unseen_image_paths)} available. Visualizing all available.")
            num_viz_unseen = len(unseen_image_paths)

        # Select random unseen images
        selected_unseen_paths = np.random.choice(unseen_image_paths, num_viz_unseen, replace=False)

        for i, img_path in enumerate(tqdm(selected_unseen_paths, desc="Unseen Viz")):
            original_file_name = os.path.basename(img_path)
            
            # --- Preprocess unseen image ---
            raw_rgb_pil = Image.open(img_path).convert("RGB")
            
            # 1. Rotate to wide
            rotated_img_pil = rotate_to_wide(raw_rgb_pil)
            
            # 2. Rescale and pad
            padded_img_pil = rescale_and_pad_image(rotated_img_pil, TARGET_SIZE)
            
            # 3. Create 7-channel input
            input_tensor = create_7channel_input(padded_img_pil).unsqueeze(0).to(DEVICE) # Add batch dim

            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # (H, W)

            # Visualize and save. No ground truth available for unseen data.
            output_path = os.path.join(OUTPUT_VIZ_DIR, f"unseen_prediction_{os.path.splitext(original_file_name)[0]}.png")
            visualize_segmentation(padded_img_pil, None, predicted_mask, output_path) # Pass padded_img_pil as original for visualization
            
    else:
        print("\nSkipping unseen data visualization. Set 'unseen_data_root_dir' in config to a valid path.")

    print(f"\n✅ All visualizations saved to '{OUTPUT_VIZ_DIR}'.")

if __name__ == "__main__":
    # Initialize the model (it will be loaded with trained weights later)
    model = UNet(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_CLASSES)
    
    # --- IMPORTANT: Set your unseen data root directory here ---
    # Example: UNSEEN_DATA_ROOT = "/Users/chitwoo9/Desktop/MyUnseenLeafData"
    # Make sure this directory contains subfolders with your unseen images
    # UNSEEN_DATA_ROOT = None # Set to None if you don't have unseen data yet or don't want to visualize it now
    UNSEEN_DATA_ROOT = "/Users/chitwoo9/Desktop/example_unseen_leaves" # <--- REPLACE THIS IF YOU WANT TO VISUALIZE UNSEEN DATA

    run_visualization(model, 
                      num_viz_validation=5, # Number of validation images to visualize
                      num_viz_unseen=5,     # Number of unseen images to visualize
                      unseen_data_root_dir=UNSEEN_DATA_ROOT)