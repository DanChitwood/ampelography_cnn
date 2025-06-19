import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import color, filters, exposure, img_as_float
from skimage.measure import label, regionprops
from scipy import ndimage
import pandas as pd

# --- CONFIGURATION ---
INFERENCE_INPUT_ROOT = "INFERENCE_INPUT" # Directory containing class subfolders of JPGs
BEST_MODEL_PATH = "V1_best_model_vein_dice_0.7697_epoch29.pt" # Update with your best model path
OUTPUT_PREDICTIONS_DIR = "INFERENCE_OUTPUTS" # Renamed output folder as requested

# Subdirectories for organized output
OUTPUT_COMPONENT_MASKS_DIR = os.path.join(OUTPUT_PREDICTIONS_DIR, "COMPONENT_MASKS")
OUTPUT_COMPONENT_RGB_CROPS_DIR = os.path.join(OUTPUT_PREDICTIONS_DIR, "COMPONENT_RGB_CROPS")
OUTPUT_COMPONENT_OVERLAYS_DIR = os.path.join(OUTPUT_PREDICTIONS_DIR, "COMPONENT_OVERLAYS")
METADATA_CSV_PATH = os.path.join(OUTPUT_PREDICTIONS_DIR, "component_metadata.csv")

# Ensure output parent directory exists
os.makedirs(OUTPUT_PREDICTIONS_DIR, exist_ok=True)
# Ensure subdirectories exist
os.makedirs(OUTPUT_COMPONENT_MASKS_DIR, exist_ok=True)
os.makedirs(OUTPUT_COMPONENT_RGB_CROPS_DIR, exist_ok=True)
os.makedirs(OUTPUT_COMPONENT_OVERLAYS_DIR, exist_ok=True)


# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model parameters
NUM_SEG_CLASSES = 3 # Background, Blade, Vein
IN_CHANNELS = 11 # As used in your UNet

# Filter parameters for 11-channel generation
SIGMA_RANGE_VEINS = range(1, 5)
SIGMA_RANGE_LARGER_FEATURES = range(1, 7)

# New filter parameter for connected components
MIN_BBOX_DIMENSION = 50 # Ignore components if any bounding box dimension is less than 50 pixels

# Overlay parameters
VEIN_OVERLAY_COLORMAP_NAME = 'plasma'
VEIN_ALPHA = 1.0 # Alpha for vein overlay. Keep 1.0 for full opacity

# --- HELPER FUNCTIONS for 11-channel generation (copied from your script) ---
def enhance_contrast(arr, percentile=99.9):
    """Clips values above percentile and rescales to 0-1."""
    if arr.size == 0:
        return np.array([])
    vmax = np.percentile(arr, percentile)
    if vmax == 0:
        return np.zeros_like(arr)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled

def generate_11_channel_input(rgb_image_path):
    """Generates an 11-channel input array from an RGB image."""
    try:
        image_rgb_pil = Image.open(rgb_image_path).convert("RGB")
        image_rgb_float = np.array(image_rgb_pil).astype(np.float32) / 255.0

        gray_image = color.rgb2gray(image_rgb_float)
        gray_channel = np.expand_dims(gray_image, axis=-1).astype(np.float32)

        image_lab_float = color.rgb2lab(image_rgb_float)
        l_channel = np.expand_dims(image_lab_float[:,:,0] / 100.0, axis=-1).astype(np.float32)
        a_channel = np.expand_dims((image_lab_float[:,:,1] + 128) / 255.0, axis=-1).astype(np.float32)
        b_channel = np.expand_dims((image_lab_float[:,:,2] + 128) / 255.0, axis=-1).astype(np.float32)

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
            image_rgb_float, gray_channel, l_channel, a_channel, b_channel,
            sato_channel, meijering_channel, frangi_channel, hessian_channel
        ], axis=-1)
        return eleven_channel_input

    except Exception as e:
        print(f"❌ Error generating 11-channel input for {os.path.basename(rgb_image_path)}: {e}")
        return None

# --- UNet Model Definition (Must be identical to your training script) ---
class UNet(nn.Module):
    def __init__(self, in_channels, num_seg_classes):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
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
        self.dec2 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        self.seg_out_conv = nn.Conv2d(64, num_seg_classes, 1)
        self.geo_out_conv = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d4 = self.up4(e4)
        if d4.shape != e3.shape: d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        if d3.shape != e2.shape: d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        if d2.shape != e1.shape: d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
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

    # --- Resumable Prediction Logic ---
    processed_original_images = set()
    all_component_metadata = []
    
    # Initialize global_component_serial_id
    # If resuming, we need to find the max serial ID already used.
    global_component_serial_id = 1 

    if os.path.exists(METADATA_CSV_PATH):
        print(f"Metadata CSV found at {METADATA_CSV_PATH}. Loading existing data for resuming...")
        existing_df = pd.read_csv(METADATA_CSV_PATH)
        all_component_metadata = existing_df.to_dict('records')
        processed_original_images = set(existing_df['original_image_path'].unique())
        
        # Determine the next available global_component_serial_id
        # We need to parse 'component_name' to extract the numerical part.
        # This assumes 'component_name' is formatted as 'ALGERIA_CLASS_FOLDER_NUMBER'.
        if not existing_df.empty:
            max_existing_id = 0
            for comp_name in existing_df['component_name']:
                try:
                    # Split at the last underscore to get the number
                    parts = comp_name.rsplit('_', 1)
                    if len(parts) > 1:
                        max_existing_id = max(max_existing_id, int(parts[-1]))
                except ValueError:
                    # Handle cases where component_name might not end with _NUMBER or other parsing issues
                    pass
            global_component_serial_id = max_existing_id + 1

        print(f"Found {len(processed_original_images)} unique original images already processed. Resuming...")
        print(f"Next global component ID will start from: {global_component_serial_id}")
    else:
        # Create empty CSV with header if it doesn't exist
        print(f"Metadata CSV not found. Creating a new one at {METADATA_CSV_PATH}")
        metadata_df = pd.DataFrame(columns=[
            "original_image_path", "component_name", # component_name will be like 'ALGERIA_ClassFolder_SerialID'
            "component_idx_in_original_image", # New field to indicate its index within its source image
            "blade_pixels", "vein_pixels", "background_pixels_internal_holes", 
            "total_bbox_pixels", "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
            "mask_file", "rgb_crop_file", "overlay_file"
        ])
        metadata_df.to_csv(METADATA_CSV_PATH, index=False)


    # Get the colormap for vein overlay (using the recommended matplotlib.colormaps for future compatibility)
    vein_cmap = plt.colormaps.get_cmap(VEIN_OVERLAY_COLORMAP_NAME)

    # Collect all image paths to process
    images_to_process = []
    for class_folder in sorted(os.listdir(INFERENCE_INPUT_ROOT)):
        class_folder_path = os.path.join(INFERENCE_INPUT_ROOT, class_folder)
        if not os.path.isdir(class_folder_path):
            continue
        for fname in os.listdir(class_folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_image_path = os.path.join(class_folder_path, fname)
                images_to_process.append((class_folder, fname, full_image_path))
    
    # Filter out already processed images
    filtered_images_to_process = []
    for class_folder, fname, full_image_path in images_to_process:
        if full_image_path not in processed_original_images:
            filtered_images_to_process.append((class_folder, fname, full_image_path))
    
    # Use tqdm on the filtered list for accurate progress
    print(f"Starting processing on {len(filtered_images_to_process)} new images.")
    for class_folder, fname, original_image_full_path in tqdm(filtered_images_to_process, desc="Overall Prediction Progress"):
        
        # This image will be processed, so add its path to the set for the current run
        # This is important for resuming partial progress within a single script execution
        processed_original_images.add(original_image_full_path)

        # Generate 11-channel input from the original RGB image
        eleven_channel_input_np = generate_11_channel_input(original_image_full_path)
        
        if eleven_channel_input_np is None:
            continue # Skip if 11-channel generation failed

        # Load original RGB image for cropping and overlay
        original_rgb_pil = Image.open(original_image_full_path).convert("RGB")
        original_rgb_np = np.array(original_rgb_pil)

        # Convert to PyTorch tensor (H, W, C) -> (C, H, W) and add batch dimension
        image_tensor = torch.from_numpy(eleven_channel_input_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            seg_output_logits, _ = model(image_tensor) # Discard geo_output
            
        # Process Segmentation Output
        predicted_seg_mask_np = torch.argmax(seg_output_logits.squeeze(0), dim=0).cpu().numpy() # (H, W)

        # --- Connected Component Analysis ---
        # Combine blade (1) and vein (2) into a single binary mask for connected component labeling
        combined_leaf_mask_binary = ((predicted_seg_mask_np == 1) | (predicted_seg_mask_np == 2)).astype(np.uint8)
        labeled_mask = label(combined_leaf_mask_binary, connectivity=2) # 8-connectivity
        properties = regionprops(labeled_mask)

        current_image_components_metadata = [] # Temporarily store metadata for components from current image
        
        # This counter tracks components *within the current original image*
        component_idx_in_original_image = 1 
        
        for region in properties:
            min_row, min_col, max_row, max_col = region.bbox
            bbox_height = max_row - min_row
            bbox_width = max_col - min_col

            # --- Apply Minimum Bounding Box Dimension Filter ---
            if bbox_height < MIN_BBOX_DIMENSION or bbox_width < MIN_BBOX_DIMENSION:
                # print(f"Skipping component (bbox too small): {bbox_width}x{bbox_height}")
                continue # Skip this component if its bounding box is too small
            
            # --- Generate unique name for the component (GLOBAL uniqueness now) ---
            # Added "ALGERIA_" prefix as requested
            component_base_name = f"ALGERIA_{class_folder}_{global_component_serial_id}"
            
            # --- Crop Images and Masks for this component ---
            cropped_rgb = original_rgb_np[min_row:max_row, min_col:max_col, :]
            
            # Get the segmentation mask for only this specific component's bounding box
            cropped_seg_mask_full_classes = predicted_seg_mask_np[min_row:max_row, min_col:max_col]
            
            # Create a binary mask that only includes the pixels belonging to the current region from `labeled_mask`
            # This ensures we only consider the *current* connected component within its bbox, ignoring other components
            component_binary_mask_in_bbox = (labeled_mask[min_row:max_row, min_col:max_col] == region.label).astype(np.uint8)
            
            # Apply this binary mask to the cropped segmentation. 
            # Pixels not part of this specific connected component (within its bbox) will be set to 0 (background).
            cropped_component_seg_mask_filtered = cropped_seg_mask_full_classes * component_binary_mask_in_bbox

            # --- Count Pixels for Metadata ---
            blade_pixels = np.sum(cropped_component_seg_mask_filtered == 1)
            vein_pixels = np.sum(cropped_component_seg_mask_filtered == 2)
            
            # To calculate internal holes, we fill the holes in the component's binary mask (blade + vein)
            component_binary_only_leaf = ((cropped_component_seg_mask_filtered == 1) | 
                                          (cropped_component_seg_mask_filtered == 2)).astype(bool)
            filled_component_binary = ndimage.binary_fill_holes(component_binary_only_leaf)
            filled_leaf_area = np.sum(filled_component_binary)
            
            background_pixels_internal_holes = filled_leaf_area - (blade_pixels + vein_pixels)

            total_pixels_in_bbox = cropped_component_seg_mask_filtered.size
            
            # --- Save Cropped Mask ---
            mask_save_path = os.path.join(OUTPUT_COMPONENT_MASKS_DIR, f"{component_base_name}_mask.png")
            Image.fromarray(cropped_component_seg_mask_filtered.astype(np.uint8), mode='L').save(mask_save_path)

            # --- Save Cropped RGB Image ---
            rgb_crop_save_path = os.path.join(OUTPUT_COMPONENT_RGB_CROPS_DIR, f"{component_base_name}_rgb_crop.png")
            Image.fromarray(cropped_rgb).save(rgb_crop_save_path)

            # --- Generate and Save Custom Overlay ---
            plt.figure(figsize=(cropped_rgb.shape[1]/100, cropped_rgb.shape[0]/100), dpi=100)
            
            # Create an empty black canvas for the overlay
            overlay_image = np.zeros_like(cropped_rgb, dtype=np.uint8) # Start with black
            
            # Apply blade pixels with original RGB values
            blade_mask = (cropped_component_seg_mask_filtered == 1)
            overlay_image[blade_mask] = cropped_rgb[blade_mask]

            # Apply vein pixels with colormap
            vein_mask = (cropped_component_seg_mask_filtered == 2)
            if np.any(vein_mask):
                vein_original_rgb_pixels = cropped_rgb[vein_mask]
                vein_grayscale_intensity = color.rgb2gray(vein_original_rgb_pixels / 255.0) 
                colored_vein_pixels_rgba = vein_cmap(vein_grayscale_intensity)
                
                # Assign the RGB part of the colored vein pixels (discarding alpha if not needed for blending)
                # Since VEIN_ALPHA is 1.0, direct assignment is fine
                overlay_image[vein_mask] = (colored_vein_pixels_rgba[:, :3] * 255).astype(np.uint8)

            plt.imshow(overlay_image) # Display the constructed overlay image
            
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            overlay_save_path = os.path.join(OUTPUT_COMPONENT_OVERLAYS_DIR, f"{component_base_name}_overlay.png")
            # Explicitly set facecolor to black for the saved figure and transparent=False
            plt.savefig(overlay_save_path, bbox_inches='tight', pad_inches=0, facecolor='black', transparent=False)
            plt.close() # Close plot to free memory

            # --- Append Metadata for Current Component ---
            current_image_components_metadata.append({
                "original_image_path": original_image_full_path,
                "component_name": component_base_name, # This is the globally unique ID name
                "component_idx_in_original_image": component_idx_in_original_image, # This is its index within the original image
                "blade_pixels": blade_pixels,
                "vein_pixels": vein_pixels,
                "background_pixels_internal_holes": background_pixels_internal_holes,
                "total_bbox_pixels": total_pixels_in_bbox,
                "bbox_min_row": min_row,
                "bbox_min_col": min_col,
                "bbox_max_row": max_row,
                "bbox_max_col": max_col,
                "mask_file": os.path.basename(mask_save_path),
                "rgb_crop_file": os.path.basename(rgb_crop_save_path),
                "overlay_file": os.path.basename(overlay_save_path)
            })
            
            global_component_serial_id += 1 # Increment global counter for the next component
            component_idx_in_original_image += 1 # Increment local counter for the next component in this image
        
        # Append components from the current image to the overall metadata list
        # and immediately save the updated CSV
        if current_image_components_metadata:
            all_component_metadata.extend(current_image_components_metadata)
            metadata_df = pd.DataFrame(all_component_metadata)
            # Overwrite the CSV each time with all accumulated data
            metadata_df.to_csv(METADATA_CSV_PATH, index=False)
            # print(f"  -> Metadata updated for {os.path.basename(original_image_full_path)}")


    print(f"\n✅ Prediction and component extraction complete! Results saved to '{OUTPUT_PREDICTIONS_DIR}' folder.")

if __name__ == "__main__":
    main()