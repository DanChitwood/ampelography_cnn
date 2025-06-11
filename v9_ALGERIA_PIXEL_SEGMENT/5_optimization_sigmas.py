# optimize_sigmas.py

# === IMPORTS ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split # <--- ADD random_split here
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
from skimage import color, filters
import numpy as np
from tqdm import tqdm
import itertools # For generating combinations of sigmas
import csv # For logging results

# === CONFIGURATION ===
# Path to your OPTIMIZATION_SUBSET folder (e.g., where you copied selected RGBs and _mask.pngs)
OPTIMIZATION_SUBSET_DIR = "OPTIMIZATION_SUBSET" # <--- MAKE SURE THIS PATH IS CORRECT

# Output directory for optimization logs
OPTIMIZATION_LOGS_DIR = "OPTIMIZATION_LOGS"
os.makedirs(OPTIMIZATION_LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(OPTIMIZATION_LOGS_DIR, "sigma_optimization_results.csv")

# Image standardization parameters (MUST MATCH main_data_preprocessing.py)
# THESE SHOULD BE THE SAME TARGET_SIZE VALUES YOU USED FOR YOUR MAIN DATASET
TARGET_WIDTH = 2048 # <--- REPLACE WITH YOUR ACTUAL TARGET_WIDTH
TARGET_HEIGHT = 2040 # <--- REPLACE WITH YOUR ACTUAL TARGET_HEIGHT
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# Define the ranges of sigmas to test for each filter
# Start with a broad range, then narrow down based on results
SIGMA_RANGES = {
    'sato': [1.0, 2.0, 3.0],      # Example: Test 1.0, 2.0, 3.0
    'meijering': [1.0, 2.0, 3.0], # Example: Test 1.0, 2.0, 3.0
    'frangi': [0.5, 1.0, 1.5],    # Example: Test 0.5, 1.0, 1.5
    'hessian': [1.0, 2.0, 3.0]    # Example: Test 1.0, 2.0, 3.0
}
# A good starting point for sigmas could be [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# You can adjust these ranges based on your initial performance or knowledge of filter effects.
# More values means longer optimization time.

# Mini-training parameters for optimization
MINI_BATCH_SIZE = 1 # Keep batch size small for quick iterations
MINI_EPOCHS = 5      # Number of epochs for each sigma combination (e.g., 5-10)
LEARNING_RATE = 1e-4 # Use a small learning rate
VALIDATION_SPLIT_RATIO = 0.2 # Ratio of optimization subset for validation

# UNet Model Configuration (must match your trained model)
NUM_INPUT_CHANNELS = 7
NUM_CLASSES = 3 # 0: Background, 1: Blade, 2: Vein

# Dice Loss and Dice Metric (copied from train_segmentation_model.py)
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.weight = weight # Class weights

    def forward(self, inputs, targets, smooth=1):
        # inputs are logits from the model, apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)       
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Apply class weights if provided (for CrossEntropy, but can be adapted for Dice)
        # For Dice loss, it's typically better to incorporate weights into the per-class dice calculation
        # and then average, or use a weighted sum of per-class dice.
        # For simplicity here, we'll assume weights are handled by CrossEntropy and focus on Dice for metric.
        # If class weights were passed to DiceLoss, this part would need more complex logic.
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

def dice_coefficient(predictions, targets, num_classes=3, smooth=1e-6, ignore_background=False):
    """
    Calculates the Dice coefficient for each class.
    predictions: Tensor of shape (N, C, H, W) (logits)
    targets: Tensor of shape (N, H, W) (long integers representing class indices)
    """
    # Apply softmax to predictions to get probabilities, then argmax to get class indices
    predicted_classes = torch.argmax(predictions, dim=1) # (N, H, W)

    dice_scores = {}
    for class_id in range(num_classes):
        if ignore_background and class_id == 0: # Background
            dice_scores[class_id] = float('nan') # Not Applicable
            continue

        pred_mask = (predicted_classes == class_id).float()
        target_mask = (targets == class_id).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores[class_id] = dice.item()

    return dice_scores

# Device for training
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# =============== Helper Functions (Replicated from main_data_preprocessing.py) ===============
def rotate_to_wide(image_pil):
    width, height = image_pil.size
    if height > width:
        image_pil = image_pil.transpose(Image.Transpose.ROTATE_270)
    return image_pil

def rescale_and_pad_image(image_pil, target_size):
    original_width, original_height = image_pil.size
    target_width, target_height = target_size
    
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale_factor = min(scale_w, scale_h)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    scaled_img = image_pil.resize((new_width, new_height), Image.LANCZOS)

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # --- IMPORTANT CHANGE HERE ---
    # Check the mode of the input image and use it for the new padded image
    if image_pil.mode == 'L':
        # For masks (L mode), pad with 0 (black background for segmentation)
        padded_img = Image.new("L", target_size, 0)
    else:
        # For RGB images, pad with 255 (white background)
        padded_img = Image.new("RGB", target_size, (255, 255, 255))
    # --- END IMPORTANT CHANGE ---
        
    padded_img.paste(scaled_img, (paste_x, paste_y))
    
    return padded_img

def enhance_contrast(arr, percentile=99.9):
    vmax = np.percentile(arr, percentile)
    if vmax == 0: return np.zeros_like(arr)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled

def create_7channel_input(image_rgb_pil, sato_sigma, meijering_sigma, frangi_sigma, hessian_sigma):
    """Applies ridge filters with specified sigmas and stacks channels."""
    image_rgb_float = np.array(image_rgb_pil).astype(np.float32) / 255.0
    gray_image = color.rgb2gray(image_rgb_float)

    sato_raw = filters.sato(gray_image, sigmas=[sato_sigma], black_ridges=False)
    meijering_raw = filters.meijering(gray_image, sigmas=[meijering_sigma], black_ridges=False)
    frangi_raw = filters.frangi(gray_image, sigmas=[frangi_sigma], black_ridges=False)
    hessian_raw = filters.hessian(gray_image, sigmas=[hessian_sigma], black_ridges=True)

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

# =============== Dataset Class for Optimization ===============
class OptimizationDataset(Dataset):
    def __init__(self, root_dir, target_size, current_sigmas):
        self.root_dir = root_dir
        self.target_size = target_size
        self.current_sigmas = current_sigmas # Dictionary: {'sato': s, 'meijering': m, ...}

        # Find all RGB image files in the root_dir (assuming they are .png)
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".png") and not f.endswith("_mask.png")])
        
        if not self.image_files:
            raise FileNotFoundError(f"No RGB image files found in {root_dir}. Please ensure your OPTIMIZATION_SUBSET_DIR is correct and contains .png files.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        rgb_filename = self.image_files[idx]
        base_name = os.path.splitext(rgb_filename)[0]
        
        # Assuming mask filename is base_name_mask.png (as discussed and manually renamed)
        mask_filename = f"{base_name}_mask.png"

        rgb_path = os.path.join(self.root_dir, rgb_filename)
        mask_path = os.path.join(self.root_dir, mask_filename)

        # Load RGB image
        image_pil = Image.open(rgb_path).convert("RGB")

        # Load Mask (L mode for grayscale, pixel values 0, 1, 2)
        mask_pil = Image.open(mask_path).convert("L")

        # Preprocessing steps for both image and mask
        # 1. Rotate to wide (only apply to image, mask implicitly aligns)
        image_pil_rotated = rotate_to_wide(image_pil)
        # Note: Mask is a direct pixel map, no need to rotate its coordinates manually if
        # the mask itself was generated from transformed coordinates of rotated image.
        # For this optimization, we'll assume the masks are already aligned to the target orientation
        # (or will be rotated if we explicitly apply rotation to mask before saving for optimization).
        # Given the previous pipeline, the masks are already oriented correctly.

        # 2. Rescale and pad (apply to both image and mask)
        image_pil_padded = rescale_and_pad_image(image_pil_rotated, self.target_size)
        # Masks are also pixel maps, so they need the exact same scaling and padding
        # Apply the same logic to mask_pil.
        # To do this correctly, we need the original `rescale_and_pad_image` to return the size/paste_pos.
        # However, for simply scaling a mask, `ImageOps.contain` and then paste onto a blank image works too.
        
        # Re-using the image preprocessing logic for the mask (as it's a pixel map)
        mask_pil_rotated = rotate_to_wide(mask_pil) # Rotate mask too
        mask_pil_padded = rescale_and_pad_image(mask_pil_rotated, self.target_size) # Pad mask too

        # Create 7-channel input for image using current sigmas
        input_tensor = create_7channel_input(
            image_pil_padded,
            self.current_sigmas['sato'],
            self.current_sigmas['meijering'],
            self.current_sigmas['frangi'],
            self.current_sigmas['hessian']
        )
        
        # Convert mask to tensor (Long type for CrossEntropyLoss targets)
        target_tensor = torch.from_numpy(np.array(mask_pil_padded)).long()

        return input_tensor, target_tensor

# =============== UNet Model Definition (MUST BE IDENTICAL TO TRAINING) ===============
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
        self.dec2 = nn.Sequential(CBR(128, 64), CBR(64, 64)) # <--- Corrected line

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

# =============== Main Optimization Loop ===============
def run_sigma_optimization():
    print(f"Starting sigma optimization with ranges: {SIGMA_RANGES}")
    print(f"Mini-training for {MINI_EPOCHS} epochs per combination...")

    best_vein_dice = -1.0
    best_sigmas = None
    all_results = [] # To store results for CSV

    # Prepare CSV log file header
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sato_sigma', 'meijering_sigma', 'frangi_sigma', 'hessian_sigma', 'avg_val_loss', 'blade_dice', 'vein_dice'])

    # Generate all combinations of sigmas
    sigma_combinations = list(itertools.product(
        SIGMA_RANGES['sato'],
        SIGMA_RANGES['meijering'],
        SIGMA_RANGES['frangi'],
        SIGMA_RANGES['hessian']
    ))
    total_combinations = len(sigma_combinations)
    print(f"Total sigma combinations to test: {total_combinations}")

    for i, (sato_s, meijering_s, frangi_s, hessian_s) in enumerate(sigma_combinations):
        current_sigmas = {
            'sato': sato_s,
            'meijering': meijering_s,
            'frangi': frangi_s,
            'hessian': hessian_s
        }
        print(f"\n--- Testing Combination {i+1}/{total_combinations}: Sigmas {current_sigmas} ---")

        # Initialize dataset with current sigmas
        full_dataset = OptimizationDataset(
            root_dir=OPTIMIZATION_SUBSET_DIR,
            target_size=TARGET_SIZE,
            current_sigmas=current_sigmas
        )

        # Split into mini-train and mini-validation
        val_size = int(len(full_dataset) * VALIDATION_SPLIT_RATIO)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=MINI_BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)

        # Initialize a new model for each combination
        model = UNet(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(weight=None) # We can add class weights later if needed for optimization. For now, focus on sigmas.

        # Mini-training loop
        for epoch in range(MINI_EPOCHS):
            model.train()
            train_loss_sum = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            
            avg_train_loss = train_loss_sum / len(train_loader)

            # Mini-validation loop
            model.eval()
            val_loss_sum = 0
            all_predictions = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss_sum += loss.item()
                    all_predictions.append(outputs)
                    all_targets.append(targets)
            
            avg_val_loss = val_loss_sum / len(val_loader)
            
            # Concatenate predictions and targets for Dice calculation
            # Handle empty validation loader case
            if len(all_predictions) > 0:
                val_predictions = torch.cat(all_predictions, dim=0)
                val_targets = torch.cat(all_targets, dim=0)
                dice_scores = dice_coefficient(val_predictions, val_targets, num_classes=NUM_CLASSES, ignore_background=True)
            else:
                dice_scores = {1: float('nan'), 2: float('nan')} # No validation data

            blade_dice = dice_scores.get(1, float('nan'))
            vein_dice = dice_scores.get(2, float('nan'))

            print(f"  Epoch {epoch+1}/{MINI_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Blade Dice: {blade_dice:.4f}, Vein Dice: {vein_dice:.4f}")

        # After mini-training, record results and check if this is the best
        # Use the final validation Dice for comparison
        if not np.isnan(vein_dice) and vein_dice > best_vein_dice:
            best_vein_dice = vein_dice
            best_sigmas = current_sigmas
        
        # Log to CSV
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                sato_s, meijering_s, frangi_s, hessian_s,
                f"{avg_val_loss:.4f}", f"{blade_dice:.4f}", f"{vein_dice:.4f}"
            ])

    print("\n" + "="*50)
    print("✨ Optimization Complete! ✨")
    print(f"Best Vein Dice found: {best_vein_dice:.4f}")
    print(f"Corresponding Sigmas: {best_sigmas}")
    print(f"Detailed results logged to: {LOG_FILE}")
    print("="*50)


if __name__ == "__main__":
    run_sigma_optimization()