# vinifera_training.py - Multi-task Learning for Segmentation and Geodesic Distance Prediction

import os
import torch
import torch.nn as nn
import torch.nn.functional as F # For interpolate
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as TF # For augmentations (if used)
from PIL import Image
import numpy as np
from tqdm import tqdm
import time # For timing epochs
import json # For saving config
import matplotlib.pyplot as plt # Needed for plotting at the end

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # NEW: For learning rate scheduling

# --- CONFIGURATION ---
# Base directory for all processed Vinifera data.
# This assumes your 11_CHANNEL_INPUTS, COMBINED_GROUND_TRUTH_MASKS, etc., are subdirectories here.
PROCESSED_DATA_ROOT = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA"

# Directories for your processed Vinifera data
ELEVEN_CHANNEL_DIR = os.path.join(PROCESSED_DATA_ROOT, "11_CHANNEL_INPUTS")
COMBINED_MASKS_DIR = os.path.join(PROCESSED_DATA_ROOT, "COMBINED_GROUND_TRUTH_MASKS")
GEODESIC_MASKS_DIR = os.path.join(PROCESSED_DATA_ROOT, "GEODESIC_MASKS")
CHECKPOINTS_DIR = os.path.join(PROCESSED_DATA_ROOT, "checkpoints_vinifera")

# Ensure all necessary directories exist
os.makedirs(COMBINED_MASKS_DIR, exist_ok=True)
os.makedirs(GEODESIC_MASKS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Pre-trained model path (if starting from a pre-existing model)
# Set to None if you want to train from scratch
PRETRAINED_MODEL_PATH = "V1_best_model_vein_dice_0.7697_epoch29.pt" # Example path
# PRETRAINED_MODEL_PATH = None # Set to None for initial training from scratch

# Training Hyperparameters
BATCH_SIZE = 1 # Keep batch size small for high-res images and memory is limited
LEARNING_RATE = 1e-5
NUM_EPOCHS = 50
VALIDATION_SPLIT_RATIO = 0.2 # 20% of data for validation
SAVE_EVERY_N_EPOCHS = 5 # Save model every N epochs
EARLY_STOPPING_PATIENCE = 10 # Stop if validation Vein Dice doesn't improve for this many epochs
NUM_WORKERS = 2

# NEW: L2 Regularization (Weight Decay)
WEIGHT_DECAY = 1e-4 # Common value, helps prevent overfitting

# NEW: Learning Rate Scheduler Parameters
LR_SCHEDULER_FACTOR = 0.5 # Factor by which the learning rate will be reduced
LR_SCHEDULER_PATIENCE = 5 # Number of epochs with no improvement after which learning rate will be reduced
MIN_LEARNING_RATE = 1e-7 # Minimum learning rate, training stops if it goes below this

# Class Labels (must match how masks are generated)
# 0: Background, 1: Blade, 2: Vein
NUM_SEG_CLASSES = 3 # Number of classes for segmentation
CLASS_LABELS = {0: "Background", 1: "Blade", 2: "Vein"}
# Class weights for CrossEntropyLoss (optional, adjust based on observed class imbalance)
# These will be moved to the DEVICE later
CLASS_WEIGHTS_RAW = [0.1, 1.0, 5.0] # Higher weight for Vein

# Multi-task Loss Weights
SEGMENTATION_LOSS_WEIGHT = 1.0 # Weight for CrossEntropyLoss
GEODESIC_LOSS_WEIGHT = 0.5    # Weight for MSELoss for geodesic distance. Adjust this!

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Save training configuration for future reference ---
TRAINING_CONFIG = {
    "PRETRAINED_MODEL_PATH": PRETRAINED_MODEL_PATH,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "VALIDATION_SPLIT_RATIO": VALIDATION_SPLIT_RATIO,
    "NUM_SEG_CLASSES": NUM_SEG_CLASSES,
    "CLASS_LABELS": CLASS_LABELS,
    "CLASS_WEIGHTS": CLASS_WEIGHTS_RAW, # Stored as list in JSON
    "SEGMENTATION_LOSS_WEIGHT": SEGMENTATION_LOSS_WEIGHT,
    "GEODESIC_LOSS_WEIGHT": GEODESIC_LOSS_WEIGHT,
    "DEVICE": str(DEVICE),
    "ELEVEN_CHANNEL_DIR": ELEVEN_CHANNEL_DIR,
    "COMBINED_MASKS_DIR": COMBINED_MASKS_DIR,
    "GEODESIC_MASKS_DIR": GEODESIC_MASKS_DIR,
    "WEIGHT_DECAY": WEIGHT_DECAY, # NEW
    "LR_SCHEDULER_FACTOR": LR_SCHEDULER_FACTOR, # NEW
    "LR_SCHEDULER_PATIENCE": LR_SCHEDULER_PATIENCE, # NEW
    "MIN_LEARNING_RATE": MIN_LEARNING_RATE # NEW
}
config_output_dir = os.path.join(PROCESSED_DATA_ROOT, "config")
os.makedirs(config_output_dir, exist_ok=True)
with open(os.path.join(config_output_dir, "training_config.json"), 'w') as f:
    json.dump(TRAINING_CONFIG, f, indent=4)
print(f"Training configuration saved to: {os.path.join(config_output_dir, 'training_config.json')}")


# =============== Dataset Definition ===============
# MultiChannelLeafDataset to load 11-channel input, combined segmentation mask, and geodesic distance map.
class MultiChannelLeafDataset(Dataset):
    def __init__(self, eleven_channel_dir, blade_masks_dir, vein_masks_dir, combined_masks_output_dir, geodesic_masks_dir, transform=None):
        self.eleven_channel_dir = eleven_channel_dir
        self.blade_masks_dir = blade_masks_dir
        self.vein_masks_dir = vein_masks_dir
        self.combined_masks_output_dir = combined_masks_output_dir
        self.geodesic_masks_dir = geodesic_masks_dir # NEW: Geodesic masks directory
        self.transform = transform
        
        self.fids = [f.replace(".npy", "") for f in os.listdir(eleven_channel_dir) if f.endswith(".npy")]
        
        if not self.fids:
            raise RuntimeError(f"No .npy files found in {eleven_channel_dir}. Please ensure preprocessing Stage 2 completed.")

        # Pre-generate combined segmentation masks if they don't exist
        self._generate_combined_segmentation_masks()

    def _generate_combined_segmentation_masks(self):
        print("Checking for or generating combined segmentation masks...")
        for fid in tqdm(self.fids, desc="Generating Combined Segmentation Masks"):
            combined_mask_path = os.path.join(self.combined_masks_output_dir, f"{fid}.png")
            if os.path.exists(combined_mask_path):
                continue # Skip if already exists

            blade_mask_npy_path = os.path.join(self.blade_masks_dir, f"{fid}_blade_mask.npy")
            vein_mask_npy_path = os.path.join(self.vein_masks_dir, f"{fid}_vein_mask.npy")

            if not os.path.exists(blade_mask_npy_path) or not os.path.exists(vein_mask_npy_path):
                print(f"WARNING: Missing blade or vein mask for {fid}. Cannot generate combined segmentation mask.")
                continue

            blade_mask = np.load(blade_mask_npy_path) # 0 for background, 1 for blade
            vein_mask = np.load(vein_mask_npy_path)    # 0 for background, 1 for vein
            
            combined_mask = np.zeros_like(blade_mask, dtype=np.uint8) # Start with background (0)
            combined_mask[blade_mask == 1] = 1 # Set blade pixels
            combined_mask[vein_mask == 1] = 2  # Set vein pixels (veins overwrite blade where they overlap)

            combined_mask_pil = Image.fromarray(combined_mask, mode='L')
            combined_mask_pil.save(combined_mask_path)
        print("Finished combined segmentation mask generation check.")

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, idx):
        fid = self.fids[idx]
        
        # Load 11-channel input (H, W, C)
        image_path = os.path.join(self.eleven_channel_dir, f"{fid}.npy")
        image = np.load(image_path).astype(np.float32) # Load as float32

        # Load combined segmentation mask (H, W)
        seg_mask_path = os.path.join(self.combined_masks_output_dir, f"{fid}.png")
        seg_mask = Image.open(seg_mask_path).convert('L') # Load as grayscale
        seg_mask = np.array(seg_mask).astype(np.int64) # Convert to numpy long integer array (for CrossEntropyLoss)

        # NEW: Load geodesic distance mask (H, W)
        # This mask has float values (0-1) for vein pixels, 0 for non-vein pixels.
        geodesic_mask_path = os.path.join(self.geodesic_masks_dir, f"{fid}_geodesic_mask.npy")
        if not os.path.exists(geodesic_mask_path):
            # Fallback if geodesic mask is missing, though preprocessing should create it.
            print(f"WARNING: Geodesic mask not found for {fid}. Using zeros.")
            geodesic_mask = np.zeros_like(seg_mask, dtype=np.float32)
        else:
            geodesic_mask = np.load(geodesic_mask_path).astype(np.float32)


        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1) # H, W, C -> C, H, W
        seg_mask_tensor = torch.from_numpy(seg_mask) # (H, W)
        geodesic_mask_tensor = torch.from_numpy(geodesic_mask) # (H, W)

        # Apply transformations if provided
        if self.transform:
            image_tensor, seg_mask_tensor, geodesic_mask_tensor = self.transform(
                image_tensor, seg_mask_tensor, geodesic_mask_tensor
            )
            
        # Return both segmentation mask and geodesic mask
        return image_tensor, seg_mask_tensor, geodesic_mask_tensor

# Data Augmentation (for training data) - Now applies to all three outputs
class DataAugmentation:
    def __init__(self):
        pass

    def __call__(self, img, seg_mask, geo_mask):
        # Random horizontal flip
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
            seg_mask = TF.hflip(seg_mask)
            geo_mask = TF.hflip(geo_mask)
        # Random vertical flip
        if np.random.rand() < 0.5:
            img = TF.vflip(img)
            seg_mask = TF.vflip(seg_mask)
            geo_mask = TF.vflip(geo_mask)
            
        return img, seg_mask, geo_mask

# ===================== U-Net Model Definition with Two Output Heads =====================
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

        # NEW: Output head for Geodesic Distance Regression
        # This will output a single channel (float 0-1) for geodesic distance
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

        # Output from segmentation head
        seg_output = self.seg_out_conv(d2)

        # NEW: Output from geodesic head
        geo_output = self.geo_out_conv(d2) # Shape: (N, 1, H, W)

        return seg_output, geo_output # Return both outputs

# ===================== Metrics (Adapted for Multi-class) =====================
def dice_coefficient(predictions, targets, num_classes, ignore_index=None):
    """
    Calculates the Dice Coefficient for each class.
    predictions: (N, C, H, W) where N is batch size, C is number of classes (logits)
    targets: (N, H, W) with class indices (long int)
    """
    preds_max = torch.argmax(predictions, dim=1)
    
    dice_scores = []
    
    for c in range(num_classes):
        if c == ignore_index:
            dice_scores.append(float('nan'))
            continue

        pred_mask = (preds_max == c).float()
        target_mask = (targets == c).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()

        if union == 0:
            if target_mask.sum() == 0 and pred_mask.sum() == 0:
                dice_scores.append(1.0)
            else:
                dice_scores.append(0.0)
        else:
            dice = (2. * intersection) / union
            dice_scores.append(dice.item())
            
    return dice_scores

# NEW: Metric for Geodesic Distance (Mean Absolute Error on Vein Pixels)
def geodesic_mae(pred_geodesic, target_geodesic, vein_target_mask):
    """
    Calculates Mean Absolute Error for geodesic distance, only considering vein pixels.
    pred_geodesic: (N, 1, H, W) predicted geodesic values
    target_geodesic: (N, H, W) ground truth geodesic values
    vein_target_mask: (N, H, W) binary mask for ground truth vein pixels (1 for vein, 0 for non-vein)
    """
    # Squeeze pred_geodesic to (N, H, W) for element-wise comparison
    pred_geodesic = pred_geodesic.squeeze(1) 
    
    # Calculate absolute difference
    abs_diff = torch.abs(pred_geodesic - target_geodesic)
    
    # Apply mask: only consider pixels where target_geodesic > 0 (i.e., vein pixels)
    # Or, more robustly, use the vein_target_mask derived from the segmentation mask.
    masked_diff = abs_diff * vein_target_mask.float()
    
    num_vein_pixels = vein_target_mask.sum().float()
    
    if num_vein_pixels == 0:
        return float('nan') # Avoid division by zero if no vein pixels are present
    
    mae = masked_diff.sum() / num_vein_pixels
    return mae.item()


# ===================== Training Loop =====================
def train_model(pretrained_model_path=None):
    # 1. Initialize Dataset and DataLoader
    print("Loading Vinifera dataset...")
    full_dataset = MultiChannelLeafDataset(
        eleven_channel_dir=ELEVEN_CHANNEL_DIR,
        blade_masks_dir=os.path.join(PROCESSED_DATA_ROOT, "BLADE_MASKS"),
        vein_masks_dir=os.path.join(PROCESSED_DATA_ROOT, "VEIN_MASKS"),
        combined_masks_output_dir=COMBINED_MASKS_DIR,
        geodesic_masks_dir=GEODESIC_MASKS_DIR,
        transform=DataAugmentation()
    )
    print(f"Found {len(full_dataset)} Vinifera images.")

    if len(full_dataset) < 2:
        print("Not enough images to perform a train-validation split. Need at least 2 images.")
        print("Consider adding more data or adjusting VALIDATION_SPLIT_RATIO if possible.")
        return

    total_size = len(full_dataset)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 2. Initialize the UNet Model with two output heads
    model = UNet(in_channels=11, num_seg_classes=NUM_SEG_CLASSES).to(DEVICE)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained model from: {pretrained_model_path}")
        state_dict = torch.load(pretrained_model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False) # strict=False handles new/missing keys
        print("Pre-trained model loaded successfully!")
        print("Note: If the loaded model was pre-multi-task, all layers are loaded. If single-task, "
              "the new geodesic output head will be initialized randomly.")
        for param in model.parameters():
            param.requires_grad = True
        print("All model parameters are unfrozen for fine-tuning.")
    else:
        print("Starting training from scratch (no pre-trained model loaded).")
        for param in model.parameters():
            param.requires_grad = True

    # 3. Define Loss Functions and Optimizer
    # NEW: Ensure CLASS_WEIGHTS is a tensor on the correct device
    seg_criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS_RAW, dtype=torch.float32).to(DEVICE))
    geo_criterion = nn.MSELoss(reduction='none') # Use reduction='none' to apply mask
    
    # NEW: Add weight_decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # NEW: Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=LR_SCHEDULER_FACTOR,
                                  patience=LR_SCHEDULER_PATIENCE, verbose=True, min_lr=MIN_LEARNING_RATE)

    # For tracking best model
    best_val_vein_dice = -1.0
    epochs_no_improve = 0
    
    history = {
        'train_total_loss': [], 'val_total_loss': [],
        'train_seg_loss': [], 'val_seg_loss': [],
        'train_geo_loss': [], 'val_geo_loss': [],
        'train_dice_background': [], 'train_dice_blade': [], 'train_dice_vein': [],
        'val_dice_background': [], 'val_dice_blade': [], 'val_dice_vein': [],
        'train_geodesic_mae': [], 'val_geodesic_mae': [],
        'learning_rate': [] # NEW: To track LR changes
    }

    print("\n--- Starting Vinifera Multi-task Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        
        # === Training Phase ===
        model.train()
        total_train_seg_loss = 0.0
        total_train_geo_loss = 0.0
        train_preds_list = []
        train_targets_list = []
        train_geo_preds_list = []
        train_geo_targets_list = []
        train_vein_masks_list = []

        for images, seg_masks, geo_masks in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Train)"):
            images, seg_masks, geo_masks = images.to(DEVICE), seg_masks.to(DEVICE), geo_masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            seg_outputs, geo_outputs = model(images) # Model now returns two outputs

            # Segmentation Loss
            seg_loss = seg_criterion(seg_outputs, seg_masks)

            # Geodesic Regression Loss
            vein_target_mask = (seg_masks == 2) # Boolean mask: True for vein, False otherwise
            
            # Ensure geo_outputs is (N, H, W) for element-wise multiplication with mask
            geo_outputs_squeezed = geo_outputs.squeeze(1) # Remove channel dimension
            
            # Calculate MSE only on vein pixels (where vein_target_mask is True)
            # Apply mask to both prediction and target to ensure only vein pixels contribute to loss
            masked_geo_loss = geo_criterion(geo_outputs_squeezed * vein_target_mask, geo_masks * vein_target_mask)
            
            # Sum up the losses for vein pixels and divide by the number of vein pixels
            # Handle cases where no vein pixels are present to avoid NaN/inf
            num_vein_pixels_in_batch = vein_target_mask.sum().float()
            if num_vein_pixels_in_batch > 0:
                geo_loss = masked_geo_loss.sum() / num_vein_pixels_in_batch
            else:
                geo_loss = torch.tensor(0.0, device=DEVICE) # No geodesic loss if no veins

            # Combine losses with weights
            total_loss = (SEGMENTATION_LOSS_WEIGHT * seg_loss) + (GEODESIC_LOSS_WEIGHT * geo_loss)
            
            total_loss.backward()
            optimizer.step()
            
            total_train_seg_loss += seg_loss.item() * images.size(0)
            total_train_geo_loss += geo_loss.item() * images.size(0) # Scale by batch size, though geo_loss is already an average if veins exist

            # Store predictions/targets for overall Dice and MAE calculation
            train_preds_list.append(seg_outputs.cpu())
            train_targets_list.append(seg_masks.cpu())
            train_geo_preds_list.append(geo_outputs.cpu())
            train_geo_targets_list.append(geo_masks.cpu())
            train_vein_masks_list.append(vein_target_mask.cpu())


        avg_train_seg_loss = total_train_seg_loss / len(train_loader.dataset)
        avg_train_geo_loss = total_train_geo_loss / len(train_loader.dataset)
        avg_train_total_loss = avg_train_seg_loss * SEGMENTATION_LOSS_WEIGHT + avg_train_geo_loss * GEODESIC_LOSS_WEIGHT

        # Calculate overall training Dice and MAE for this epoch
        all_train_seg_preds_tensor = torch.cat(train_preds_list, dim=0)
        all_train_seg_targets_tensor = torch.cat(train_targets_list, dim=0)
        train_dice_scores = dice_coefficient(all_train_seg_preds_tensor, all_train_seg_targets_tensor, NUM_SEG_CLASSES, ignore_index=0)
        
        all_train_geo_preds_tensor = torch.cat(train_geo_preds_list, dim=0)
        all_train_geo_targets_tensor = torch.cat(train_geo_targets_list, dim=0)
        all_train_vein_masks_tensor = torch.cat(train_vein_masks_list, dim=0)
        train_geodesic_mae_score = geodesic_mae(all_train_geo_preds_tensor, all_train_geo_targets_tensor, all_train_vein_masks_tensor)


        # === Validation Phase ===
        model.eval()
        val_seg_losses = []
        val_geo_losses = []
        all_val_seg_preds = []
        all_val_seg_targets = []
        all_val_geo_preds = []
        all_val_geo_targets = []
        all_val_vein_masks = []

        with torch.no_grad():
            for images, seg_masks, geo_masks in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Val)"):
                images, seg_masks, geo_masks = images.to(DEVICE), seg_masks.to(DEVICE), geo_masks.to(DEVICE)
                
                seg_outputs, geo_outputs = model(images)
                
                # Segmentation Loss
                val_seg_loss = seg_criterion(seg_outputs, seg_masks)
                val_seg_losses.append(val_seg_loss.item())

                # Geodesic Regression Loss
                val_vein_target_mask = (seg_masks == 2)
                val_geo_outputs_squeezed = geo_outputs.squeeze(1)
                val_masked_geo_loss = geo_criterion(val_geo_outputs_squeezed * val_vein_target_mask, geo_masks * val_vein_target_mask)
                num_val_vein_pixels_in_batch = val_vein_target_mask.sum().float()
                if num_val_vein_pixels_in_batch > 0:
                    val_geo_loss = val_masked_geo_loss.sum() / num_val_vein_pixels_in_batch
                else:
                    val_geo_loss = torch.tensor(0.0, device=DEVICE)
                val_geo_losses.append(val_geo_loss.item())

                all_val_seg_preds.append(seg_outputs.cpu())
                all_val_seg_targets.append(seg_masks.cpu())
                all_val_geo_preds.append(geo_outputs.cpu())
                all_val_geo_targets.append(geo_masks.cpu())
                all_val_vein_masks.append(val_vein_target_mask.cpu())


        avg_val_seg_loss = sum(val_seg_losses) / len(val_loader)
        avg_val_geo_loss = sum(val_geo_losses) / len(val_loader)
        avg_val_total_loss = avg_val_seg_loss * SEGMENTATION_LOSS_WEIGHT + avg_val_geo_loss * GEODESIC_LOSS_WEIGHT

        # Concatenate all predictions and targets for Dice and MAE calculation
        all_val_seg_preds_tensor = torch.cat(all_val_seg_preds, dim=0)
        all_val_seg_targets_tensor = torch.cat(all_val_seg_targets, dim=0)
        val_dice_scores = dice_coefficient(all_val_seg_preds_tensor, all_val_seg_targets_tensor, NUM_SEG_CLASSES, ignore_index=0)

        all_val_geo_preds_tensor = torch.cat(all_val_geo_preds, dim=0)
        all_val_geo_targets_tensor = torch.cat(all_val_geo_targets, dim=0)
        all_val_vein_masks_tensor = torch.cat(all_val_vein_masks, dim=0)
        val_geodesic_mae_score = geodesic_mae(all_val_geo_preds_tensor, all_val_geo_targets_tensor, all_val_vein_masks_tensor)

        # NEW: Step the learning rate scheduler based on validation vein Dice
        # We need to handle potential NaN for current_val_vein_dice if no veins are in validation set
        current_val_vein_dice = val_dice_scores[2] if not np.isnan(val_dice_scores[2]) else -1.0 # Default to -1.0 if NaN
        scheduler.step(current_val_vein_dice)
        
        # Record current learning rate
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Print detailed scores
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}") # Print current LR
        print(f"Train Total Loss: {avg_train_total_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, Geo: {avg_train_geo_loss:.4f})")
        print(f"Val Total Loss:   {avg_val_total_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, Geo: {avg_val_geo_loss:.4f})")
        
        for i, score in enumerate(train_dice_scores):
            if not np.isnan(score):
                print(f"  Train {CLASS_LABELS[i]} Dice: {score:.4f}")
        for i, score in enumerate(val_dice_scores):
            if not np.isnan(score):
                print(f"  Val {CLASS_LABELS[i]} Dice: {score:.4f}")
            else:
                print(f"  Val {CLASS_LABELS[i]} Dice: N/A (class not present or ignored)")
            
        print(f"  Train Geodesic MAE (Veins): {train_geodesic_mae_score:.4f}")
        print(f"  Val Geodesic MAE (Veins):   {val_geodesic_mae_score:.4f}")
        
        # Save history for plotting
        history['train_total_loss'].append(avg_train_total_loss)
        history['val_total_loss'].append(avg_val_total_loss)
        history['train_seg_loss'].append(avg_train_seg_loss)
        history['val_seg_loss'].append(avg_val_seg_loss)
        history['train_geo_loss'].append(avg_train_geo_loss)
        history['val_geo_loss'].append(avg_val_geo_loss)

        history['train_dice_background'].append(train_dice_scores[0] if not np.isnan(train_dice_scores[0]) else 0)
        history['train_dice_blade'].append(train_dice_scores[1] if not np.isnan(train_dice_scores[1]) else 0)
        history['train_dice_vein'].append(train_dice_scores[2] if not np.isnan(train_dice_scores[2]) else 0)
        history['val_dice_background'].append(val_dice_scores[0] if not np.isnan(val_dice_scores[0]) else 0)
        history['val_dice_blade'].append(val_dice_scores[1] if not np.isnan(val_dice_scores[1]) else 0)
        history['val_dice_vein'].append(val_dice_scores[2] if not np.isnan(val_dice_scores[2]) else 0)
        
        history['train_geodesic_mae'].append(train_geodesic_mae_score if not np.isnan(train_geodesic_mae_score) else 0)
        history['val_geodesic_mae'].append(val_geodesic_mae_score if not np.isnan(val_geodesic_mae_score) else 0)


        # Save best model based on Vein Dice Coefficient
        # Use current_val_vein_dice as calculated for scheduler, which handles NaN
        if current_val_vein_dice > best_val_vein_dice:
            best_val_vein_dice = current_val_vein_dice
            epochs_no_improve = 0
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"best_model_vein_dice_{best_val_vein_dice:.4f}_epoch{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🎉 Saved new best model checkpoint: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break
        
        # Save checkpoints periodically
        if (epoch) % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_epoch{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"-> Saved checkpoint for epoch {epoch}")
            
        # Check if learning rate has dropped to minimum, if so, stop training
        if optimizer.param_groups[0]['lr'] < MIN_LEARNING_RATE:
            print(f"Learning rate dropped below {MIN_LEARNING_RATE:.2e}. Stopping training.")
            break


    print("\n✅ Multi-task Training complete.")
    print(f"Best Vein Dice Coefficient on Validation Set: {best_val_vein_dice:.4f}")
    print(f"Model checkpoints saved to '{CHECKPOINTS_DIR}'")
    return history


# --- Main Execution ---
if __name__ == "__main__":
    # Create required combined mask directory if it doesn't exist
    os.makedirs(COMBINED_MASKS_DIR, exist_ok=True)
    # Ensure GEODESIC_MASKS_DIR exists
    os.makedirs(GEODESIC_MASKS_DIR, exist_ok=True)

    history = train_model(pretrained_model_path=PRETRAINED_MODEL_PATH)

    # Plot Training History
    plt.figure(figsize=(22, 10)) # Increased figure size for more plots

    # Plot Total Loss
    plt.subplot(2, 3, 1) # 2 rows, 3 columns, 1st plot
    plt.plot(history['train_total_loss'], label='Train Total Loss')
    plt.plot(history['val_total_loss'], label='Validation Total Loss')
    plt.title('Total Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Segmentation Loss
    plt.subplot(2, 3, 2) # 2 rows, 3 columns, 2nd plot
    plt.plot(history['train_seg_loss'], label='Train Seg. Loss')
    plt.plot(history['val_seg_loss'], label='Validation Seg. Loss')
    plt.title('Segmentation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Geodesic Regression Loss
    plt.subplot(2, 3, 3) # 2 rows, 3 columns, 3rd plot
    plt.plot(history['train_geo_loss'], label='Train Geodesic Loss (MSE)')
    plt.plot(history['val_geo_loss'], label='Validation Geodesic Loss (MSE)')
    plt.title('Geodesic Regression Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    # Plot Dice Scores
    plt.subplot(2, 3, 4) # 2 rows, 3 columns, 4th plot
    plt.plot(history['train_dice_background'], label='Train Background Dice', linestyle=':')
    plt.plot(history['val_dice_background'], label='Val Background Dice', linestyle=':')
    plt.plot(history['train_dice_blade'], label='Train Blade Dice')
    plt.plot(history['val_dice_blade'], label='Val Blade Dice')
    plt.plot(history['train_dice_vein'], label='Train Vein Dice', linewidth=2, color='green')
    plt.plot(history['val_dice_vein'], label='Val Vein Dice', linewidth=2, color='red')
    plt.title('Dice Scores over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot Geodesic MAE
    plt.subplot(2, 3, 5) # 2 rows, 3 columns, 5th plot
    plt.plot(history['train_geodesic_mae'], label='Train Geodesic MAE (Veins)', linestyle='--', color='purple')
    plt.plot(history['val_geodesic_mae'], label='Val Geodesic MAE (Veins)', linestyle='--', color='orange')
    plt.title('Geodesic MAE (Veins) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    # Plot Learning Rate
    plt.subplot(2, 3, 6) # 2 rows, 3 columns, 6th plot
    plt.plot(history['learning_rate'], label='Learning Rate', color='blue')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log') # Log scale is often useful for LR plots
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINTS_DIR, "training_history_vinifera_multi_task.png"))
    plt.show()
    print(f"Training history plot saved to: {os.path.join(CHECKPOINTS_DIR, 'training_history_vinifera_multi_task.png')}")