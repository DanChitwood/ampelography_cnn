# vinifera_transfer_learning.py

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

import torch.optim as optim # <-- Corrected: Added this import

# --- CONFIGURATION ---
# Directories for your processed Vinifera data
PROCESSED_DATA_ROOT = "PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA"
SEVEN_CHANNEL_DIR = os.path.join(PROCESSED_DATA_ROOT, "7_CHANNEL_INPUTS")
# IMPORTANT: Your Algerian setup used "GROUND_TRUTH_MASKS" which implies single file for all classes.
# The Vinifera preprocessing stage 2 created separate blade and vein masks.
# We need to adapt the dataset to create a combined mask like your Algerian setup.
# Let's assume you have a 'combined_masks' folder in PROCESSED_DATA_FOR_SEGMENTATION_VINIFERA.
# If not, the script will create them on the fly from blade/vein masks.
COMBINED_MASKS_DIR = os.path.join(PROCESSED_DATA_ROOT, "COMBINED_GROUND_TRUTH_MASKS") # New directory
os.makedirs(COMBINED_MASKS_DIR, exist_ok=True)

CHECKPOINTS_DIR = os.path.join(PROCESSED_DATA_ROOT, "checkpoints_vinifera")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Pre-trained model path
PRETRAINED_MODEL_PATH = "VINIFERA2_best_model_vein_dice_0.6564_epoch18.pt" # Ensure this path is correct

# Training Hyperparameters
BATCH_SIZE = 1 # Keep batch size small for high-res images and memory is limited
LEARNING_RATE = 1e-5 # Start with a lower learning rate for fine-tuning (tuned from 1e-4)
NUM_EPOCHS = 50 # Number of epochs for fine-tuning
VALIDATION_SPLIT_RATIO = 0.2 # 20% of data for validation
SAVE_EVERY_N_EPOCHS = 5 # Save model every N epochs
EARLY_STOPPING_PATIENCE = 10 # Stop if validation Dice doesn't improve for this many epochs
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0 # Use half CPU cores for DataLoader

# Class Labels (must match how masks are generated)
# 0: Background, 1: Blade, 2: Vein
NUM_CLASSES = 3
CLASS_LABELS = {0: "Background", 1: "Blade", 2: "Vein"}
# Class weights for CrossEntropyLoss (optional, adjust based on observed class imbalance)
# These are from your Algerian script.
CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float32)

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
    "NUM_CLASSES": NUM_CLASSES,
    "CLASS_LABELS": CLASS_LABELS,
    "CLASS_WEIGHTS": CLASS_WEIGHTS.tolist(), # Convert tensor to list for JSON
    "DEVICE": str(DEVICE),
    "SEVEN_CHANNEL_DIR": SEVEN_CHANNEL_DIR,
    "COMBINED_MASKS_DIR": COMBINED_MASKS_DIR
}
config_output_dir = os.path.join(PROCESSED_DATA_ROOT, "config")
os.makedirs(config_output_dir, exist_ok=True)
with open(os.path.join(config_output_dir, "training_config.json"), 'w') as f:
    json.dump(TRAINING_CONFIG, f, indent=4)
print(f"Training configuration saved to: {os.path.join(config_output_dir, 'training_config.json')}")


# =============== Dataset Definition ===============
# We need to adapt the MultiChannelLeafDataset to combine blade and vein masks into a single 3-class mask.
class MultiChannelLeafDataset(Dataset):
    def __init__(self, seven_channel_dir, blade_masks_dir, vein_masks_dir, combined_masks_output_dir, transform=None):
        self.seven_channel_dir = seven_channel_dir
        self.blade_masks_dir = blade_masks_dir
        self.vein_masks_dir = vein_masks_dir
        self.combined_masks_output_dir = combined_masks_output_dir # Where to save combined masks
        self.transform = transform
        
        self.fids = [f.replace(".npy", "") for f in os.listdir(seven_channel_dir) if f.endswith(".npy")]
        
        if not self.fids:
            raise RuntimeError(f"No .npy files found in {seven_channel_dir}. Please ensure preprocessing Stage 2 completed.")

        # Pre-generate combined masks if they don't exist
        self._generate_combined_masks()

    def _generate_combined_masks(self):
        print("Checking for or generating combined masks...")
        for fid in tqdm(self.fids, desc="Generating Combined Masks"):
            combined_mask_path = os.path.join(self.combined_masks_output_dir, f"{fid}.png")
            if os.path.exists(combined_mask_path):
                continue # Skip if already exists

            blade_mask_npy_path = os.path.join(self.blade_masks_dir, f"{fid}_blade_mask.npy")
            vein_mask_npy_path = os.path.join(self.vein_masks_dir, f"{fid}_vein_mask.npy")

            if not os.path.exists(blade_mask_npy_path) or not os.path.exists(vein_mask_npy_path):
                print(f"WARNING: Missing blade or vein mask for {fid}. Cannot generate combined mask.")
                continue

            blade_mask = np.load(blade_mask_npy_path) # 0 for background, 1 for blade
            vein_mask = np.load(vein_mask_npy_path)   # 0 for background, 1 for vein

            # Create combined mask:
            # 0: Background (where blade_mask is 0)
            # 1: Blade (where blade_mask is 1 and vein_mask is 0)
            # 2: Vein (where vein_mask is 1) - Veins take precedence over blade
            
            combined_mask = np.zeros_like(blade_mask, dtype=np.uint8) # Start with background (0)
            
            # Set blade pixels (where blade_mask is 1 and not a vein)
            combined_mask[blade_mask == 1] = 1 
            
            # Set vein pixels (where vein_mask is 1) - veins overwrite blade where they overlap
            combined_mask[vein_mask == 1] = 2 

            # Save as PNG
            combined_mask_pil = Image.fromarray(combined_mask, mode='L')
            combined_mask_pil.save(combined_mask_path)
        print("Finished combined mask generation check.")

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, idx):
        fid = self.fids[idx]
        
        # Load 7-channel input (H, W, C)
        image_path = os.path.join(self.seven_channel_dir, f"{fid}.npy")
        image = np.load(image_path).astype(np.float32) # Load as float32

        # Load combined mask (H, W)
        mask_path = os.path.join(self.combined_masks_output_dir, f"{fid}.png")
        mask = Image.open(mask_path).convert('L') # Load as grayscale
        mask = np.array(mask).astype(np.int64) # Convert to numpy long integer array (for CrossEntropyLoss)

        # Convert to PyTorch tensors
        # Images need to be (Channels, Height, Width)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1) # H, W, C -> C, H, W
        # Masks are (Height, Width) for CrossEntropyLoss
        mask_tensor = torch.from_numpy(mask)

        # Apply transformations if provided
        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

# Data Augmentation (for training data) - Copied from your Algerian script
class DataAugmentation:
    def __init__(self):
        pass

    def __call__(self, img, mask):
        # Random horizontal flip
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        # Random vertical flip
        if np.random.rand() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        
        # Add more transforms here if desired (e.g., random rotation, color jitter, affine)
        # Be careful with color jitter on 7-channel data, only apply to first 3 RGB channels
        
        return img, mask

# ===================== U-Net Model Definition (Copied directly from your script) =====================
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Helper function for Conv-BatchNorm-ReLU block
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
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512)) # Added another encoder block
        
        self.pool = nn.MaxPool2d(2)

        # Decoder path
        # Using ConvTranspose2d (upsampling)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(512, 256), CBR(256, 256)) # Input channels are 256 (from up4) + 256 (from enc3) = 512

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(256, 128), CBR(128, 128)) # Input channels are 128 (from up3) + 128 (from enc2) = 256

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(128, 64), CBR(64, 64)) # Input channels are 64 (from up2) + 64 (from enc1) = 128

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3)) # New encoder block

        # Decoder
        d4 = self.up4(e4)
        # Handle potential size mismatch for skip connections (due to odd dimensions or padding)
        if d4.shape != e3.shape:
             d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1) # Concatenate skip connection
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

# ===================== Metrics (Adapted for Multi-class) =====================
def dice_coefficient(predictions, targets, num_classes, ignore_index=None):
    """
    Calculates the Dice Coefficient for each class.
    predictions: (N, C, H, W) where N is batch size, C is number of classes (logits)
    targets: (N, H, W) with class indices (long int)
    """
    # Convert predictions to class indices (N, H, W)
    preds_max = torch.argmax(predictions, dim=1) 
    
    dice_scores = []
    
    for c in range(num_classes):
        if c == ignore_index:
            dice_scores.append(float('nan'))
            continue

        # Create binary masks for the current class
        pred_mask = (preds_max == c).float()
        target_mask = (targets == c).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()

        if union == 0:
            # If target class is not present and prediction also doesn't predict it, Dice is 1.0.
            # If target class is not present, but prediction predicts it (false positive), Dice is 0.0.
            if target_mask.sum() == 0 and pred_mask.sum() == 0:
                dice_scores.append(1.0)
            else:
                dice_scores.append(0.0) # Predicted where nothing exists
        else:
            dice = (2. * intersection) / union
            dice_scores.append(dice.item())
            
    return dice_scores # Returns a list of Dice scores for each class

# ===================== Training Loop =====================
def train_model():
    # 1. Initialize Dataset and DataLoader
    print("Loading Vinifera dataset...")
    # Your Vinifera preprocessing output blade_mask.npy and vein_mask.npy.
    # The Algerian model expects a single mask with 0/1/2 for background/blade/vein.
    # The MultiChannelLeafDataset will now generate these combined masks.
    full_dataset = MultiChannelLeafDataset(
        seven_channel_dir=SEVEN_CHANNEL_DIR,
        blade_masks_dir=os.path.join(PROCESSED_DATA_ROOT, "BLADE_MASKS"), # Path to blade masks from stage 2
        vein_masks_dir=os.path.join(PROCESSED_DATA_ROOT, "VEIN_MASKS"),   # Path to vein masks from stage 2
        combined_masks_output_dir=COMBINED_MASKS_DIR, # Where to save and then load combined masks
        transform=DataAugmentation() # Data augmentation for training
    )
    print(f"Found {len(full_dataset)} Vinifera images.")

    if len(full_dataset) < 2:
        print("Not enough images to perform a train-validation split. Need at least 2 images.")
        print("Consider adding more data or adjusting VALIDATION_SPLIT_RATIO if possible.")
        return # Use return instead of exit() in a function

    # Split dataset into training and validation
    total_size = len(full_dataset)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - val_size
    
    # Ensure reproducibility of split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) # <-- Corrected: Use val_loader consistently

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 2. Load the Pre-trained UNet Model
    model = UNet(in_channels=7, out_channels=NUM_CLASSES).to(DEVICE) # Ensure out_channels is NUM_CLASSES (3)

    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading pre-trained model from: {PRETRAINED_MODEL_PATH}")
        state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Pre-trained model loaded successfully!")
    else:
        print(f"Error: Pre-trained model not found at {PRETRAINED_MODEL_PATH}. Training from scratch or please provide correct path.")
        # Decide if you want to exit or continue training from scratch here
        # For transfer learning, we typically want to exit if pre-trained model is missing
        return
        
    # Optional: Freeze early layers for initial epochs (common in transfer learning)
    # This can help prevent catastrophic forgetting of learned features.
    # Example: Freeze all but the last few decoder layers and the output layer.
    # for name, param in model.named_parameters():
    #     if 'dec2' not in name and 'out_conv' not in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    # print("Frozen some early layers. Only deeper layers will be fine-tuned.")


    # 3. Define Loss Functions and Optimizer
    # CrossEntropyLoss directly expects raw logits from the model output.
    # CLASS_WEIGHTS must be on the same device as the model.
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE)) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # For tracking best model
    best_val_vein_dice = -1.0
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_dice_background': [], 'train_dice_blade': [], 'train_dice_vein': [],
               'val_dice_background': [], 'val_dice_blade': [], 'val_dice_vein': []}

    print("\n--- Starting Vinifera Transfer Learning ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        
        # === Training Phase ===
        model.train()
        total_loss = 0.0
        train_preds_list = []
        train_targets_list = []

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Train)"):
            images, masks = images.to(DEVICE), masks.to(DEVICE) # Masks are (N, H, W)
            optimizer.zero_grad()
            outputs = model(images) # Outputs are (N, C, H, W) raw logits
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0) # Accumulate batch loss scaled by batch size

            # Store predictions/targets for overall Dice calculation
            train_preds_list.append(outputs.cpu())
            train_targets_list.append(masks.cpu())

        avg_train_loss = total_loss / len(train_loader.dataset)
        
        # Calculate overall training Dice scores for this epoch
        all_train_preds_tensor = torch.cat(train_preds_list, dim=0)
        all_train_targets_tensor = torch.cat(train_targets_list, dim=0)
        train_dice_scores = dice_coefficient(all_train_preds_tensor, all_train_targets_tensor, NUM_CLASSES, ignore_index=0)


        # === Validation Phase ===
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Val)"): # <-- Corrected: Use val_loader
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_losses.append(loss.item())

                all_val_preds.append(outputs.cpu())
                all_val_targets.append(masks.cpu())

        avg_val_loss = sum(val_losses) / len(val_loader) # <-- Corrected: Use val_loader

        # Concatenate all predictions and targets for Dice calculation
        all_val_preds_tensor = torch.cat(all_val_preds, dim=0) # (N, C, H, W)
        all_val_targets_tensor = torch.cat(all_val_targets, dim=0) # (N, H, W)

        # Calculate Dice Coefficients
        val_dice_scores = dice_coefficient(all_val_preds_tensor, all_val_targets_tensor, NUM_CLASSES, ignore_index=0) # Ignore background for mean Dice

        # Print detailed scores
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        for i, score in enumerate(train_dice_scores):
            if not np.isnan(score):
                print(f"  Train {CLASS_LABELS[i]} Dice: {score:.4f}")
        for i, score in enumerate(val_dice_scores):
            if not np.isnan(score):
                print(f"  Val {CLASS_LABELS[i]} Dice: {score:.4f}")
            else:
                print(f"  Val {CLASS_LABELS[i]} Dice: N/A (class not present or ignored)")
        
        current_val_vein_dice = val_dice_scores[2] # Index 2 for 'Vein'

        # Save history for plotting
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice_background'].append(train_dice_scores[0] if not np.isnan(train_dice_scores[0]) else 0)
        history['train_dice_blade'].append(train_dice_scores[1] if not np.isnan(train_dice_scores[1]) else 0)
        history['train_dice_vein'].append(train_dice_scores[2] if not np.isnan(train_dice_scores[2]) else 0)
        history['val_dice_background'].append(val_dice_scores[0] if not np.isnan(val_dice_scores[0]) else 0)
        history['val_dice_blade'].append(val_dice_scores[1] if not np.isnan(val_dice_scores[1]) else 0)
        history['val_dice_vein'].append(val_dice_scores[2] if not np.isnan(val_dice_scores[2]) else 0)


        # Save best model based on Vein Dice Coefficient
        if not np.isnan(current_val_vein_dice) and current_val_vein_dice > best_val_vein_dice:
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

    print("\n✅ Training complete.")
    print(f"Best Vein Dice Coefficient on Validation Set: {best_val_vein_dice:.4f}")
    print(f"Model checkpoints saved to '{CHECKPOINTS_DIR}'")
    return history


# --- Main Execution ---
if __name__ == "__main__":
    # Create required combined mask directory if it doesn't exist
    os.makedirs(COMBINED_MASKS_DIR, exist_ok=True)

    history = train_model()

    # Plot Training History
    plt.figure(figsize=(15, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Dice Scores
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice_background'], label='Train Background Dice', linestyle=':')
    plt.plot(history['val_dice_background'], label='Val Background Dice', linestyle=':')
    plt.plot(history['train_dice_blade'], label='Train Blade Dice')
    plt.plot(history['val_dice_blade'], label='Val Blade Dice')
    plt.plot(history['train_dice_vein'], label='Train Vein Dice', linewidth=2, color='green')
    plt.plot(history['val_dice_vein'], label='Val Vein Dice', linewidth=2, color='red')
    plt.title('Dice Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINTS_DIR, "training_history_vinifera.png"))
    plt.show()
    print(f"Training history plot saved to: {os.path.join(CHECKPOINTS_DIR, 'training_history_vinifera.png')}")