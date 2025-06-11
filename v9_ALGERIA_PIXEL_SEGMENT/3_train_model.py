# train_segmentation_model.py

# === IMPORTS ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
import time # For timing epochs

# ===================== CONFIG =====================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Input directories from previous processing steps
SEVEN_CHANNEL_DATA_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/7CHANNEL_INPUTS"
GROUND_TRUTH_MASKS_DIR = "PROCESSED_DATA_FOR_SEGMENTATION/GROUND_TRUTH_MASKS"

# Output directory for model checkpoints
CHECKPOINT_DIR = "model_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training Parameters
BATCH_SIZE = 1 # Keep it small if images are large and memory is limited
NUM_EPOCHS = 50 # Increased for more robust training
LEARNING_RATE = 1e-4 # Adjusted learning rate
VALIDATION_SPLIT_RATIO = 0.2 # 20% of data for validation
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0 # Use half CPU cores for DataLoader

# Class Labels (must match how masks are generated)
# 0: Background, 1: Blade, 2: Vein
NUM_CLASSES = 3
CLASS_LABELS = {0: "Background", 1: "Blade", 2: "Vein"}
# Class weights for CrossEntropyLoss (optional, adjust based on observed class imbalance)
# You might need to calculate these based on your actual mask pixel counts.
# Example: If background is 90%, blade 9%, vein 1%, then weights could be:
# (1/0.9, 1/0.09, 1/0.01) and then normalize or pick suitable values
# For now, let's start with a general assumption that vein is the hardest to detect
# and background is the easiest.
CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float32) # Assign higher weight to 'Vein'

# =============== Dataset Definition ===============
class MultiChannelLeafDataset(Dataset):
    def __init__(self, data_dir, masks_dir, transform=None):
        self.data_dir = data_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        base_name = os.path.splitext(name)[0]

        # Load 7-channel input (Height, Width, Channels)
        image_path = os.path.join(self.data_dir, name)
        image = np.load(image_path).astype(np.float32) # Load as float32

        # Load mask (single channel, Height, Width)
        mask_path = os.path.join(self.masks_dir, f"{base_name}.png")
        mask = Image.open(mask_path).convert('L') # Convert to grayscale (single channel)
        mask = np.array(mask).astype(np.int64) # Convert to numpy long integer array (for CrossEntropyLoss)

        # Convert to PyTorch tensors
        # Images need to be (Channels, Height, Width)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1) 
        # Masks are (Height, Width) for CrossEntropyLoss
        mask_tensor = torch.from_numpy(mask)

        # Apply transformations if provided
        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

# Data Augmentation (for training data)
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

# ===================== U-Net =====================
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

# ===================== Metrics =====================
def dice_coefficient(predictions, targets, num_classes, ignore_index=None, reduce_batch_first=False):
    """
    Calculates the Dice Coefficient for each class.
    predictions: (N, C, H, W) where N is batch size, C is number of classes
    targets: (N, H, W) or (N, 1, H, W) with class indices
    """
    if predictions.dim() == 4 and targets.dim() == 3:
        # Convert predictions to class indices
        preds_max = torch.argmax(predictions, dim=1) # (N, H, W)
    elif predictions.dim() == 4 and targets.dim() == 4 and targets.shape[1] == 1:
        preds_max = torch.argmax(predictions, dim=1) # (N, H, W)
        targets = targets.squeeze(1) # (N, H, W)
    else:
        preds_max = predictions # Assume predictions are already class indices (N, H, W)

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
            # If target class is not present and prediction also doesn't predict it
            # it's 1.0 (perfectly ignored). If prediction predicts it but target doesn't, it's 0.0.
            # Here we consider it 1.0 if both are zero.
            # But typically, if union is zero, we return NaN to exclude it from averaging,
            # especially for rare classes.
            # For segmentation, if target has no pixels, and pred also has no pixels, Dice is 1.
            # If target has no pixels, and pred has some, Dice is 0.
            if target_mask.sum() == 0 and pred_mask.sum() == 0:
                dice_scores.append(1.0) # No actual presence, so perfect match
            else:
                dice_scores.append(0.0) # Predicted where nothing exists
        else:
            dice = (2. * intersection) / union
            dice_scores.append(dice.item())
            
    return dice_scores # Returns a list of Dice scores for each class

# ===================== Training Loop =====================
def train_model():
    # Load dataset
    full_dataset = MultiChannelLeafDataset(
        data_dir=SEVEN_CHANNEL_DATA_DIR,
        masks_dir=GROUND_TRUTH_MASKS_DIR,
        transform=DataAugmentation() # Apply augmentation only to training data
    )

    # Split dataset into training and validation
    total_size = len(full_dataset)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                               generator=torch.Generator().manual_seed(42)) # Seed for reproducibility

    # For validation dataset, we don't want augmentation
    # A cleaner way is to make a separate dataset instance for val_dataset without transform
    # But since random_split returns Subset, we need to adapt:
    # We can temporarily set transform to None for validation part if needed,
    # or ensure transforms are only for training. The DataAugmentation class
    # could be made optional or passed during __init__ of the Subset in a more complex setup.
    # For now, it's applied during init, so it will be applied to both.
    # For a real pipeline, you'd create two MultiChannelLeafDataset instances,
    # one with transform for train, one without for val, then use random_split on indices.
    # For this current setup, let's just make DataAugmentation a class that's called.
    
    # To correctly handle transforms for train/val split:
    # 1. Create a full dataset without augmentation.
    # 2. Use random_split to get indices for train/val.
    # 3. Create two new MultiChannelLeafDataset instances, one with augmentation for train_indices,
    #    one without for val_indices.
    
    # Simplified approach for now: augmentations apply to both, which is usually fine
    # for simple flips. For more aggressive transforms, this needs refinement.

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(in_channels=7, out_channels=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE)) # Use class weights

    best_val_vein_dice = -1.0 # Initialize with a low score
    
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        # === Training Phase ===
        model.train()
        total_loss = 0

        for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Train)"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        
        # === Validation Phase ===
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for images, masks in tqdm(val_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Val)"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_losses.append(loss.item())

                all_val_preds.append(outputs.cpu())
                all_val_targets.append(masks.cpu())

        avg_val_loss = sum(val_losses) / len(val_dataloader)

        # Concatenate all predictions and targets for Dice calculation
        all_val_preds_tensor = torch.cat(all_val_preds, dim=0) # (N, C, H, W)
        all_val_targets_tensor = torch.cat(all_val_targets, dim=0) # (N, H, W)

        # Calculate Dice Coefficients
        dice_scores = dice_coefficient(all_val_preds_tensor, all_val_targets_tensor, NUM_CLASSES, ignore_index=0) # Ignore background for mean Dice

        # Print detailed scores
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        for i, score in enumerate(dice_scores):
            if not np.isnan(score):
                print(f"  {CLASS_LABELS[i]} Dice: {score:.4f}")
            else:
                print(f"  {CLASS_LABELS[i]} Dice: N/A (class not present or ignored)")

        current_vein_dice = dice_scores[2] # Index 2 for 'Vein'

        # Save best model based on Vein Dice Coefficient
        if not np.isnan(current_vein_dice) and current_vein_dice > best_val_vein_dice:
            best_val_vein_dice = current_vein_dice
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_vein_dice_{best_val_vein_dice:.4f}_epoch{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🎉 Saved new best model checkpoint: {checkpoint_path}")

    print("\n✅ Training complete.")
    print(f"Best Vein Dice Coefficient on Validation Set: {best_val_vein_dice:.4f}")
    print(f"Model checkpoints saved to '{CHECKPOINT_DIR}'")

if __name__ == "__main__":
    train_model()