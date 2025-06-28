import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys
import math

# --- NEW IMPORTS for Albumentations ---
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # Albumentations uses OpenCV

# NEW IMPORT for Euclidean Distance Transform
from scipy.ndimage import distance_transform_edt
from skimage.draw import line

# --- Configuration Constants ---
IMAGE_SIZE = (512, 512) # Assuming all images are already 512x512 (Width, Height)
HEATMAP_SIGMA = 15 # Standard deviation for Gaussian heatmap - DECREASED FROM 30
NUM_LANDMARKS = 2 # Base and Tip (for output heatmaps)

# Paths - assuming the script is run from the directory containing FINAL_ALIGNED_LEAVES_512x512
BASE_DIR = Path(".")
RGB_CROPS_DIR = BASE_DIR / "FINAL_ALIGNED_LEAVES_512x512" / "RGB_CROPS"
MASKS_DIR = BASE_DIR / "FINAL_ALIGNED_LEAVES_512x512" / "MASKS"
TRAINING_CSV_PATH = BASE_DIR / "training_data_results.csv"

# Output directory for saved model
OUTPUT_BASE_DIR = BASE_DIR # Save best_landmark_model.pth in the current working directory


# --- Helper Function: CSV Parsing ---
def parse_training_csv(csv_path):
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by=['Label', 'index']).reset_index(drop=True)
    landmark_data = {}
    grouped = df_sorted.groupby('Label')
    for label, group in grouped:
        if len(group) == 2:
            base_coords = (group.iloc[0]['X'], group.iloc[0]['Y'])
            tip_coords = (group.iloc[1]['X'], group.iloc[1]['Y'])
            landmark_data[label] = {'base': base_coords, 'tip': tip_coords}
        else:
            print(f"Warning: Skipping '{label}' - found {len(group)} rows, expected 2.")
    return landmark_data

# --- Helper Function: Generate Gaussian Heatmap ---
def generate_gaussian_heatmap(coords, img_size, sigma):
    heatmap = np.zeros((img_size[1], img_size[0]), dtype=np.float32) 
    x, y = float(coords[0]), float(coords[1]) # x is col, y is row
    sigma_f = float(sigma)

    y_grid, x_grid = np.ogrid[0:img_size[1], 0:img_size[0]]
    y_grid = y_grid.astype(np.float32)
    x_grid = x_grid.astype(np.float32)

    exponent = -((x_grid - x)**2 + (y_grid - y)**2) / (2.0 * sigma_f**2)
    heatmap = np.exp(exponent)
    
    if np.max(heatmap) > 0: # Only normalize if there's actual content
        heatmap = heatmap / np.max(heatmap)
    return heatmap

# --- Helper Function: Calculate Orthogonal Euclidean Distance from a Line ---
def calculate_line_distance_map(p1, p2, img_shape, mask_np):
    height, width = img_shape
    distance_map = np.zeros((height, width), dtype=np.float32)

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    if x1 == x2: # Vertical line
        distance_map = np.abs(np.arange(width) - x1).astype(np.float32)
        distance_map = np.tile(distance_map, (height, 1)) 
    elif y1 == y2: # Horizontal line
        distance_map = np.abs(np.arange(height) - y1).astype(np.float32)
        distance_map = np.tile(distance_map[:, np.newaxis], (1, width)) 
    else:
        A = y2 - y1
        B = x1 - x2
        C = -A * x1 - B * y1

        denom = np.sqrt(A**2 + B**2)
        if denom == 0:
            return distance_map 

        y_grid, x_grid = np.ogrid[0:height, 0:width]
        x_grid = x_grid.astype(np.float32)
        y_grid = y_grid.astype(np.float32)

        distance_map = np.abs(A * x_grid + B * y_grid + C) / denom
        distance_map = np.clip(distance_map, 0, np.max(img_shape) * 2) 

    distance_map = distance_map * (mask_np > 0) # Apply the mask

    max_val = np.max(distance_map)
    if max_val > 0:
        distance_map = distance_map / max_val
    
    return distance_map.astype(np.float32)

# --- PyTorch Dataset Class ---
class LeafLandmarkDataset(Dataset):
    def __init__(self, csv_path, rgb_dir, masks_dir, img_size, heatmap_sigma):
        self.rgb_dir = Path(rgb_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size # (Width, Height)
        self.heatmap_sigma = heatmap_sigma

        self.landmark_data = parse_training_csv(csv_path)
        self.image_files = list(self.landmark_data.keys())

        initial_count = len(self.image_files)
        valid_image_files = []
        for filename in self.image_files:
            rgb_path = self.rgb_dir / filename.replace('_overlay.png', '_rgb_crop.png')
            mask_path = self.masks_dir / filename.replace('_overlay.png', '_mask.png')
            if rgb_path.exists() and mask_path.exists():
                valid_image_files.append(filename)
            else:
                print(f"Warning: Skipping '{filename}' - missing RGB or Mask file.")
        self.image_files = valid_image_files
        print(f"Dataset initialized: {len(self.image_files)} valid samples out of {initial_count} initially parsed.")
        
        # --- Albumentations Transform Pipeline ---
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=360, p=1.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, crop_border=False),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- Debugging Flag ---
        DEBUG_VIZ = False # Set to False for actual training
        
        filename = self.image_files[idx]
        rgb_path = self.rgb_dir / filename.replace('_overlay.png', '_rgb_crop.png')
        mask_path = self.masks_dir / filename.replace('_overlay.png', '_mask.png')

        # Load images as NumPy arrays (Albumentations prefers NumPy)
        rgb_img_np = np.array(Image.open(rgb_path).convert("RGB")) # Shape (H, W, 3)
        mask_img_np = np.array(Image.open(mask_path).convert("L")) # Shape (H, W)

        original_landmarks_xy = self.landmark_data[filename]
        original_base_coords = original_landmarks_xy['base']
        original_tip_coords = original_landmarks_xy['tip']
        
        # Prepare keypoints for Albumentations
        keypoints = [
            original_base_coords,
            original_tip_coords
        ]

        if DEBUG_VIZ:
            rgb_img_original_viz = rgb_img_np.copy()
            mask_img_original_viz = mask_img_np.copy()
            gt_base_hm_original_viz = generate_gaussian_heatmap(original_base_coords, self.img_size, self.heatmap_sigma)
            gt_tip_hm_original_viz = generate_gaussian_heatmap(original_tip_coords, self.img_size, self.heatmap_sigma)

        # --- Apply Data Augmentation using Albumentations ---
        transformed = self.transform(image=rgb_img_np, mask=mask_img_np, keypoints=keypoints)
        
        rgb_img_tensor = transformed['image'] # Already normalized and ToTensorV2'd
        mask_img_tensor = transformed['mask'].float() # Ensure mask is float

        # --- FIX: Ensure mask_img_tensor has a channel dimension (1, H, W) ---
        if mask_img_tensor.dim() == 2:
            mask_img_tensor = mask_img_tensor.unsqueeze(0)
        # --- END FIX ---

        transformed_keypoints = transformed['keypoints'] # List of (x, y) tuples

        # Extract transformed coordinates
        transformed_base_coords = transformed_keypoints[0] 
        transformed_tip_coords = transformed_keypoints[1]

        # Convert mask to 0-1 float for distance maps
        mask_np_float = mask_img_tensor.squeeze(0).cpu().numpy() 

        img_h, img_w = self.img_size[1], self.img_size[0]

        # Generate Ground Truth Heatmaps (using transformed coordinates)
        base_heatmap = generate_gaussian_heatmap(transformed_base_coords, self.img_size, self.heatmap_sigma)
        tip_heatmap = generate_gaussian_heatmap(transformed_tip_coords, self.img_size, self.heatmap_sigma)
        gt_heatmaps = np.stack([base_heatmap, tip_heatmap], axis=0) 

        midrib_dist_map = calculate_line_distance_map(transformed_base_coords, transformed_tip_coords, (img_h, img_w), mask_np_float)
        
        # Midpoint of the transformed coordinates
        mid_x = (transformed_base_coords[0] + transformed_tip_coords[0]) / 2
        mid_y = (transformed_base_coords[1] + transformed_tip_coords[1]) / 2
        
        # Calculate a line perpendicular to the midrib, passing through its midpoint
        if (transformed_tip_coords[0] - transformed_base_coords[0]) == 0: # Vertical midrib line
            ortho_p1 = (0, mid_y)
            ortho_p2 = (img_w - 1, mid_y)
        else:
            m_midrib = (transformed_tip_coords[1] - transformed_base_coords[1]) / (transformed_tip_coords[0] - transformed_base_coords[0])
            if m_midrib == 0: 
                ortho_p1 = (mid_x, 0)
                ortho_p2 = (mid_x, img_h - 1)
            else:
                m_ortho = -1 / m_midrib

                # Extend the line beyond image bounds for distance calculation
                ortho_p1 = (mid_x - img_w, mid_y - img_w * m_ortho) 
                ortho_p2 = (mid_x + img_w, mid_y + img_w * m_ortho) 

        ortho_midpoint_dist_map = calculate_line_distance_map(ortho_p1, ortho_p2, (img_h, img_w), mask_np_float)

        rr, cc = np.ogrid[0:img_h, 0:img_w] 
        weight_map = np.ones_like(gt_heatmaps, dtype=np.float32) 
        
        # --- Custom weight map logic commented out for initial testing ---
        # dist_to_base = np.sqrt((rr - transformed_base_coords[1])**2 + (cc - transformed_base_coords[0])**2)
        # weight_map[0, dist_to_base < (2 * self.heatmap_sigma)] = 5.0
        
        # dist_to_tip = np.sqrt((rr - transformed_tip_coords[1])**2 + (cc - transformed_tip_coords[0])**2)
        # weight_map[1, dist_to_tip < (2 * self.heatmap_sigma)] = 5.0
        # --- End custom weight map logic ---

        weight_map_tensor = torch.from_numpy(weight_map)

        input_tensor = torch.cat(
            (
                rgb_img_tensor,
                mask_img_tensor, 
                torch.from_numpy(midrib_dist_map).unsqueeze(0),
                torch.from_numpy(ortho_midpoint_dist_map).unsqueeze(0)
            ),
            dim=0
        )

        gt_heatmaps_tensor = torch.from_numpy(gt_heatmaps)
        
        gt_coords_tensor = torch.tensor([
            [transformed_base_coords[0], transformed_base_coords[1]],
            [transformed_tip_coords[0], transformed_tip_coords[1]]
        ], dtype=torch.float32)

        if DEBUG_VIZ:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            axes[0, 0].imshow(rgb_img_original_viz)
            axes[0, 0].scatter(original_base_coords[0], original_base_coords[1], color='lime', s=100, marker='+', label='Original Base')
            axes[0, 0].scatter(original_tip_coords[0], original_tip_coords[1], color='cyan', s=100, marker='x', label='Original Tip')
            axes[0, 0].set_title("Original Image & Points")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(rgb_img_original_viz)
            axes[0, 1].imshow(gt_base_hm_original_viz, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[0, 1].imshow(gt_tip_hm_original_viz, cmap='hot', alpha=0.5, vmin=0, vmax=1)
            axes[0, 1].set_title("Original Heatmaps")
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(mask_img_original_viz, cmap='gray')
            axes[0, 2].set_title("Original Mask")
            axes[0, 2].axis('off')

            # Un-normalize and convert tensor back to numpy for display
            display_img = rgb_img_tensor.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            display_img = (display_img * std + mean) * 255
            display_img = np.clip(display_img, 0, 255).astype(np.uint8)

            axes[1, 0].imshow(display_img)
            axes[1, 0].scatter(transformed_base_coords[0], transformed_base_coords[1], color='red', s=100, marker='+', label='Transformed Base')
            axes[1, 0].scatter(transformed_tip_coords[0], transformed_tip_coords[1], color='yellow', s=100, marker='x', label='Transformed Tip')
            axes[1, 0].set_title(f"Transformed Image & Points (Albumentations)")
            axes[1, 0].axis('off')

            axes[1, 1].imshow(display_img)
            axes[1, 1].imshow(base_heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[1, 1].imshow(tip_heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
            axes[1, 1].set_title("Transformed Heatmaps")
            axes[1, 1].axis('off')

            axes[1, 2].imshow(mask_np_float, cmap='gray')
            axes[1, 2].set_title("Transformed Mask")
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.show()

            # --- NEW: Visualize ALL Input Channels for the model after transformation ---
            fig_inputs, axes_inputs = plt.subplots(2, 3, figsize=(18, 12)) 

            # Channel 0-2: RGB (un-normalized for display)
            axes_inputs[0, 0].imshow(display_img) # display_img is already un-normalized RGB
            axes_inputs[0, 0].set_title("Model Input: RGB (Channels 0-2)")
            axes_inputs[0, 0].axis('off')

            # Channel 3: Mask
            axes_inputs[0, 1].imshow(mask_img_tensor.squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes_inputs[0, 1].set_title("Model Input: Mask (Channel 3)")
            axes_inputs[0, 1].axis('off')

            # Channel 4: Midrib Distance Map
            axes_inputs[0, 2].imshow(midrib_dist_map, cmap='viridis', vmin=0, vmax=1)
            axes_inputs[0, 2].set_title("Model Input: Midrib Distance (Channel 4)")
            axes_inputs[0, 2].axis('off')

            # Channel 5: Orthogonal Midpoint Distance Map
            axes_inputs[1, 0].imshow(ortho_midpoint_dist_map, cmap='viridis', vmin=0, vmax=1)
            axes_inputs[1, 0].set_title("Model Input: Ortho Midpoint Dist (Channel 5)")
            axes_inputs[1, 0].axis('off')
            
            # Ground Truth Base Heatmap (Target 0)
            axes_inputs[1, 1].imshow(base_heatmap, cmap='jet', vmin=0, vmax=1)
            axes_inputs[1, 1].set_title("Ground Truth: Base Heatmap")
            axes_inputs[1, 1].axis('off')

            # Ground Truth Tip Heatmap (Target 1)
            axes_inputs[1, 2].imshow(tip_heatmap, cmap='hot', vmin=0, vmax=1)
            axes_inputs[1, 2].set_title("Ground Truth: Tip Heatmap")
            axes_inputs[1, 2].axis('off')

            plt.tight_layout()
            plt.show()

            # --- NEW: Visualize Weight Map ---
            fig_weights, axes_weights = plt.subplots(1, 2, figsize=(12, 6))
            axes_weights[0].imshow(weight_map[0], cmap='magma', vmin=1, vmax=5) # Assuming weights are 1 or 5
            axes_weights[0].set_title("Weight Map for Base Heatmap")
            axes_weights[0].axis('off')
            
            axes_weights[1].imshow(weight_map[1], cmap='magma', vmin=1, vmax=5) # Assuming weights are 1 or 5
            axes_weights[1].set_title("Weight Map for Tip Heatmap")
            axes_weights[1].axis('off')
            plt.tight_layout()
            plt.show()

            sys.exit(0) # Exit after showing one example in DEBUG_VIZ mode

        return input_tensor, gt_heatmaps_tensor, gt_coords_tensor, weight_map_tensor

# --- Model Architecture (Simple U-Net) ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if its size doesn't match x2 due to pooling/upsampling differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                 diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class LandmarkUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LandmarkUNet, self).__init__()
        self.n_channels = n_channels 
        self.n_classes = n_classes   

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024 + 512, 512) 
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        outputs = torch.sigmoid(logits)
        return outputs

# --- Evaluation Metric: Euclidean Distance Error (EDE) ---
def calculate_ede(pred_heatmaps, gt_coords_batch, img_size, device):
    ede_list = []
    
    for i in range(pred_heatmaps.shape[0]):
        for j in range(pred_heatmaps.shape[1]): # Iterate over each landmark's heatmap
            pred_heatmap = pred_heatmaps[i, j] # Shape (H, W)
            
            max_val = pred_heatmap.max()
            
            # DECREASED threshold for considering a peak meaningful
            min_peak_threshold = 0.001 # Reduced from 0.01
            
            if max_val < min_peak_threshold: 
                # If the max value is very low, consider it a failed prediction and default to center
                pred_x = float(img_size[0] / 2)
                pred_y = float(img_size[1] / 2)
            else:
                max_idx_flat = torch.argmax(pred_heatmap)
                pred_y = (max_idx_flat // img_size[0]).float() # row index
                pred_x = (max_idx_flat % img_size[0]).float() # col index
            
            pred_coord = torch.tensor([pred_x, pred_y], dtype=torch.float32, device=device)
            gt_coord = gt_coords_batch[i, j] 
            
            distance = torch.norm(pred_coord - gt_coord)
            ede_list.append(distance.item())
            
    return np.mean(ede_list) if ede_list else 0.0

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, img_size):
    best_val_loss = float('inf')
    best_val_ede = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_weighted_loss = 0.0
        for inputs, gt_heatmaps, _, weight_maps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            inputs = inputs.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            weight_maps = weight_maps.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) 

            raw_loss = criterion(outputs, gt_heatmaps)
            # Use raw_loss directly since weight_maps is now all ones
            weighted_loss = (raw_loss * weight_maps).mean() # Still multiply by all-ones weight_maps to keep code consistent

            weighted_loss.backward()
            optimizer.step()

            running_weighted_loss += weighted_loss.item() * inputs.size(0)

        epoch_weighted_loss = running_weighted_loss / len(train_loader.dataset)
        
        model.eval()
        val_running_weighted_loss = 0.0
        val_ede_list = []
        with torch.no_grad():
            for inputs, gt_heatmaps, gt_coords_batch, weight_maps in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                inputs = inputs.to(device)
                gt_heatmaps = gt_heatmaps.to(device)
                gt_coords_batch = gt_coords_batch.to(device)
                weight_maps = weight_maps.to(device)
                
                outputs = model(inputs)
                raw_loss = criterion(outputs, gt_heatmaps)
                weighted_loss = (raw_loss * weight_maps).mean()
                val_running_weighted_loss += weighted_loss.item() * inputs.size(0)

                val_ede_list.append(calculate_ede(outputs, gt_coords_batch, img_size, device))

        val_weighted_loss = val_running_weighted_loss / len(val_loader.dataset)
        avg_val_ede = np.mean(val_ede_list) if val_ede_list else 0.0

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_weighted_loss:.4f}, Val Loss: {val_weighted_loss:.4f}, Val EDE: {avg_val_ede:.2f} pixels")
        
        scheduler.step(val_weighted_loss)

        if avg_val_ede < best_val_ede:
            best_val_ede = avg_val_ede
            OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True) 
            torch.save(model.state_dict(), OUTPUT_BASE_DIR / "best_landmark_model.pth")
            print(f"Saved best model with Val EDE: {best_val_ede:.2f} pixels")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Training Configuration ---
    BATCH_SIZE = 8
    NUM_EPOCHS = 100 # Set to 100 epochs for actual training
    LEARNING_RATE = 1e-4 

    # --- Device Setup ---
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using MPS GPU for training.")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA GPU for training.")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU for training. Consider enabling MPS/CUDA for faster training.")

    # 1. Initialize Dataset
    print(f"Initializing dataset from {TRAINING_CSV_PATH}...")
    dataset = LeafLandmarkDataset(
        csv_path=TRAINING_CSV_PATH,
        rgb_dir=RGB_CROPS_DIR,
        masks_dir=MASKS_DIR,
        img_size=IMAGE_SIZE,
        heatmap_sigma=HEATMAP_SIGMA
    )

    if not dataset:
        print("No valid data found to create a dataset. Exiting.")
        sys.exit(1)

    # 2. Split Dataset into Training and Validation Sets
    print("Splitting dataset into training and validation sets...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE.type == 'mps' or DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE.type == 'mps' or DEVICE.type == 'cuda' else False
    )
    print("DataLoaders created.")

    # 4. Initialize Model, Loss, Optimizer
    print("Initializing model, loss function, and optimizer...")
    # Input channels: 3 (RGB) + 1 (Mask) + 1 (Midrib Dist) + 1 (Ortho Midpoint Dist) = 6
    model = LandmarkUNet(n_channels=6, n_classes=NUM_LANDMARKS).to(DEVICE)
    criterion = nn.MSELoss(reduction='none') 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, verbose=True)

    # 5. Train the Model
    print("\nStarting model training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE, IMAGE_SIZE)
    print("\nTraining complete!")
    print(f"Best model saved to: {OUTPUT_BASE_DIR / 'best_landmark_model.pth'}")

    # --- Optional: Visualize some predictions (for debugging/demonstration) ---
    import matplotlib.pyplot as plt 
    print("\n--- Visualizing some predictions ---")
    
    model_path = OUTPUT_BASE_DIR / "best_landmark_model.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}. Skipping visualization.")
        if NUM_EPOCHS == 0: # Only exit if we explicitly ran for viz (NUM_EPOCHS=0)
             sys.exit(0) 

    model.eval()
    num_display = 3 
    display_count = 0

    with torch.no_grad():
        for inputs, gt_heatmaps, gt_coords_batch, _ in val_loader: 
            if display_count >= num_display:
                break
            
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            for i in range(inputs.shape[0]):
                if display_count >= num_display:
                    break

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Un-normalize and convert tensor back to numpy for display
                rgb_img_display = inputs[i, :3].permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                rgb_img_display = (rgb_img_display * std + mean) * 255
                rgb_img_display = np.clip(rgb_img_display, 0, 255).astype(np.uint8)

                axes[0].imshow(rgb_img_display)
                axes[0].set_title("Input RGB Image")
                axes[0].axis('off')

                gt_base_hm = gt_heatmaps[i, 0].cpu().numpy()
                gt_tip_hm = gt_heatmaps[i, 1].cpu().numpy()
                
                gt_base_x_coord, gt_base_y_coord = gt_coords_batch[i, 0].cpu().numpy()
                gt_tip_x_coord, gt_tip_y_coord = gt_coords_batch[i, 1].cpu().numpy()

                axes[1].imshow(rgb_img_display)
                axes[1].imshow(gt_base_hm, cmap='jet', alpha=0.5, vmin=0, vmax=1)
                axes[1].imshow(gt_tip_hm, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                axes[1].scatter(gt_base_x_coord, gt_base_y_coord, color='lime', s=100, marker='+', label='GT Base')
                axes[1].scatter(gt_tip_x_coord, gt_tip_y_coord, color='cyan', s=100, marker='x', label='GT Tip')
                axes[1].set_title("Ground Truth Heatmaps & Points")
                axes[1].axis('off')

                # --- Extract predicted points using argmax method for visualization too ---
                pred_base_hm = outputs[i, 0] 
                pred_tip_hm = outputs[i, 1]

                # Apply the same logic as in calculate_ede for consistency
                max_val_base = pred_base_hm.max()
                min_peak_threshold_viz = 0.001 # Use same threshold as in calculate_ede
                if max_val_base < min_peak_threshold_viz: 
                    pred_base_x_coord_display = float(IMAGE_SIZE[0] / 2)
                    pred_base_y_coord_display = float(IMAGE_SIZE[1] / 2)
                else:
                    max_idx_flat_base = torch.argmax(pred_base_hm)
                    pred_base_y_coord_display = (max_idx_flat_base // IMAGE_SIZE[0]).float().item()
                    pred_base_x_coord_display = (max_idx_flat_base % IMAGE_SIZE[0]).float().item()

                max_val_tip = pred_tip_hm.max()
                if max_val_tip < min_peak_threshold_viz: # Use same threshold as in calculate_ede
                    pred_tip_x_coord_display = float(IMAGE_SIZE[0] / 2)
                    pred_tip_y_coord_display = float(IMAGE_SIZE[1] / 2)
                else:
                    max_idx_flat_tip = torch.argmax(pred_tip_hm)
                    pred_tip_y_coord_display = (max_idx_flat_tip // IMAGE_SIZE[0]).float().item()
                    pred_tip_x_coord_display = (max_idx_flat_tip % IMAGE_SIZE[0]).float().item()


                axes[2].imshow(rgb_img_display)
                axes[2].imshow(pred_base_hm.cpu().numpy(), cmap='jet', alpha=0.5, vmin=0, vmax=1)
                axes[2].imshow(pred_tip_hm.cpu().numpy(), cmap='hot', alpha=0.5, vmin=0, vmax=1)
                axes[2].scatter(pred_base_x_coord_display, pred_base_y_coord_display, color='red', s=100, marker='+', label='Pred Base (Argmax)')
                axes[2].scatter(pred_tip_x_coord_display, pred_tip_y_coord_display, color='yellow', s=100, marker='x', label='Pred Tip (Argmax)')
                axes[2].set_title("Predicted Heatmaps & Points (Argmax)")
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
                display_count += 1