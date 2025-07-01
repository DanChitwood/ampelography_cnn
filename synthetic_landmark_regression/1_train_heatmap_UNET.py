import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm.auto import tqdm

# --- Configuration Parameters (consistent with synthetic data generation) ---
IMAGE_SIZE = (256, 256) # Output size for all images (masks, ECT, RGB)
SYNTHETIC_DATA_OUTPUT_DIR = Path("synthetic_leaf_data/")
SYNTHETIC_METADATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"

# --- Training Parameters (Adjust these!) ---
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
VALIDATION_SPLIT_RATIO = 0.2 # 20% of data for validation
NUM_WORKERS = 0 # Set to 0 for initial stability, adjust higher for faster data loading if system allows
MODEL_SAVE_PATH = Path("trained_leaf_heatmap_prediction_model.pth") # Path to save the best model

# --- New Configuration for Heatmaps ---
HEATMAP_SIGMA = 5 # Standard deviation for Gaussian heatmap, in pixels
HEATMAP_PEAK_VALUE = 1.0 # Max value of the heatmap at the peak location

# --- BOUND_RADIUS from 0_generate_synthetic_leaves.py (needed for coordinate conversion) ---
# This value defines the extent of the ECT space, typically ECT coordinates are in [-BOUND_RADIUS, BOUND_RADIUS]
BOUND_RADIUS = 1.0 

# --- Device Configuration ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# --- Helper functions for ECT coords to pixels and heatmap generation ---
def ect_coords_to_single_pixel(coords_ect: np.ndarray, image_size: tuple, bound_radius: float):
    """
    Helper to convert a single (y_ect, x_ect) ECT coordinate (as stored in metadata)
    to (pixel_x, pixel_y) in image space.
    """
    if np.isnan(coords_ect).any():
        return None # Indicate invalid coordinate

    # ECT coords are typically stored as (y_ect, x_ect) in the metadata (or row['base_y'], row['base_x'])
    # The ECT space has Y-axis pointing up, X-axis pointing right, origin at center.
    # Image space has Y-axis pointing down, X-axis pointing right, origin at top-left.
    display_y_conceptual = coords_ect[0] # Y from ECT
    display_x_conceptual = coords_ect[1] # X from ECT

    # Calculate scale factor and offsets to map [-bound_radius, bound_radius] to [0, image_size]
    scale_factor = image_size[0] / (2 * bound_radius) # Assumes square image, scales range 2*bound_radius to image_size
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2

    # Apply transformation and cast to integer pixels
    pixel_x = int(display_x_conceptual * scale_factor + offset_x)
    pixel_y = int(-display_y_conceptual * scale_factor + offset_y) # Y-axis inversion for image coords

    # Clamp to image boundaries to prevent out-of-bounds errors for edge cases
    pixel_x = np.clip(pixel_x, 0, image_size[0] - 1)
    pixel_y = np.clip(pixel_y, 0, image_size[1] - 1)

    return (pixel_x, pixel_y) # Return as (x_pixel, y_pixel) tuple for heatmap generation

def generate_gaussian_heatmap(center_pixel: tuple, image_size: tuple, sigma: float, peak_value: float):
    """Generates a 2D Gaussian heatmap centered at center_pixel (x_pixel, y_pixel)."""
    if center_pixel is None:
        return np.zeros(image_size, dtype=np.float32)

    # Create coordinate grids
    x = np.arange(0, image_size[1], 1, dtype=np.float32) # Columns (width)
    y = np.arange(0, image_size[0], 1, dtype=np.float32)[:, np.newaxis] # Rows (height)

    x0, y0 = center_pixel # (x_pixel, y_pixel) from ect_coords_to_single_pixel

    # Gaussian formula: A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
    heatmap = peak_value * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    return heatmap

# --- 1. Custom Dataset Class ---
class SyntheticLeafDataset(Dataset):
    def __init__(self, metadata_file: Path, base_dir: Path, transform=None):
        self.metadata_df = pd.read_csv(metadata_file)
        # Filter for successfully processed samples that also have valid landmark coordinates
        self.metadata_df = self.metadata_df[
            self.metadata_df['is_processed_valid'] &
            self.metadata_df['base_x'].notna() & self.metadata_df['base_y'].notna() &
            self.metadata_df['tip_x'].notna() & self.metadata_df['tip_y'].notna()
        ].reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

        if self.metadata_df.empty:
            raise ValueError(f"No valid processed samples with complete landmark data found in metadata file: {metadata_file}")
            
        print(f"Loaded {len(self.metadata_df)} valid synthetic samples for training with landmark data.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        
        # Load Blade ECT (Input 1)
        blade_ect_path = self.base_dir / row['file_blade_ect']
        blade_ect_img = Image.open(blade_ect_path).convert("L") # Ensure grayscale
        
        # Load Blade Mask (Input 2)
        blade_mask_path = self.base_dir / row['file_blade_mask']
        blade_mask_img = Image.open(blade_mask_path).convert("L") # Ensure grayscale

        # Convert to numpy arrays, normalize
        blade_ect_np = np.array(blade_ect_img) / 255.0 # Normalize to [0, 1]
        blade_mask_np = np.array(blade_mask_img) / 255.0 # Normalize to [0, 1]

        # Stack inputs along a new channel dimension (2, H, W) for PyTorch
        inputs_stacked = np.stack([blade_ect_np, blade_mask_np], axis=0)
        inputs_tensor = torch.from_numpy(inputs_stacked).float()

        # --- Generate Heatmap Targets for Base and Tip ---
        # Extract base and tip coordinates (already in blade ECT space as per generation script)
        # Note: metadata stores (x, y), but ect_coords_to_single_pixel expects (y_ect, x_ect)
        base_coords_ect_from_metadata = np.array([row['base_y'], row['base_x']])
        tip_coords_ect_from_metadata = np.array([row['tip_y'], row['tip_x']])

        base_pixel = ect_coords_to_single_pixel(base_coords_ect_from_metadata, IMAGE_SIZE, BOUND_RADIUS)
        tip_pixel = ect_coords_to_single_pixel(tip_coords_ect_from_metadata, IMAGE_SIZE, BOUND_RADIUS)

        base_heatmap = generate_gaussian_heatmap(base_pixel, IMAGE_SIZE, HEATMAP_SIGMA, HEATMAP_PEAK_VALUE)
        tip_heatmap = generate_gaussian_heatmap(tip_pixel, IMAGE_SIZE, HEATMAP_SIGMA, HEATMAP_PEAK_VALUE)

        # Stack heatmaps: channel 0 for base, channel 1 for tip (Shape: (2, H, W))
        target_heatmaps = np.stack([base_heatmap, tip_heatmap], axis=0) 
        target_tensor = torch.from_numpy(target_heatmaps).float()

        # --- Future Improvement: Data Augmentation ---
        # Augmentation logic should be applied here. For heatmaps, transformations
        # (rotation, scaling, translation) must be applied identically to inputs and heatmaps.
        if self.transform:
             pass # Placeholder for actual transformations

        return inputs_tensor, target_tensor

# --- 2. Model Architecture: U-Net (remains the same structure) ---
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
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels) 
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2's spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1) # Concatenate along channel dimension
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // (2 if bilinear else 1)) 
        
        self.up1 = Up(1024, 512 // (2 if bilinear else 1), bilinear)
        self.up2 = Up(512, 256 // (2 if bilinear else 1), bilinear)
        self.up3 = Up(256, 128 // (2 if bilinear else 1), bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes_out)

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
        return logits

# --- 3. Loss Function: Mean Squared Error (MSELoss) for Heatmaps ---
# Removed DiceLoss and CombinedLoss, as they are for binary segmentation.
# MSE is standard for regression tasks like heatmap prediction.
# The `criterion` object will be an instance of `nn.MSELoss()`.

# --- Helper for EDE Metric ---
def get_peak_coordinates(heatmap: np.ndarray):
    """
    Finds the pixel coordinates (x, y) of the peak of a heatmap.
    """
    # np.unravel_index returns (row_idx, col_idx) which corresponds to (y, x)
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return x, y # Return as (x, y) for consistency with plotting/Euclidean distance

def calculate_ede(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Calculates the Expected Distance Error (EDE) for a batch of predicted heatmaps.
    predictions: (B, 2, H, W) tensor of predicted heatmaps (raw output from model)
    targets: (B, 2, H, W) tensor of ground truth heatmaps
    Returns: Average EDE across the batch and both landmarks (base/tip) in pixels.
    """
    batch_size = predictions.size(0)
    total_ede_sum = 0.0
    num_landmarks = 2 # Base and Tip

    for i in range(batch_size):
        # Base heatmap
        pred_base_heatmap = predictions[i, 0].cpu().numpy()
        gt_base_heatmap = targets[i, 0].cpu().numpy()

        pred_base_x, pred_base_y = get_peak_coordinates(pred_base_heatmap)
        gt_base_x, gt_base_y = get_peak_coordinates(gt_base_heatmap)

        base_distance = np.sqrt((pred_base_x - gt_base_x)**2 + (pred_base_y - gt_base_y)**2)
        total_ede_sum += base_distance

        # Tip heatmap
        pred_tip_heatmap = predictions[i, 1].cpu().numpy()
        gt_tip_heatmap = targets[i, 1].cpu().numpy()

        pred_tip_x, pred_tip_y = get_peak_coordinates(pred_tip_heatmap)
        gt_tip_x, gt_tip_y = get_peak_coordinates(gt_tip_heatmap)

        tip_distance = np.sqrt((pred_tip_x - gt_tip_x)**2 + (pred_tip_y - gt_tip_y)**2)
        total_ede_sum += tip_distance
    
    # Divide by (batch_size * num_landmarks) for the average EDE per landmark
    return total_ede_sum / (batch_size * num_landmarks)


# --- Training and Validation Functions ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_PATH):
    best_val_ede = float('inf') # Initialize with a very high EDE (lower is better)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # Training phase
        model.train()
        running_train_loss = 0.0
        train_loader_tqdm = tqdm(dataloaders['train'], desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_loader_tqdm):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, targets) # MSE Loss

                loss.backward()
                optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(batch_loss=f"{loss.item():.6f}") # Increased precision for loss display

        epoch_train_loss = running_train_loss / len(dataloaders['train'].dataset)
        print(f"Train Loss: {epoch_train_loss:.6f}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        running_val_ede = 0.0
        val_loader_tqdm = tqdm(dataloaders['val'], desc=f"Validation Epoch {epoch+1}/{num_epochs}")
        with torch.no_grad():
            for inputs, targets in val_loader_tqdm:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, targets) # MSE Loss for validation
                
                # Calculate EDE
                ede = calculate_ede(outputs, targets) # Pass raw outputs for EDE calculation

                running_val_loss += loss.item() * inputs.size(0)
                running_val_ede += ede * inputs.size(0) # Accumulate EDE weighted by batch size
                val_loader_tqdm.set_postfix(batch_loss=f"{loss.item():.6f}", batch_ede=f"{ede:.2f}")


        epoch_val_loss = running_val_loss / len(dataloaders['val'].dataset)
        epoch_val_ede = running_val_ede / len(dataloaders['val'].dataset)
        print(f"Validation Loss: {epoch_val_loss:.6f}, Validation EDE: {epoch_val_ede:.2f} pixels")

        # Save the best model based on validation EDE (lower EDE is better)
        if epoch_val_ede < best_val_ede:
            best_val_ede = epoch_val_ede
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path} with Validation EDE: {best_val_ede:.2f} pixels")
            
        print() # Newline for readability

    print("Training complete!")
    return model

# --- Main Execution ---
if __name__ == "__main__":
    # Create dataset
    try:
        dataset = SyntheticLeafDataset(SYNTHETIC_METADATA_FILE, SYNTHETIC_DATA_OUTPUT_DIR)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        print("Please ensure the synthetic data generation script has been run successfully and generated valid landmark data.")
        sys.exit(1)

    # Split dataset into training and validation
    train_size = int((1 - VALIDATION_SPLIT_RATIO) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    }

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    # n_channels_in = 2 (Blade ECT + Blade Mask)
    # n_classes_out = 2 (Base Heatmap, Tip Heatmap)
    model = UNet(n_channels_in=2, n_classes_out=2).to(DEVICE)

    # Loss function and optimizer
    criterion = nn.MSELoss() # Changed to MSELoss for heatmap regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    trained_model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH)

    print("\n--- Model Training Finished ---")

    # --- Example Inference (using the best saved model) ---
    print("\n--- Example Inference ---")
    if MODEL_SAVE_PATH.exists():
        # Load the best model's state
        inference_model = UNet(n_channels_in=2, n_classes_out=2).to(DEVICE)
        inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        inference_model.eval() # Set to evaluation mode

        # Take one sample from the validation set for demonstration
        if len(val_dataset) > 0:
            sample_inputs, sample_target_heatmaps = val_dataset[0]
            
            # Add batch dimension and move to device
            sample_inputs = sample_inputs.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                predicted_heatmaps = inference_model(sample_inputs)
                # No sigmoid/thresholding needed for heatmaps, they are direct regression outputs
            
            # Move tensors back to CPU for plotting
            sample_inputs_cpu = sample_inputs.squeeze(0).cpu() # Remove batch dim
            blade_ect_viz = sample_inputs_cpu[0].numpy()
            blade_mask_viz = sample_inputs_cpu[1].numpy()
            
            # Ground truth heatmaps
            gt_base_heatmap_viz = sample_target_heatmaps[0].squeeze(0).cpu().numpy()
            gt_tip_heatmap_viz = sample_target_heatmaps[1].squeeze(0).cpu().numpy()

            # Predicted heatmaps
            # Ensure these are numpy arrays for get_peak_coordinates
            pred_base_heatmap_viz = predicted_heatmaps[0, 0].cpu().numpy()
            pred_tip_heatmap_viz = predicted_heatmaps[0, 1].cpu().numpy()

            # Find predicted pixel coordinates from heatmaps (peak location)
            pred_base_pixel_x, pred_base_pixel_y = get_peak_coordinates(pred_base_heatmap_viz)
            pred_tip_pixel_x, pred_tip_pixel_y = get_peak_coordinates(pred_tip_heatmap_viz)

            # Find ground truth pixel coordinates from heatmaps (for comparison)
            gt_base_pixel_x, gt_base_pixel_y = get_peak_coordinates(gt_base_heatmap_viz)
            gt_tip_pixel_x, gt_tip_pixel_y = get_peak_coordinates(gt_tip_heatmap_viz)

            # Plotting
            fig, axes = plt.subplots(2, 3, figsize=(18, 12)) # Rows: GT, Pred; Cols: Input1, Input2, Base, Tip

            # Input Images
            axes[0, 0].imshow(blade_ect_viz, cmap='gray')
            axes[0, 0].set_title("Input: Blade ECT")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(blade_mask_viz, cmap='gray')
            axes[0, 1].set_title("Input: Blade Mask")
            axes[0, 1].axis('off')

            # Ground Truth Heatmaps
            axes[0, 2].imshow(gt_base_heatmap_viz, cmap='viridis')
            axes[0, 2].set_title(f"GT Base Heatmap (Pix: ({gt_base_pixel_x},{gt_base_pixel_y}))")
            axes[0, 2].axis('off')
            # Overlay GT base point on Blade ECT for context
            axes[0, 0].scatter(gt_base_pixel_x, gt_base_pixel_y, c='red', marker='o', s=100, label='GT Base', edgecolors='white')
            axes[0, 0].scatter(gt_tip_pixel_x, gt_tip_pixel_y, c='blue', marker='o', s=100, label='GT Tip', edgecolors='white')
            axes[0,0].legend(loc='upper right')


            # Predicted Heatmaps
            axes[1, 0].imshow(pred_base_heatmap_viz, cmap='viridis')
            axes[1, 0].set_title(f"Predicted Base Heatmap (Pix: ({pred_base_pixel_x},{pred_base_pixel_y}))")
            axes[1, 0].axis('off')

            axes[1, 1].imshow(pred_tip_heatmap_viz, cmap='viridis')
            axes[1, 1].set_title(f"Predicted Tip Heatmap (Pix: ({pred_tip_pixel_x},{pred_tip_pixel_y}))")
            axes[1, 1].axis('off')

            # Show Blade ECT with predicted points
            axes[1, 2].imshow(blade_ect_viz, cmap='gray')
            axes[1, 2].set_title("Input ECT with Predicted Points")
            axes[1, 2].axis('off')
            axes[1, 2].scatter(pred_base_pixel_x, pred_base_pixel_y, c='red', marker='x', s=200, label='Pred Base', edgecolors='black')
            axes[1, 2].scatter(pred_tip_pixel_x, pred_tip_pixel_y, c='blue', marker='x', s=200, label='Pred Tip', edgecolors='black')
            axes[1, 2].legend(loc='upper right')


            plt.tight_layout()
            plt.show()
            print("Displayed an example inference. Look for the plot window.")
        else:
            print("Validation dataset is empty, cannot perform example inference.")
    else:
        print("No trained model found to perform inference. Ensure training completed successfully.")