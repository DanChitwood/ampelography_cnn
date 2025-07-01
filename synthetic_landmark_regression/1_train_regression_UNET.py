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
MODEL_SAVE_PATH = Path("trained_leaf_regression_prediction_model.pth") # Path to save the best model

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

# --- Helper functions for ECT coords to pixels (now used in EDE calculation, not dataset) ---
def ect_coords_to_pixel_coords(ect_x, ect_y, image_size: tuple, bound_radius: float):
    """
    Converts ECT coordinates (typically in [-BOUND_RADIUS, BOUND_RADIUS]) to pixel coordinates (0 to image_size-1).
    ect_x, ect_y: individual float coordinates in ECT space.
    image_size: (height, width) tuple, e.g., (256, 256).
    bound_radius: The maximum extent of the ECT space (e.g., 1.0).

    Returns: (pixel_x, pixel_y) tuple
    """
    scale_factor_x = image_size[1] / (2 * bound_radius)
    scale_factor_y = image_size[0] / (2 * bound_radius)
    offset_x = image_size[1] / 2
    offset_y = image_size[0] / 2

    # ECT's Y-axis is often "up" (positive is up), image Y-axis is "down" (positive is down)
    # So, invert Y for conversion.
    pixel_x = int(ect_x * scale_factor_x + offset_x)
    pixel_y = int(-ect_y * scale_factor_y + offset_y) # Invert Y-axis

    # Clamp to image boundaries
    pixel_x = np.clip(pixel_x, 0, image_size[1] - 1)
    pixel_y = np.clip(pixel_y, 0, image_size[0] - 1)

    return pixel_x, pixel_y

# --- 1. Custom Dataset Class (Modified for Regression) ---
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

        # --- Direct Regression Targets ---
        # Target is now a tensor of 4 coordinates: [base_x, base_y, tip_x, tip_y]
        # These are already in the normalized ECT space (e.g., -1 to 1)
        target_coords = np.array([row['base_x'], row['base_y'], row['tip_x'], row['tip_y']], dtype=np.float32)
        target_tensor = torch.from_numpy(target_coords).float()

        # --- Future Improvement: Data Augmentation ---
        # For regression, augmentations must be applied to inputs AND their corresponding coordinates.
        # This means applying same rotation/scaling matrices to both image and coord vectors.
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
        # For regression, the output is a fixed number of coordinates, not an image.
        # This requires a global pooling layer before the final FC layer.
        # However, the UNet's OutConv typically maps to output channels of the *same spatial size*.
        # For coordinate regression with UNet, typically you add:
        # 1. An adaptive pooling layer (e.g., AdaptiveAvgPool2d(1)) at the end of the UNet's encoder
        #    or after the last Up-sampling block.
        # 2. A final fully connected layer (nn.Linear) that maps pooled features to coordinates.

        # Let's modify the UNet class's forward pass to add this for simplicity,
        # but keep OutConv as a standard 1x1 conv if you want to reuse this for
        # other segmentation tasks.
        # For direct regression, we'll bypass this OutConv entirely in UNet's forward
        # and add linear layers.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # This will not be used in the new UNet for regression

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out # This will be 4 for regression
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // (2 if bilinear else 1)) # Bottleneck

        # Upsampling path (typically brings it back to original image size or smaller for segmentation)
        self.up1 = Up(1024, 512 // (2 if bilinear else 1), bilinear)
        self.up2 = Up(512, 256 // (2 if bilinear else 1), bilinear)
        self.up3 = Up(256, 128 // (2 if bilinear else 1), bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Instead of OutConv for pixel-wise prediction, we need to adapt for regression.
        # We take the features from the last upsampling block (x_up4, shape: Bx64xHxW)
        # and apply global average pooling, then a fully connected layer.
        self.avgpool = nn.AdaptiveAvgPool2d(1) # Pools to 1x1 spatial dimension
        self.fc = nn.Linear(64, n_classes_out) # Maps 64 features to 4 coordinates

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Bottleneck features

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) # Features before final output convolution for segmentation

        # For regression, flatten these features after pooling
        pooled_features = self.avgpool(x) # Output shape: B x 64 x 1 x 1
        pooled_features = pooled_features.view(pooled_features.size(0), -1) # Flatten to B x 64
        
        logits = self.fc(pooled_features) # Output shape: B x n_classes_out (i.e., B x 4)
        return logits

# --- 3. Loss Function: Mean Squared Error (MSELoss) for Regression ---
# This remains nn.MSELoss() as it's suitable for regressing continuous values.

# --- Helper for EDE Metric (Modified for Regression Output) ---
def calculate_ede(predictions: torch.Tensor, targets: torch.Tensor, image_size: tuple, bound_radius: float):
    """
    Calculates the Expected Distance Error (EDE) for a batch of predicted coordinates.
    predictions: (B, 4) tensor of predicted coordinates [base_x, base_y, tip_x, tip_y] (in ECT space)
    targets: (B, 4) tensor of ground truth coordinates [base_x, base_y, tip_x, tip_y] (in ECT space)
    image_size: (H, W) tuple to convert ECT coords to pixels for EDE calculation.
    bound_radius: Scalar, the range for ECT coordinates (e.g., 1.0 for [-1, 1]).
    Returns: Average EDE across the batch and both landmarks (base/tip) in pixels.
    """
    batch_size = predictions.size(0)
    total_ede_sum = 0.0
    num_landmarks = 2 # Base and Tip

    for i in range(batch_size):
        # Extract predicted and ground truth coordinates (in ECT space)
        pred_coords = predictions[i].cpu().numpy()
        gt_coords = targets[i].cpu().numpy()

        # Base point
        pred_base_x_ect, pred_base_y_ect = pred_coords[0], pred_coords[1]
        gt_base_x_ect, gt_base_y_ect = gt_coords[0], gt_coords[1]

        # Convert to pixel coordinates for EDE calculation
        pred_base_x_pix, pred_base_y_pix = ect_coords_to_pixel_coords(pred_base_x_ect, pred_base_y_ect, image_size, bound_radius)
        gt_base_x_pix, gt_base_y_pix = ect_coords_to_pixel_coords(gt_base_x_ect, gt_base_y_ect, image_size, bound_radius)

        base_distance = np.sqrt((pred_base_x_pix - gt_base_x_pix)**2 + (pred_base_y_pix - gt_base_y_pix)**2)
        total_ede_sum += base_distance

        # Tip point
        pred_tip_x_ect, pred_tip_y_ect = pred_coords[2], pred_coords[3]
        gt_tip_x_ect, gt_tip_y_ect = gt_coords[2], gt_coords[3]

        # Convert to pixel coordinates for EDE calculation
        pred_tip_x_pix, pred_tip_y_pix = ect_coords_to_pixel_coords(pred_tip_x_ect, pred_tip_y_ect, image_size, bound_radius)
        gt_tip_x_pix, gt_tip_y_pix = ect_coords_to_pixel_coords(gt_tip_x_ect, gt_tip_y_ect, image_size, bound_radius)

        tip_distance = np.sqrt((pred_tip_x_pix - gt_tip_x_pix)**2 + (pred_tip_y_pix - gt_tip_y_pix)**2)
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
                outputs = model(inputs) # Outputs are now (B, 4) coordinates
                loss = criterion(outputs, targets) # MSE Loss

                loss.backward()
                optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(batch_loss=f"{loss.item():.6f}")

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

                outputs = model(inputs) # Outputs are (B, 4) coordinates
                loss = criterion(outputs, targets) # MSE Loss for validation
                
                # Calculate EDE
                # Pass original IMAGE_SIZE and BOUND_RADIUS to calculate_ede
                ede = calculate_ede(outputs, targets, IMAGE_SIZE, BOUND_RADIUS)

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
    # n_classes_out = 4 (base_x, base_y, tip_x, tip_y)
    model = UNet(n_channels_in=2, n_classes_out=4).to(DEVICE)

    # Loss function and optimizer
    criterion = nn.MSELoss() # Still MSELoss for coordinate regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    trained_model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH)

    print("\n--- Model Training Finished ---")

    # --- Example Inference (using the best saved model) ---
    print("\n--- Example Inference ---")
    if MODEL_SAVE_PATH.exists():
        # Load the best model's state
        inference_model = UNet(n_channels_in=2, n_classes_out=4).to(DEVICE)
        inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        inference_model.eval() # Set to evaluation mode

        # Take one sample from the validation set for demonstration
        if len(val_dataset) > 0:
            sample_inputs, sample_target_coords_ect = val_dataset[0]
            
            # Add batch dimension and move to device
            sample_inputs = sample_inputs.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                predicted_coords_ect = inference_model(sample_inputs) # Output is (1, 4) tensor

            # Move tensors back to CPU and convert to numpy
            predicted_coords_ect_np = predicted_coords_ect.squeeze(0).cpu().numpy()
            sample_target_coords_ect_np = sample_target_coords_ect.cpu().numpy()

            # Extract individual coordinates in ECT space
            pred_base_x_ect, pred_base_y_ect, pred_tip_x_ect, pred_tip_y_ect = predicted_coords_ect_np
            gt_base_x_ect, gt_base_y_ect, gt_tip_x_ect, gt_tip_y_ect = sample_target_coords_ect_np

            # Convert all to pixel coordinates for plotting
            pred_base_x_pix, pred_base_y_pix = ect_coords_to_pixel_coords(pred_base_x_ect, pred_base_y_ect, IMAGE_SIZE, BOUND_RADIUS)
            pred_tip_x_pix, pred_tip_y_pix = ect_coords_to_pixel_coords(pred_tip_x_ect, pred_tip_y_ect, IMAGE_SIZE, BOUND_RADIUS)
            gt_base_x_pix, gt_base_y_pix = ect_coords_to_pixel_coords(gt_base_x_ect, gt_base_y_ect, IMAGE_SIZE, BOUND_RADIUS)
            gt_tip_x_pix, gt_tip_y_pix = ect_coords_to_pixel_coords(gt_tip_x_ect, gt_tip_y_ect, IMAGE_SIZE, BOUND_RADIUS)


            # Plotting
            plt.figure(figsize=(15, 6))

            # Input: Blade ECT
            plt.subplot(1, 3, 1)
            blade_ect_viz = sample_inputs.squeeze(0).cpu().numpy()[0]
            plt.imshow(blade_ect_viz, cmap='gray')
            plt.title("Input: Blade ECT")
            plt.axis('off')

            # Input: Blade Mask
            plt.subplot(1, 3, 2)
            blade_mask_viz = sample_inputs.squeeze(0).cpu().numpy()[1]
            plt.imshow(blade_mask_viz, cmap='gray')
            plt.title("Input: Blade Mask")
            plt.axis('off')

            # Input ECT with GT and Predicted Points
            plt.subplot(1, 3, 3)
            plt.imshow(blade_ect_viz, cmap='gray')
            plt.title("Blade ECT with GT & Predicted Points")
            plt.axis('off')

            # Plot Ground Truth Points
            plt.scatter(gt_base_x_pix, gt_base_y_pix, c='red', marker='o', s=150, label='GT Base', edgecolors='white', linewidth=2)
            plt.scatter(gt_tip_x_pix, gt_tip_y_pix, c='blue', marker='o', s=150, label='GT Tip', edgecolors='white', linewidth=2)

            # Plot Predicted Points
            plt.scatter(pred_base_x_pix, pred_base_y_pix, c='red', marker='x', s=200, label='Pred Base', edgecolors='black', linewidth=2)
            plt.scatter(pred_tip_x_pix, pred_tip_y_pix, c='blue', marker='x', s=200, label='Pred Tip', edgecolors='black', linewidth=2)

            plt.legend(loc='upper right')

            plt.tight_layout()
            plt.show()
            print("Displayed an example inference. Look for the plot window.")
        else:
            print("Validation dataset is empty, cannot perform example inference.")
    else:
        print("No trained model found to perform inference. Ensure training completed successfully.")