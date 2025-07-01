import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import sys
import argparse

# --- Configuration Parameters (MUST MATCH your training script) ---
IMAGE_SIZE = (256, 256)
BOUND_RADIUS = 1.0 # This value defines the extent of the ECT space, typically ECT coordinates are in [-BOUND_RADIUS, BOUND_RADIUS]

# Directories and Files (Adjust if your training script uses different paths)
SYNTHETIC_DATA_OUTPUT_DIR = Path("synthetic_leaf_data/")
SYNTHETIC_METADATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"
DEFAULT_MODEL_PATH = Path("trained_leaf_regression_prediction_model.pth")
DEFAULT_VISUALIZATION_OUTPUT_DIR = Path("prediction_visualizations_regression")

# --- Device Configuration ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# --- Helper function for ECT coords to pixels ---
def ect_coords_to_pixel_coords(ect_x: float, ect_y: float, image_size: tuple, bound_radius: float):
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

# --- Custom Dataset Class (Copied from training script) ---
class SyntheticLeafDataset(Dataset):
    def __init__(self, metadata_file: Path, base_dir: Path, transform=None):
        self.metadata_df = pd.read_csv(metadata_file)
        self.metadata_df = self.metadata_df[
            self.metadata_df['is_processed_valid'] &
            self.metadata_df['base_x'].notna() & self.metadata_df['base_y'].notna() &
            self.metadata_df['tip_x'].notna() & self.metadata_df['tip_y'].notna()
        ].reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

        if self.metadata_df.empty:
            raise ValueError(f"No valid processed samples with complete landmark data found in metadata file: {metadata_file}")
            
        print(f"Loaded {len(self.metadata_df)} valid synthetic samples for visualization.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        
        blade_ect_path = self.base_dir / row['file_blade_ect']
        blade_ect_img = Image.open(blade_ect_path).convert("L")
        
        blade_mask_path = self.base_dir / row['file_blade_mask']
        blade_mask_img = Image.open(blade_mask_path).convert("L")

        blade_ect_np = np.array(blade_ect_img) / 255.0
        blade_mask_np = np.array(blade_mask_img) / 255.0

        inputs_stacked = np.stack([blade_ect_np, blade_mask_np], axis=0)
        inputs_tensor = torch.from_numpy(inputs_stacked).float()

        target_coords = np.array([row['base_x'], row['base_y'], row['tip_x'], row['tip_y']], dtype=np.float32)
        target_tensor = torch.from_numpy(target_coords).float()

        if self.transform:
             pass 

        return inputs_tensor, target_tensor

# --- UNet Model Architecture (Copied from training script) ---
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
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

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_classes_out)

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

        pooled_features = self.avgpool(x)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        logits = self.fc(pooled_features)
        return logits

# --- Visualization Function (Main Logic) ---
def visualize_predictions(model: nn.Module, dataloader: DataLoader, num_samples: int, output_dir: Path, image_size: tuple, bound_radius: float):
    """
    Performs inference on a few samples and visualizes the predicted vs. ground truth points.
    Saves the visualizations to image files.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for the dataset (e.g., validation set).
        num_samples: Number of samples to visualize.
        output_dir: Directory to save the visualization images.
        image_size: Tuple (H, W) of the input image dimensions in pixels.
        bound_radius: The radius of the ECT coordinate space (e.g., 1.0).
    """
    model.eval() # Set model to evaluation mode
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    sample_count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            if sample_count >= num_samples:
                break

            device = next(model.parameters()).device # Get model's current device
            inputs = inputs.to(device)
            
            outputs = model(inputs) # Predicted coordinates (B, 4)

            for i in range(inputs.size(0)):
                if sample_count >= num_samples:
                    break

                current_input_ect = inputs[i, 0].cpu().numpy()
                current_gt_coords_ect = targets[i].cpu().numpy()
                current_pred_coords_ect = outputs[i].cpu().numpy()

                gt_base_x_pix, gt_base_y_pix = ect_coords_to_pixel_coords(current_gt_coords_ect[0], current_gt_coords_ect[1], image_size, bound_radius)
                gt_tip_x_pix, gt_tip_y_pix = ect_coords_to_pixel_coords(current_gt_coords_ect[2], current_gt_coords_ect[3], image_size, bound_radius)
                pred_base_x_pix, pred_base_y_pix = ect_coords_to_pixel_coords(current_pred_coords_ect[0], current_pred_coords_ect[1], image_size, bound_radius)
                pred_tip_x_pix, pred_tip_y_pix = ect_coords_to_pixel_coords(current_pred_coords_ect[2], current_pred_coords_ect[3], image_size, bound_radius)
                
                ede_base = np.sqrt((pred_base_x_pix - gt_base_x_pix)**2 + (pred_base_y_pix - gt_base_y_pix)**2)
                ede_tip = np.sqrt((pred_tip_x_pix - gt_tip_x_pix)**2 + (pred_tip_y_pix - gt_tip_y_pix)**2)
                avg_ede_sample = (ede_base + ede_tip) / 2

                plt.figure(figsize=(8, 8))
                plt.imshow(current_input_ect, cmap='gray')
                plt.title(f"Sample {sample_count+1}: GT vs. Predicted Points (Avg EDE: {avg_ede_sample:.2f} px)")
                plt.axis('off')

                plt.scatter(gt_base_x_pix, gt_base_y_pix, c='red', marker='o', s=150, label='GT Base', edgecolors='white', linewidth=2, zorder=3)
                plt.scatter(gt_tip_x_pix, gt_tip_y_pix, c='blue', marker='o', s=150, label='GT Tip', edgecolors='white', linewidth=2, zorder=3)

                plt.scatter(pred_base_x_pix, pred_base_y_pix, c='red', marker='x', s=200, label='Pred Base', edgecolors='black', linewidth=2, zorder=4)
                plt.scatter(pred_tip_x_pix, pred_tip_y_pix, c='blue', marker='x', s=200, label='Pred Tip', edgecolors='black', linewidth=2, zorder=4)

                plt.legend(loc='upper right', framealpha=0.8)
                
                output_filepath = output_dir / f"prediction_sample_{sample_count+1:03d}_ede_{avg_ede_sample:.2f}px.png"
                plt.savefig(output_filepath, bbox_inches='tight', dpi=150)
                plt.close()

                sample_count += 1
    print(f"\nSaved {sample_count} prediction visualizations to '{output_dir}'.")

# --- Main Execution for Standalone Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize leaf base/tip predictions from a trained UNet regression model.")
    parser.add_argument('--model_path', type=Path, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the trained model (.pth) file. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_VISUALIZATION_OUTPUT_DIR,
                        help=f"Directory to save visualization images. Default: {DEFAULT_VISUALIZATION_OUTPUT_DIR}")
    parser.add_argument('--num_samples', type=int, default=20,
                        help="Number of samples to visualize. Default: 20")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for loading data during visualization. Default: 16")
    parser.add_argument('--validation_split_ratio', type=float, default=0.2,
                        help="Ratio of total dataset used for validation. Must match training. Default: 0.2")

    args = parser.parse_args()

    DEVICE = get_device()
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {args.model_path}")
    print(f"Saving visualizations to: {args.output_dir}")

    if not args.model_path.exists():
        print(f"Error: Trained model not found at {args.model_path}. Please ensure the training script has been run.")
        sys.exit(1)

    # --- Data Loading ---
    try:
        full_dataset = SyntheticLeafDataset(SYNTHETIC_METADATA_FILE, SYNTHETIC_DATA_OUTPUT_DIR)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        print("Please ensure the synthetic data generation script has been run successfully and generated valid landmark data.")
        sys.exit(1)

    # Re-split the dataset to get the validation set correctly
    total_samples = len(full_dataset)
    train_size = int((1 - args.validation_split_ratio) * total_samples)
    val_size = total_samples - train_size
    # Use dummy generator for reproducibility if needed, or just let it be random.
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # We only need the validation dataloader for visualization
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) # num_workers=0 for stability

    print(f"Validation samples available: {len(val_dataset)}")
    if len(val_dataset) == 0:
        print("Error: Validation dataset is empty. Cannot perform visualizations.")
        sys.exit(1)
    
    # --- Model Loading ---
    inference_model = UNet(n_channels_in=2, n_classes_out=4).to(DEVICE)
    inference_model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    inference_model.eval()
    
    # Pass the device to the model for consistency in the visualization function
    inference_model.device = DEVICE 

    # --- Run Visualization ---
    visualize_predictions(
        inference_model, 
        val_dataloader,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        image_size=IMAGE_SIZE,
        bound_radius=BOUND_RADIUS
    )

    print("\nVisualization complete. Check the output directory for images.")