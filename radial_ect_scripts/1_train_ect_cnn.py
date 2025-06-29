import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # For smart progress bars
import math # For sqrt in EDE calculation
import sys # For sys.exit()

# --- Configuration Parameters ---
IMAGE_SIZE = (256, 256)
PROCESSED_DATA_OUTPUT_DIR = Path("processed_leaf_data/")
RADIAL_ECT_DIR = PROCESSED_DATA_OUTPUT_DIR / "radial_ects"
SHAPE_MASK_DIR = PROCESSED_DATA_OUTPUT_DIR / "shape_masks"
METADATA_FILE = PROCESSED_DATA_OUTPUT_DIR / "metadata.csv"

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_VAL_SPLIT_RATIO = 0.8 # 80% training, 20% validation
NUM_WORKERS = 2 # For DataLoader
MODEL_SAVE_DIR = PROCESSED_DATA_OUTPUT_DIR / "saved_models" # New directory for saving models

# --- 1. Custom Dataset Definition ---
class LeafLandmarkDataset(Dataset):
    def __init__(self, metadata_path, ect_dir, mask_dir, transform=None):
        self.metadata_df = pd.read_csv(metadata_path)
        # Filter out invalid entries where landmarks might be NaN
        self.metadata_df = self.metadata_df.dropna(subset=['landmark_base_x', 'landmark_tip_x'])

        self.ect_dir = ect_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Store leaf IDs for efficient lookup
        self.leaf_ids = self.metadata_df['leaf_id'].tolist()

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        leaf_id = self.leaf_ids[idx]
        
        # Load ECT image (grayscale)
        ect_img_path = self.ect_dir / f"{leaf_id}.png"
        ect_image = Image.open(ect_img_path).convert('L') # 'L' for grayscale

        # Load Mask image (grayscale)
        mask_img_path = self.mask_dir / f"{leaf_id}.png"
        mask_image = Image.open(mask_img_path).convert('L') # 'L' for grayscale

        # Get landmark coordinates from metadata
        row = self.metadata_df[self.metadata_df['leaf_id'] == leaf_id].iloc[0]
        base_x = row['landmark_base_x']
        base_y = row['landmark_base_y']
        tip_x = row['landmark_tip_x']
        tip_y = row['landmark_tip_y']
        
        # Ensure coordinates are floats (important for regression targets)
        # Coordinates should ideally be normalized to [0, 1] range if not already done.
        # Assuming they are already relative to 256x256, or we can normalize them here.
        # For now, we'll pass them as-is and the model will predict pixel values.
        landmarks = np.array([base_x, base_y, tip_x, tip_y], dtype=np.float32)

        # Apply transformations if provided
        if self.transform:
            ect_image = self.transform(ect_image)
            mask_image = self.transform(mask_image)
        
        # Concatenate ECT and Mask as two channels
        input_tensor = torch.cat([ect_image, mask_image], dim=0) # Resulting shape: (2, H, W)

        return input_tensor, torch.from_numpy(landmarks)

# --- 2. CNN Model Definition ---
class LandmarkPredictorCNN(nn.Module):
    def __init__(self):
        super(LandmarkPredictorCNN, self).__init__()
        # Input: 2 channels (ECT, Mask), Output: 4 coordinates (base_x, base_y, tip_x, tip_y)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32x128x128

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x64x64

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128x32x32

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 256x16x16

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 512x8x8
        )

        # Calculate the size of the flattened features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Reduces spatial dimensions to 1x1
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Regularization
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Regularization
            nn.Linear(512, 4) # Output 4 coordinates (base_x, base_y, tip_x, tip_y)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x) # Apply adaptive pooling
        x = torch.flatten(x, 1) # Flatten starting from channel dimension
        x = self.classifier(x)
        return x

# --- Helper Function: Calculate Euclidean Distance Error (EDE) ---
def calculate_ede(predictions, targets):
    """
    Calculates the average Euclidean Distance Error for base and tip landmarks.
    predictions, targets: (batch_size, 4) tensors, where 4 are [base_x, base_y, tip_x, tip_y]
    """
    # Reshape to (batch_size, num_landmarks, 2)
    predictions_reshaped = predictions.view(-1, 2, 2)
    targets_reshaped = targets.view(-1, 2, 2)

    # Calculate squared differences
    squared_diff = (predictions_reshaped - targets_reshaped)**2
    
    # Sum across x,y coordinates for each landmark to get squared EDE
    squared_ede_per_landmark = squared_diff.sum(dim=-1) # (batch_size, num_landmarks)

    # Take square root to get EDE
    ede_per_landmark = torch.sqrt(squared_ede_per_landmark) # (batch_size, num_landmarks)

    # Average EDE across all landmarks for each sample, then across batch
    avg_ede = ede_per_landmark.mean() # Scalar
    
    return avg_ede

# --- 3. Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS):
    train_losses = []
    val_losses = []
    val_edes = [] # To store validation EDE

    best_val_ede = float('inf') # Initialize with a very high value
    best_epoch = -1

    # Create directory for saving models if it doesn't exist
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", unit="batch")
        
        for inputs, targets in train_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad() # Zero the parameter gradients

            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, targets) # Calculate loss (MSE)

            loss.backward() # Backward pass
            optimizer.step() # Optimize

            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}") # Update tqdm bar with current batch loss

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        val_running_ede = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", unit="batch")

        with torch.no_grad(): # Disable gradient calculations
            for inputs, targets in val_bar:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                ede = calculate_ede(outputs, targets) # Calculate EDE

                val_running_loss += loss.item() * inputs.size(0)
                val_running_ede += ede.item() * inputs.size(0) # EDE is already averaged per sample, just sum for epoch average

                val_bar.set_postfix(loss=f"{loss.item():.4f}", ede=f"{ede.item():.2f}")

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_ede = val_running_ede / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_edes.append(epoch_val_ede)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val EDE: {epoch_val_ede:.2f} pixels")

        # --- Save Best Model ---
        if epoch_val_ede < best_val_ede:
            best_val_ede = epoch_val_ede
            best_epoch = epoch + 1
            model_save_path = MODEL_SAVE_DIR / f"best_model_epoch_{best_epoch:03d}_ede_{best_val_ede:.2f}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved at {model_save_path} with EDE: {best_val_ede:.2f} pixels")
            
    print("\nTraining complete!")
    print(f"Best model achieved Val EDE: {best_val_ede:.2f} pixels at Epoch {best_epoch}")
    return train_losses, val_losses, val_edes

# --- Main Execution ---
if __name__ == "__main__":
    # Check if data directories exist
    if not RADIAL_ECT_DIR.exists() or not SHAPE_MASK_DIR.exists() or not METADATA_FILE.exists():
        print(f"Error: One or more required data directories/files not found:")
        print(f"  ECTs: {RADIAL_ECT_DIR.exists()}")
        print(f"  Masks: {SHAPE_MASK_DIR.exists()}")
        print(f"  Metadata: {METADATA_FILE.exists()}")
        print("Please ensure you have successfully run the previous data processing script.")
        sys.exit(1)

    # Define transformations for the images
    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), # Ensure consistent sizing
        transforms.ToTensor(), # Converts PIL Image to Tensor (H, W) -> (C, H, W) and scales to [0.0, 1.0]
    ])

    # Instantiate the dataset
    full_dataset = LeafLandmarkDataset(
        metadata_path=METADATA_FILE,
        ect_dir=RADIAL_ECT_DIR,
        mask_dir=SHAPE_MASK_DIR,
        transform=image_transform
    )

    if len(full_dataset) == 0:
        print("Error: No valid data found after loading metadata and filtering. Check metadata.csv and image paths.")
        sys.exit(1)

    # Split dataset into training and validation sets
    train_size = int(TRAIN_VAL_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Set up device (MPS or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS GPU for training.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Falling back to CPU for training.")

    # Instantiate the model, loss function, and optimizer
    model = LandmarkPredictorCNN().to(device)
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting CNN Training ---")
    train_losses, val_losses, val_edes = train_model(model, train_loader, val_loader, criterion, optimizer, device, NUM_EPOCHS)

    # --- Plotting Loss and EDE ---
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.plot(train_losses, label='Training MSE Loss')
    plt.plot(val_losses, label='Validation MSE Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    plt.plot(val_edes, label='Validation EDE (pixels)', color='orange')
    plt.title('Validation EDE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('EDE (pixels)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjust layout to prevent overlap
    loss_plot_path = PROCESSED_DATA_OUTPUT_DIR / "cnn_metrics_plot.png"
    plt.savefig(loss_plot_path)
    plt.show()
    print(f"Loss and EDE plots saved to {loss_plot_path}")

    print("\nCNN setup and training script complete!")