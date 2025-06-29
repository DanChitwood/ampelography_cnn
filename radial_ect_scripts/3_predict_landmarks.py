import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm # For smart progress bars
import sys # For sys.exit()

# --- Configuration Parameters ---
IMAGE_SIZE = (256, 256) # Model input size (as used in training)

# Directory containing your model inputs (radial_ects, shape_masks, metadata.csv)
MODEL_INPUTS_BASE_DIR = Path("model_inputs")

# Path to your trained model (.pth file). YOU MUST UPDATE THIS!
# Example: Path("processed_leaf_data") / "saved_models" / "best_model_epoch_030_ede_0.15.pth"
# Ensure this path is correct relative to where you run this script.
PATH_TO_SAVED_MODEL = Path("processed_leaf_data") / "saved_models" / "best_model_epoch_030_ede_0.15.pth" # <<< UPDATE THIS!

# Output file for predictions (will be saved in the current working directory)
PREDICTED_METADATA_OUTPUT_FILE = Path("predictions.csv")

# Prediction parameters
BATCH_SIZE = 64 # Can be adjusted
NUM_WORKERS = 2 # For DataLoader

# --- 1. Custom Dataset Definition (Adapted for Prediction) ---
class LeafLandmarkDataset(Dataset):
    def __init__(self, metadata_path, ect_dir, mask_dir, transform=None):
        self.original_metadata_df = pd.read_csv(metadata_path)
        self.ect_dir = ect_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Filter out rows for which image files do not exist
        # This prevents FileNotFoundError during __getitem__ and ensures only valid samples are processed.
        self.valid_leaf_ids = []
        print(f"Verifying existence of image files in {self.ect_dir} and {self.mask_dir}...")
        
        # Use a list to hold valid leaf_ids and their corresponding metadata rows
        # This ensures we retain all original metadata for the valid samples.
        self.filtered_metadata_df = pd.DataFrame(columns=self.original_metadata_df.columns)
        
        for _, row in tqdm(self.original_metadata_df.iterrows(), total=len(self.original_metadata_df), desc="Preparing prediction inputs"):
            leaf_id = row['leaf_id']
            ect_img_path = self.ect_dir / f"{leaf_id}.png"
            mask_img_path = self.mask_dir / f"{leaf_id}.png"

            if ect_img_path.exists() and mask_img_path.exists():
                self.valid_leaf_ids.append(leaf_id)
                # Append the entire row for valid leaf_id
                # Using pd.concat for appending single rows can be inefficient in large loops.
                # For 11,000 images, this might be okay, but keep in mind for future.
                self.filtered_metadata_df = pd.concat([self.filtered_metadata_df, pd.DataFrame([row])], ignore_index=True)
            # else:
                # print(f"Skipping {leaf_id}: ECT or mask file not found.") # Optional: uncomment for verbose skipping
        
        print(f"Found {len(self.valid_leaf_ids)} valid samples for prediction after file check.")
        if len(self.valid_leaf_ids) == 0:
            print("Warning: No valid image files found. The prediction process will be empty.")

    # --- ADDED: The __len__ method ---
    def __len__(self):
        return len(self.valid_leaf_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        leaf_id = self.valid_leaf_ids[idx] # Use pre-filtered valid_leaf_ids
        
        ect_img_path = self.ect_dir / f"{leaf_id}.png"
        mask_img_path = self.mask_dir / f"{leaf_id}.png"

        ect_image = Image.open(ect_img_path).convert('L') # 'L' for grayscale
        mask_image = Image.open(mask_img_path).convert('L') # 'L' for grayscale

        if self.transform:
            ect_image = self.transform(ect_image)
            mask_image = self.transform(mask_image)
        
        # Concatenate ECT and Mask as two channels
        input_tensor = torch.cat([ect_image, mask_image], dim=0) # Resulting shape: (2, H, W)

        return input_tensor, leaf_id # Return leaf_id for linking predictions later

# --- 2. CNN Model Definition (Identical to Training Script) ---
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

# --- 3. Prediction Function ---
def predict_landmarks(model, data_loader, device):
    model.eval() # Set model to evaluation mode
    predictions_list = []
    leaf_ids_list = []

    print("\n--- Starting Landmark Prediction ---")
    prediction_bar = tqdm(data_loader, desc="Predicting Landmarks", unit="batch")

    with torch.no_grad(): # Disable gradient calculations
        for inputs, leaf_ids in prediction_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # The model was trained to output landmark coordinates in the -1 to 1 normalized ECT space.
            outputs_normalized_coords = outputs.cpu().numpy()

            predictions_list.extend(outputs_normalized_coords)
            leaf_ids_list.extend(leaf_ids)

    print("\nPrediction complete!")
    return leaf_ids_list, np.array(predictions_list)


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure input directories exist
    ect_dir = MODEL_INPUTS_BASE_DIR / "radial_ects"
    mask_dir = MODEL_INPUTS_BASE_DIR / "shape_masks"
    metadata_file = MODEL_INPUTS_BASE_DIR / "metadata.csv"

    if not ect_dir.exists():
        print(f"Error: Radial ECT directory not found at {ect_dir}")
        sys.exit(1)
    if not mask_dir.exists():
        print(f"Error: Shape mask directory not found at {mask_dir}")
        sys.exit(1)
    if not metadata_file.exists():
        print(f"Error: Metadata file not found at {metadata_file}")
        sys.exit(1)

    # Define transformations for the images (same as during training)
    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), # Ensure consistent sizing
        transforms.ToTensor(), # Converts PIL Image to Tensor (H, W) -> (C, H, W) and scales to [0.0, 1.0]
    ])

    # Instantiate the dataset, which will filter for existing image files
    full_dataset = LeafLandmarkDataset(
        metadata_path=metadata_file,
        ect_dir=ect_dir,
        mask_dir=mask_dir,
        transform=image_transform
    )

    if len(full_dataset) == 0:
        print("Error: No valid image files found after checking. Cannot proceed with prediction.")
        sys.exit(1)

    # Create DataLoader for prediction
    prediction_loader = DataLoader(
        full_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # No need to shuffle for prediction
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    print(f"Total samples being processed for prediction: {len(full_dataset)}")

    # Set up device (MPS or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS GPU for prediction.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Falling back to CPU for prediction.")

    # Instantiate the model and load trained weights
    model = LandmarkPredictorCNN().to(device)
    
    # Validate model path before loading
    if not PATH_TO_SAVED_MODEL.exists():
        print(f"Error: Trained model not found at {PATH_TO_SAVED_MODEL}")
        print("Please ensure PATH_TO_SAVED_MODEL is correctly set to your .pth file.")
        sys.exit(1)

    # Load model state dictionary
    model.load_state_dict(torch.load(PATH_TO_SAVED_MODEL, map_location=device, weights_only=True))
    print(f"Loaded trained model from: {PATH_TO_SAVED_MODEL}")

    # Perform prediction
    leaf_ids, predicted_landmarks_normalized = predict_landmarks(model, prediction_loader, device)

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'leaf_id': leaf_ids,
        'predicted_base_x': predicted_landmarks_normalized[:, 0],
        'predicted_base_y': predicted_landmarks_normalized[:, 1],
        'predicted_tip_x': predicted_landmarks_normalized[:, 2],
        'predicted_tip_y': predicted_landmarks_normalized[:, 3]
    })

    # Save predictions to CSV in the current working directory
    predictions_df.to_csv(PREDICTED_METADATA_OUTPUT_FILE, index=False)
    print(f"\nPredicted landmarks saved to: {PREDICTED_METADATA_OUTPUT_FILE.resolve()}") # .resolve() to show full path

    print("\nLandmark prediction script (Step 1) complete!")