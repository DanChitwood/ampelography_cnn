import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import joblib # For loading the scaler
import torch.nn.functional as F

# ====================================================================================================
# USER CONFIGURABLE INPUTS
# ====================================================================================================

# Path to the folder containing your prepared mask images for prediction
PREDICTION_MASKS_DIR = "PREDICTION_MASKS"

# Path to your trained model file (the one with the best performance)
MODEL_PATH = "cnn_multitask_model_v6.pt" # Adjust if your best model is saved elsewhere

# Path to the PC scaler saved during training. This is CRITICAL for denormalizing predictions.
# If your model was trained without PC normalization, set this to None.
PC_SCALER_PATH = "pc_scaler_v6.pkl"

# List of class names IN THE EXACT ORDER they were encoded during training.
# This is crucial for mapping predicted class IDs back to human-readable labels.
# Example: If your classes were 'algeria', 'dissected', 'rootstock', 'vinifera', 'wild'
# and your LabelEncoder encoded them in that order (0, 1, 2, 3, 4 respectively).
CLASS_NAMES_IN_ORDER = ['algeria', 'dissected', 'rootstock', 'vinifera', 'wild'] # <--- IMPORTANT: Adjust this list!

# Output CSV file name for predictions
OUTPUT_RESULTS_CSV = "prediction_results.csv"

# ====================================================================================================
# Device setup (from your training script)
# ====================================================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================================================================================================
# Helper classes and model architecture (copied from your train_and_evaluate_model.py)
# ====================================================================================================

# Simplified Dataset for prediction - it doesn't need ground truth labels/PCs
class PredictionLeafDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.filenames = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
        self.transform = transform
        self.filenames.sort() # Ensure consistent order

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path).convert("L") # Convert to grayscale (L)

        if self.transform:
            image = self.transform(image)

        return image, self.filenames[idx] # Return image and its filename


# CNNMultitask Model (copied directly from your script)
class CNNMultitask(nn.Module):
    def __init__(self, n_classes, resolution=512):
        super(CNNMultitask, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()

        self.flattened_size = 128 * (resolution // 16) * (resolution // 16)

        self.fc_embedding = nn.Linear(self.flattened_size, 256)
        self.dropout_embedding = nn.Dropout(0.3)

        self.regressor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)

        if x.shape[1] != self.flattened_size:
            raise ValueError(f"Flattened feature size mismatch. Expected {self.flattened_size}, got {x.shape[1]}. "
                             f"Please ensure your input image resolution is compatible with the CNN layers, "
                             f"e.g., 512 for this model, or adjust CNN layers/use AdaptiveAvgPool2d.")

        embeddings = self.dropout_embedding(F.relu(self.fc_embedding(x)))

        pred_pcs = self.regressor_head(embeddings)
        pred_labels = self.classifier_head(embeddings)

        return pred_pcs, pred_labels, embeddings

# ====================================================================================================
# Main Prediction Function
# ====================================================================================================
def predict_on_masks(
    prediction_masks_dir: str,
    model_path: str,
    pc_scaler_path: str,
    class_names: list[str],
    output_csv_filename: str,
    resolution: int = 512 # Must match training resolution
):
    """
    Loads a trained CNNMultitask model and predicts on a folder of mask images.

    Args:
        prediction_masks_dir (str): Path to the folder containing prepared mask PNGs.
        model_path (str): Path to the saved PyTorch model state_dict (.pt file).
        pc_scaler_path (str): Path to the saved StandardScaler for PC values.
                              Set to None if PCs were not normalized during training.
        class_names (list[str]): Ordered list of class names used in training.
                                 Crucial for mapping predicted class IDs back to labels.
        output_csv_filename (str): Name for the output CSV file containing predictions.
        resolution (int): Resolution of input images to the model (must match training).
    """
    print(f"--- Starting Prediction using model: '{os.path.basename(model_path)}' ---")
    print(f"  Input masks from: '{prediction_masks_dir}'")

    if not os.path.exists(prediction_masks_dir):
        raise FileNotFoundError(f"Prediction masks directory not found: '{prediction_masks_dir}'")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: '{model_path}'")
    if pc_scaler_path and not os.path.exists(pc_scaler_path):
        print(f"WARNING: PC scaler not found at '{pc_scaler_path}'. Predicted PC values will remain normalized.")
        scaler = None
    elif pc_scaler_path:
        scaler = joblib.load(pc_scaler_path)
        print(f"Loaded PC scaler from: '{pc_scaler_path}'")
    else:
        scaler = None
        print("PC scaler path not provided. Predicted PC values will remain normalized.")

    # Data transformations for prediction (must match training input transformations)
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL Image to PyTorch Tensor (H, W) to (C, H, W) and scales to [0,1]
    ])

    # Initialize dataset and dataloader
    prediction_dataset = PredictionLeafDataset(prediction_masks_dir, transform)
    prediction_dataloader = torch.utils.data.DataLoader(prediction_dataset, batch_size=16, shuffle=False)

    if len(prediction_dataset) == 0:
        print(f"No PNG images found in '{prediction_masks_dir}'. Exiting prediction.")
        return

    # Initialize model and load trained weights
    model = CNNMultitask(n_classes=len(class_names), resolution=resolution).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    all_filenames = []
    all_raw_classes = []
    all_predicted_classes = []
    all_predicted_pc1 = []
    all_predicted_pc2 = []

    print(f"  Predicting on {len(prediction_dataset)} images...")

    with torch.no_grad(): # Disable gradient calculation for inference
        for batch_idx, (images, filenames) in enumerate(prediction_dataloader):
            images = images.to(device)
            
            pred_pcs_norm, pred_labels_logits, _ = model(images)

            # Process PC predictions
            pred_pcs_np = pred_pcs_norm.cpu().numpy()
            if scaler is not None:
                # Apply inverse transform to denormalize PCs
                pred_pcs_denorm = scaler.inverse_transform(pred_pcs_np)
                predicted_pc1_batch = pred_pcs_denorm[:, 0]
                predicted_pc2_batch = pred_pcs_denorm[:, 1]
            else:
                predicted_pc1_batch = pred_pcs_np[:, 0]
                predicted_pc2_batch = pred_pcs_np[:, 1]

            # Process class predictions
            predicted_class_ids = pred_labels_logits.argmax(dim=1).cpu().numpy()
            predicted_class_names = [class_names[idx] for idx in predicted_class_ids]

            # Extract raw class name from filename (e.g., AHMEUR_BOU_AHMEUR_ab from AHMEUR_BOU_AHMEUR_ab_1.png)
            raw_class_names_batch = []
            for fname in filenames:
                # Remove extension, then find the last '_' and everything after it
                base_name = os.path.splitext(fname)[0]
                parts = base_name.rsplit('_', 1)
                raw_class_names_batch.append(parts[0] if len(parts) > 1 else base_name) # Handle case with no '_'

            all_filenames.extend(filenames)
            all_raw_classes.extend(raw_class_names_batch)
            all_predicted_classes.extend(predicted_class_names)
            all_predicted_pc1.extend(predicted_pc1_batch)
            all_predicted_pc2.extend(predicted_pc2_batch)
            
            print(f"  Processed batch {batch_idx + 1}/{len(prediction_dataloader)}")


    # Create DataFrame for results
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'class': all_raw_classes,
        'predicted_class': all_predicted_classes,
        'predicted_PC1': all_predicted_pc1,
        'predicted_PC2': all_predicted_pc2
    })

    # Save results to CSV
    results_df.to_csv(output_csv_filename, index=False)
    print(f"\n✅ Prediction results saved to: '{output_csv_filename}'")
    print("--- Prediction Complete ---")

# ====================================================================================================
# Main execution block
# ====================================================================================================
if __name__ == "__main__":
    predict_on_masks(
        prediction_masks_dir=PREDICTION_MASKS_DIR,
        model_path=MODEL_PATH,
        pc_scaler_path=PC_SCALER_PATH,
        class_names=CLASS_NAMES_IN_ORDER,
        output_csv_filename=OUTPUT_RESULTS_CSV
    )