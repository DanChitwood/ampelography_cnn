# train_and_evaluate_model.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, precision_recall_fscore_support, accuracy_score
)
from scipy.stats import pearsonr
from umap import UMAP
from sklearn.manifold import TSNE # Import t-SNE for plotting
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ====================
# Device setup
# ====================
# Automatically selects the best available device (MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ====================================================================================================
# 2. LeafDataset Class: PyTorch Dataset for loading images and their associated data
# ====================================================================================================
class LeafDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None):
        self.image_dir = image_dir
        self.metadata_df = pd.read_csv(metadata_file)
        self.transform = transform

        self.filenames = self.metadata_df['filename'].values
        self.label_encoder = LabelEncoder()
        # Fit label encoder on all labels from the metadata to ensure consistent mapping
        self.labels = self.label_encoder.fit_transform(self.metadata_df['label'])
        self.pcs = self.metadata_df[["PC1", "PC2"]].values.astype(np.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path).convert("L") # Convert to grayscale (L)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        pcs = torch.tensor(self.pcs[idx], dtype=torch.float32)

        return image, label, pcs, self.filenames[idx]


# ====================================================================================================
# 3. CNNMultitask Model: The multi-task Convolutional Neural Network architecture
# ====================================================================================================
class CNNMultitask(nn.Module):
    def __init__(self, n_classes, resolution=512):
        super(CNNMultitask, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), # Output: 16 x 256 x 256
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), # Output: 32 x 128 x 128
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), # Output: 64 x 64 x 64
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2) # Output: 128 x 32 x 32
        )
        self.flatten = nn.Flatten()

        # Calculate the size of the flattened features before the FC layers
        # This assumes a 512x512 input image, which is downsampled 4 times by MaxPool2d (512 / 2^4 = 32)
        self.flattened_size = 128 * (resolution // 16) * (resolution // 16) # For 512 res: 128 * 32 * 32 = 131072

        # Fully connected embedding layer
        self.fc_embedding = nn.Linear(self.flattened_size, 256)
        self.dropout_embedding = nn.Dropout(0.3) # PARAMETER: Dropout rate for the embedding layer

        # Regressor Head: Predicts 2 PC values
        self.regressor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Output: 2 PC values
        )

        # Classifier Head: Predicts n_classes (genotype labels)
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # PARAMETER: Dropout rate for the classifier head
            nn.Linear(128, n_classes) # Output: n_classes (logits for classification)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)

        # Runtime check for flattened size consistency
        if x.shape[1] != self.flattened_size:
            raise ValueError(f"Flattened feature size mismatch. Expected {self.flattened_size}, got {x.shape[1]}. "
                             "Please ensure your input image resolution is compatible with the CNN layers, "
                             "e.g., 512 for this model, or adjust CNN layers/use AdaptiveAvgPool2d.")

        embeddings = self.dropout_embedding(F.relu(self.fc_embedding(x))) # Apply ReLU and Dropout to embedding

        pred_pcs = self.regressor_head(embeddings)
        pred_labels = self.classifier_head(embeddings)

        return pred_pcs, pred_labels, embeddings # OUTPUT: Predicted PCs, Predicted Labels (logits), and Embeddings

# ====================================================================================================
# 4. Training Function: Orchestrates the training loop, including loss calculation
# (MODIFIED: Early stopping removed, runs for full epochs)
# ====================================================================================================
def train_model(data_dir="PIXEL_IMAGES",
                batch_size=16, # PARAMETER: Batch size for training
                lr=3e-4, # PARAMETER: Learning rate for the Adam optimizer
                epochs=150, # PARAMETER: Maximum number of training epochs
                alpha=0.3, # PARAMETER: Weight for the classification loss in the total loss (Regression Loss + alpha * Classification Loss)
                eval_interval=5): # PARAMETER: How often (in epochs) to evaluate on the validation set

    metadata_file = os.path.join(data_dir, "METADATA.csv")
    image_dir = os.path.join(data_dir, "MASKS")

    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL Image to PyTorch Tensor (H, W) to (C, H, W) and scales to [0,1]
    ])

    full_dataset = LeafDataset(image_dir, metadata_file, transform)

    # Calculate class weights for imbalanced classification loss (Inverse Frequency Weighting)
    original_labels = full_dataset.metadata_df['label'].values
    label_encoder_for_weights = LabelEncoder()
    encoded_labels = label_encoder_for_weights.fit_transform(original_labels)
    n_classes = len(np.unique(encoded_labels))

    class_counts = pd.Series(encoded_labels).value_counts().sort_index()
    total_samples = len(encoded_labels)

    # Ensure all classes from 0 to n_classes-1 are present in class_counts for correct indexing
    all_classes_counts = pd.Series(0, index=range(n_classes))
    all_classes_counts.update(class_counts)
    class_weights = total_samples / (n_classes * (all_classes_counts.values + 1e-6)) # Add epsilon to prevent division by zero
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Calculated class weights (Inverse Frequency): {class_weights}")


    # Split data into training and validation sets
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), test_size=0.2, random_state=42, # 20% for validation, fixed random state for reproducibility
        stratify=full_dataset.labels # Stratify to maintain class distribution in splits
    )
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_indices), batch_size=batch_size)

    # Initialize model, optimizer, and loss functions
    model = CNNMultitask(n_classes).to(device) # Move model to selected device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_reg = nn.MSELoss() # Mean Squared Error for regression
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights) # Cross-Entropy Loss for classification with weights

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    print(f"\n--- Starting Model Training ---")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Initial Learning Rate: {lr}")
    print(f"  Alpha (Classification Loss Weight): {alpha}")
    print(f"  Training for {epochs} epochs (Early stopping disabled as per request)")

    train_losses = [] # To store training loss for each epoch
    val_losses = [] # To store validation loss for evaluation intervals

    # Best model saving path
    model_save_path = os.path.join(data_dir, "cnn_multitask_model.pt")
    # We will save the model at the end of training
    # No "best_val_loss" tracking for intermediate saving if early stopping is disabled.


    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_train_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0

        for batch_idx, (imgs, labels, pcs, filenames) in enumerate(train_loader):
            imgs, labels, pcs = imgs.to(device), labels.to(device), pcs.to(device) # Move data to device

            optimizer.zero_grad() # Clear gradients

            pred_pcs, pred_labels, _ = model(imgs) # Forward pass

            # Calculate individual losses
            loss_reg = criterion_reg(pred_pcs, pcs)
            loss_cls = criterion_cls(pred_labels, labels)

            # Combined loss
            loss = loss_reg + alpha * loss_cls

            loss.backward() # Backpropagation
            optimizer.step() # Update model weights

            total_train_loss += loss.item()
            total_reg_loss += loss_reg.item()
            total_cls_loss += loss_cls.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} (Reg: {total_reg_loss/len(train_loader):.4f}, Cls: {total_cls_loss/len(train_loader):.4f})")

        # Evaluate on validation set periodically
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            model.eval() # Set model to evaluation mode
            total_val_loss = 0
            with torch.no_grad(): # Disable gradient calculation for evaluation
                for imgs, labels, pcs, filenames in val_loader:
                    imgs, labels, pcs = imgs.to(device), labels.to(device), pcs.to(device)
                    pred_pcs, pred_labels, _ = model(imgs)
                    loss = criterion_reg(pred_pcs, pcs) + alpha * criterion_cls(pred_labels, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"  Validation Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss) # Update learning rate based on validation loss

    # Save the final trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Final model saved to {model_save_path}")

    # Plot and save training/validation loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    # Adjust val_epochs to match the length of val_losses (due to eval_interval)
    val_epochs = [e for e in range(1, epochs + 1) if (e % eval_interval == 0 or e == epochs)]
    plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "training_validation_loss.png")) # OUTPUT: Loss plot
    plt.close()

    print("--- Model Training Finished ---")
    return model, train_losses, val_losses # OUTPUT: Trained model, training losses, validation losses


# ====================================================================================================
# 5. Evaluation Function: Evaluates the trained model and generates various plots and metrics
# ====================================================================================================
def evaluate_model(data_dir="PIXEL_IMAGES", data_split='all'): # PARAMETER: 'all' for full data, 'validation' for validation set
    image_dir = os.path.join(data_dir, "MASKS")
    metadata_csv = os.path.join(data_dir, "METADATA.csv")
    model_path = os.path.join(data_dir, "cnn_multitask_model.pt")
    output_dir = os.path.join(data_dir, f"CNN_EVAL_OUTPUTS_{data_split.upper()}") # Dynamic output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Starting Model Evaluation on {data_split.upper()} data ---")

    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = LeafDataset(image_dir, metadata_csv, transform)

    # Determine which subset of data to evaluate
    filenames_to_evaluate = []
    if data_split == 'validation':
        _, val_indices = train_test_split(
            range(len(full_dataset)), test_size=0.2, random_state=42,
            stratify=full_dataset.labels
        )
        dataset_to_evaluate = torch.utils.data.Subset(full_dataset, val_indices)
        eval_title_suffix = " (Validation Data)"
        filenames_to_evaluate = [full_dataset.filenames[i] for i in val_indices]
    else: # data_split == 'all'
        dataset_to_evaluate = full_dataset
        eval_title_suffix = ""
        filenames_to_evaluate = full_dataset.filenames

    dataloader = DataLoader(dataset_to_evaluate, batch_size=16, shuffle=False)

    # Load the trained model
    model = CNNMultitask(n_classes=len(np.unique(full_dataset.labels))).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # Load PC scaler if it exists, to inverse transform normalized PCs
    scaler = None
    scaler_path = os.path.join(data_dir, "SCALERS", "pc_scaler.pkl") # Corrected path for scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Loaded PC scaler for inverse transformation.")
    else:
        print(f"WARNING: PC scaler not found at {scaler_path}. PCs will remain normalized.")


    # Collect all predictions, true values, embeddings, and filenames
    all_preds_pcs = []
    all_true_pcs = []
    all_preds_labels = []
    all_true_labels = []
    all_embeddings = []
    all_filenames_in_order = [] # Store filenames in the order they appear in the dataloader

    with torch.no_grad(): # Disable gradient calculation during evaluation
        for images, labels, pcs, filenames in dataloader:
            images = images.to(device)
            pcs = pcs.to(device)
            labels = labels.to(device)

            preds_pcs, preds_labels, embeddings = model(images)

            all_preds_pcs.append(preds_pcs.cpu().numpy())
            all_true_pcs.append(pcs.cpu().numpy())
            all_preds_labels.append(preds_labels.argmax(dim=1).cpu().numpy()) # Convert logits to class predictions
            all_true_labels.append(labels.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
            all_filenames_in_order.extend(filenames)

    # Concatenate all collected data
    pred_pcs_norm = np.vstack(all_preds_pcs)
    true_pcs_norm = np.vstack(all_true_pcs)
    pred_labels = np.concatenate(all_preds_labels)
    true_labels = np.concatenate(all_true_labels)
    embeddings = np.vstack(all_embeddings)

    # Get human-readable class names
    label_names = full_dataset.label_encoder.inverse_transform(np.unique(full_dataset.labels))
    num_classes = len(label_names)

    # Inverse transform PC values if a scaler was used during preprocessing
    if scaler is not None:
        pred_pcs = scaler.inverse_transform(pred_pcs_norm)
        true_pcs = scaler.inverse_transform(true_pcs_norm)
        print("Inverse transformed PCs for plotting and metrics.")
    else:
        pred_pcs = pred_pcs_norm
        true_pcs = true_pcs_norm

    # --- OUTPUT: Save Combined Predictions and True Values to CSV ---
    true_label_names = full_dataset.label_encoder.inverse_transform(true_labels)
    pred_label_names = full_dataset.label_encoder.inverse_transform(pred_labels)

    results_df = pd.DataFrame({
        'filename': all_filenames_in_order,
        'true_label': true_label_names,
        'predicted_label': pred_label_names,
        'true_PC1': true_pcs[:, 0],
        'true_PC2': true_pcs[:, 1],
        'predicted_PC1': pred_pcs[:, 0],
        'predicted_PC2': pred_pcs[:, 1]
    })
    results_csv_path = os.path.join(output_dir, "predictions_and_true_values.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ True and predicted values saved to: {results_csv_path}")


    # --- Regression Metrics (R2 and Pearsonr) ---
    r2_pc1 = r2_score(true_pcs[:, 0], pred_pcs[:, 0])
    r2_pc2 = r2_score(true_pcs[:, 1], pred_pcs[:, 1])
    pearson_pc1, _ = pearsonr(true_pcs[:, 0], pred_pcs[:, 0])
    pearson_pc2, _ = pearsonr(true_pcs[:, 1], pred_pcs[:, 1])

    # --- Classification Metrics ---
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=np.arange(num_classes))
    # Calculate weighted averages for overall summary
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', labels=np.arange(num_classes))


    # --- OUTPUT: Save Metrics to File ---
    metrics_file_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file_path, 'w') as f:
        f.write(f"--- Evaluation Metrics {eval_title_suffix} ---\n\n")
        f.write(f"REGRESSION METRICS:\n")
        f.write(f"  R2 Score (PC1): {r2_pc1:.4f}\n")
        f.write(f"  Pearson Correlation (PC1): {pearson_pc1:.4f}\n")
        f.write(f"  R2 Score (PC2): {r2_pc2:.4f}\n")
        f.write(f"  Pearson Correlation (PC2): {pearson_pc2:.4f}\n\n")

        f.write(f"CLASSIFICATION METRICS:\n")
        f.write(f"  Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(f"  Per-Class Metrics:\n")
        for i, class_name in enumerate(label_names):
            f.write(f"    Class '{class_name}':\n")
            f.write(f"      Precision: {precision[i]:.4f}\n")
            f.write(f"      Recall: {recall[i]:.4f}\n")
            f.write(f"      F1-Score: {f1[i]:.4f}\n")
        f.write(f"\n  Weighted Average Metrics:\n")
        f.write(f"    Precision (weighted): {weighted_precision:.4f}\n")
        f.write(f"    Recall (weighted): {weighted_recall:.4f}\n")
        f.write(f"    F1-Score (weighted): {weighted_f1:.4f}\n")
    print(f"✅ Metrics saved to: {metrics_file_path}")


    # --- Print Metrics to console ---
    print(f"\n--- Regression Metrics {eval_title_suffix} ---")
    print(f"R2 Score (PC1): {r2_pc1:.4f}")
    print(f"Pearson Correlation (PC1): {pearson_pc1:.4f}")
    print(f"R2 Score (PC2): {r2_pc2:.4f}")
    print(f"Pearson Correlation (PC2): {pearson_pc2:.4f}")

    print(f"\n--- Classification Metrics {eval_title_suffix} ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nPer-Class Metrics:")
    for i, class_name in enumerate(label_names):
        print(f"  Class '{class_name}': Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")
    print(f"\nWeighted Average: Precision={weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-Score={weighted_f1:.4f}")

    # --- Plot Confusion Matrix ---
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'Confusion Matrix{eval_title_suffix}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png")) # OUTPUT: Confusion matrix plot
    plt.close()
    print(f"✅ Confusion matrix saved to: {os.path.join(output_dir, 'confusion_matrix.png')}")


    # --- Plot Predicted vs. True PCs (PC1 vs. PC2) ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(true_pcs[:, 0], pred_pcs[:, 0], alpha=0.6)
    plt.plot([min(true_pcs[:, 0]), max(true_pcs[:, 0])], [min(true_pcs[:, 0]), max(true_pcs[:, 0])], 'r--', label='Ideal')
    plt.title(f'Predicted vs. True PC1{eval_title_suffix}\nR2={r2_pc1:.2f}, Pearson={pearson_pc1:.2f}')
    plt.xlabel('True PC1')
    plt.ylabel('Predicted PC1')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(true_pcs[:, 1], pred_pcs[:, 1], alpha=0.6)
    plt.plot([min(true_pcs[:, 1]), max(true_pcs[:, 1])], [min(true_pcs[:, 1]), max(true_pcs[:, 1])], 'r--', label='Ideal')
    plt.title(f'Predicted vs. True PC2{eval_title_suffix}\nR2={r2_pc2:.2f}, Pearson={pearson_pc2:.2f}')
    plt.xlabel('True PC2')
    plt.ylabel('Predicted PC2')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predicted_vs_true_pcs.png")) # OUTPUT: Predicted vs True PCs plot
    plt.close()
    print(f"✅ Predicted vs. True PCs plot saved to: {os.path.join(output_dir, 'predicted_vs_true_pcs.png')}")


    # --- Plot True PC1 vs. PC2 with True Labels ---
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(true_pcs[:, 0], true_pcs[:, 1], c=true_labels, cmap='viridis', alpha=0.7, label=label_names)
    plt.title(f'True PC1 vs. PC2 with True Labels{eval_title_suffix}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, ticks=np.arange(num_classes), label='True Label')
    plt.clim(-0.5, num_classes - 0.5) # Ensure colorbar ticks align with class centers
    # Create legend manually for class names
    handles, _ = scatter.legend_elements(num_classes=num_classes)
    plt.legend(handles=handles, labels=label_names, title="True Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "true_pcs_with_true_labels.png")) # OUTPUT: True PCs with True Labels plot
    plt.close()
    print(f"✅ True PCs with True Labels plot saved to: {os.path.join(output_dir, 'true_pcs_with_true_labels.png')}")


    # --- Plot Predicted PC1 vs. PC2 with Predicted Labels ---
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(pred_pcs[:, 0], pred_pcs[:, 1], c=pred_labels, cmap='viridis', alpha=0.7, label=label_names)
    plt.title(f'Predicted PC1 vs. PC2 with Predicted Labels{eval_title_suffix}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, ticks=np.arange(num_classes), label='Predicted Label')
    plt.clim(-0.5, num_classes - 0.5)
    handles, _ = scatter.legend_elements(num_classes=num_classes)
    plt.legend(handles=handles, labels=label_names, title="Predicted Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predicted_pcs_with_predicted_labels.png")) # OUTPUT: Predicted PCs with Predicted Labels plot
    plt.close()
    print(f"✅ Predicted PCs with Predicted Labels plot saved to: {os.path.join(output_dir, 'predicted_pcs_with_predicted_labels.png')}")


    # --- Plot UMAP of Embeddings colored by True Labels ---
    print("Generating UMAP plot for embeddings (True Labels)...")
    umap_reducer = UMAP(n_components=2, random_state=42)
    embeddings_2d_umap = umap_reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
    plt.title(f'UMAP of Embeddings (True Labels){eval_title_suffix}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.colorbar(scatter, ticks=np.arange(num_classes), label='True Label')
    plt.clim(-0.5, num_classes - 0.5)
    handles, _ = scatter.legend_elements(num_classes=num_classes)
    plt.legend(handles=handles, labels=label_names, title="True Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_embeddings_true_labels.png")) # OUTPUT: UMAP plot by true labels
    plt.close()
    print(f"✅ UMAP of embeddings (True Labels) plot saved to: {os.path.join(output_dir, 'umap_embeddings_true_labels.png')}")


    # --- Plot UMAP of Embeddings colored by Predicted Labels ---
    print("Generating UMAP plot for embeddings (Predicted Labels)...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], c=pred_labels, cmap='viridis', alpha=0.7)
    plt.title(f'UMAP of Embeddings (Predicted Labels){eval_title_suffix}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.colorbar(scatter, ticks=np.arange(num_classes), label='Predicted Label')
    plt.clim(-0.5, num_classes - 0.5)
    handles, _ = scatter.legend_elements(num_classes=num_classes)
    plt.legend(handles=handles, labels=label_names, title="Predicted Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_embeddings_predicted_labels.png")) # OUTPUT: UMAP plot by predicted labels
    plt.close()
    print(f"✅ UMAP of embeddings (Predicted Labels) plot saved to: {os.path.join(output_dir, 'umap_embeddings_predicted_labels.png')}")


    # --- Plot t-SNE of Embeddings colored by True Labels ---
    print("Generating t-SNE plot for embeddings (True Labels)... This may take a moment.")
    tsne_reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000) # Default t-SNE parameters
    embeddings_2d_tsne = tsne_reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
    plt.title(f't-SNE of Embeddings (True Labels){eval_title_suffix}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ticks=np.arange(num_classes), label='True Label')
    plt.clim(-0.5, num_classes - 0.5)
    handles, _ = scatter.legend_elements(num_classes=num_classes)
    plt.legend(handles=handles, labels=label_names, title="True Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_embeddings_true_labels.png")) # OUTPUT: t-SNE plot by true labels
    plt.close()
    print(f"✅ t-SNE of embeddings (True Labels) plot saved to: {os.path.join(output_dir, 'tsne_embeddings_true_labels.png')}")


    # --- Plot t-SNE of Embeddings colored by Predicted Labels ---
    print("Generating t-SNE plot for embeddings (Predicted Labels)... This may take a moment.")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], c=pred_labels, cmap='viridis', alpha=0.7)
    plt.title(f't-SNE of Embeddings (Predicted Labels){eval_title_suffix}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ticks=np.arange(num_classes), label='Predicted Label')
    plt.clim(-0.5, num_classes - 0.5)
    handles, _ = scatter.legend_elements(num_classes=num_classes)
    plt.legend(handles=handles, labels=label_names, title="Predicted Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_embeddings_predicted_labels.png")) # OUTPUT: t-SNE plot by predicted labels
    plt.close()
    print(f"✅ t-SNE of embeddings (Predicted Labels) plot saved to: {os.path.join(output_dir, 'tsne_embeddings_predicted_labels.png')}")


    print(f"--- Model Evaluation Complete ({data_split.upper()}) ---")


# ====================================================================================================
# Main execution block
# ====================================================================================================
if __name__ == "__main__":
    # Define the data directory where preprocessed outputs are stored
    PREPROCESSED_DATA_DIR = "PIXEL_IMAGES"

    # --- Training the model ---
    print(f"\n--- Starting Full Training for {PREPROCESSED_DATA_DIR} ---")
    trained_model, train_losses, val_losses = train_model(
        data_dir=PREPROCESSED_DATA_DIR,
        epochs=150, # Run for the full 150 epochs as agreed
        batch_size=16,
        lr=3e-4,
        alpha=0.3,
        eval_interval=5 # Still evaluate validation loss every 5 epochs
    )
    print(f"--- Full Training Complete ---")

    # --- Evaluating the model on the training/validation data ---
    # Evaluate on the validation split
    evaluate_model(data_dir=PREPROCESSED_DATA_DIR, data_split='validation')
    # Evaluate on the entire dataset (including training and validation data)
    evaluate_model(data_dir=PREPROCESSED_DATA_DIR, data_split='all')

    print("\nTraining and Evaluation script finished.")