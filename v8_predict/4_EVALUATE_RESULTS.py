import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ====================================================================================================
# USER CONFIGURABLE INPUTS
# ====================================================================================================

# Path to your prediction results CSV file
PREDICTION_RESULTS_CSV = "prediction_results.csv"

# Output directory for plots
OUTPUT_PLOT_DIR = "PREDICTION_ANALYSIS_PLOTS"

# ====================================================================================================
# Script
# ====================================================================================================

def analyze_prediction_results(csv_path, output_dir):
    """
    Analyzes the prediction results CSV to generate a confusion matrix and box plots of PC values.

    Args:
        csv_path (str): Path to the prediction results CSV file.
        output_dir (str): Directory to save the generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Analyzing Prediction Results from: '{csv_path}' ---")
    print(f"  Saving plots to: '{output_dir}'")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'. Please check the path.")
        return

    if df.empty:
        print("Error: CSV file is empty. No data to analyze.")
        return

    # Ensure required columns exist
    required_columns = ['class', 'predicted_class', 'predicted_PC1', 'predicted_PC2']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing one or more required columns in CSV. Expected: {required_columns}")
        print(f"Found columns: {df.columns.tolist()}")
        return

    true_labels = df['class'].values
    predicted_labels = df['predicted_class'].values

    # Get unique classes in a sorted order for consistent plotting
    # This will include all unique values from both 'class' and 'predicted_class'
    all_unique_classes = sorted(list(set(true_labels) | set(predicted_labels)))
    print(f"\nDiscovered {len(all_unique_classes)} unique classes/varieties for plotting:")
    for cls in all_unique_classes:
        print(f"  - {cls}")

    # --- 1. Confusion Matrix ---
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_unique_classes)

    fig, ax = plt.subplots(figsize=(12, 10)) # Adjust size for potentially many classes
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d') # Display counts
    ax.set_title('Confusion Matrix for Prediction Results (True vs. Predicted Class)')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "prediction_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Confusion Matrix saved to: '{cm_path}'")

    # --- 2. Boxplot of Predicted PC1 by Class ---
    print("Generating Boxplot for Predicted PC1 by Class...")
    plt.figure(figsize=(12, 7)) # Adjust size as needed
    sns.boxplot(x='class', y='predicted_PC1', data=df, order=all_unique_classes, palette='viridis')
    plt.title('Predicted PC1 Values per True Class')
    plt.xlabel('True Class')
    plt.ylabel('Predicted PC1 Value')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pc1_boxplot_path = os.path.join(output_dir, "predicted_pc1_boxplot_by_true_class.png")
    plt.savefig(pc1_boxplot_path)
    plt.close()
    print(f"✅ Predicted PC1 Boxplot saved to: '{pc1_boxplot_path}'")

    # --- 3. Boxplot of Predicted PC2 by Class ---
    print("Generating Boxplot for Predicted PC2 by Class...")
    plt.figure(figsize=(12, 7)) # Adjust size as needed
    sns.boxplot(x='class', y='predicted_PC2', data=df, order=all_unique_classes, palette='viridis')
    plt.title('Predicted PC2 Values per True Class')
    plt.xlabel('True Class')
    plt.ylabel('Predicted PC2 Value')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pc2_boxplot_path = os.path.join(output_dir, "predicted_pc2_boxplot_by_true_class.png")
    plt.savefig(pc2_boxplot_path)
    plt.close()
    print(f"✅ Predicted PC2 Boxplot saved to: '{pc2_boxplot_path}'")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    import os
    analyze_prediction_results(
        csv_path=PREDICTION_RESULTS_CSV,
        output_dir=OUTPUT_PLOT_DIR
    )