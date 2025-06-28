import random
import shutil
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
# Assuming this script is run from the same directory as FINAL_ALIGNED_LEAVES_512x512
BASE_DIR = Path(".")
SOURCE_OVERLAYS_DIR = BASE_DIR / "FINAL_ALIGNED_LEAVES_512x512" / "OVERLAYS"

# Define the expected main dataset prefixes
DATASET_PREFIXES = ["ALGERIA", "UCDAVIS", "VINEYARDS", "WOLFSKILL"]

# --- Training Sample Size per Main Dataset ---
# We will sample this many images from each of the DATASET_PREFIXES.
# Based on the UCDAVIS count (119), 100 is a good, balanced starting point.
SAMPLES_PER_MAIN_DATASET = 100

# Output directory for your annotation images
OUTPUT_BASE_DIR = BASE_DIR / "_training_data_landmarks_v1"
OUTPUT_OVERLAYS_DIR = OUTPUT_BASE_DIR / "OVERLAYS"

# --- Helper Function to Parse Filenames (re-used from previous script) ---
def parse_leaf_filename(filename_str):
    """
    Parses a filename to extract the main dataset and the sub-class name.
    Expected format: DATASET_Sub-class_ID_suffix.png
    """
    if filename_str.endswith("_mask.png"):
        base_name = filename_str[:-len("_mask.png")]
    elif filename_str.endswith("_overlay.png"):
        base_name = filename_str[:-len("_overlay.png")]
    elif filename_str.endswith("_rgb_crop.png"):
        base_name = filename_str[:-len("_rgb_crop.png")]
    else:
        return None, None

    parts_without_id = base_name.rsplit('_', 1)
    if len(parts_without_id) < 2:
        return None, None

    name_without_id_and_suffix = parts_without_id[0]
    first_underscore_idx = name_without_id_and_suffix.find('_')

    if first_underscore_idx == -1:
        return None, None

    main_dataset = name_without_id_and_suffix[:first_underscore_idx]
    sub_class = name_without_id_and_suffix[first_underscore_idx + 1:]

    if main_dataset not in DATASET_PREFIXES:
        return None, None

    return main_dataset, sub_class

# --- Main Sampling and Copying Logic ---
def main():
    if not SOURCE_OVERLAYS_DIR.exists():
        print(f"Error: Source directory not found at {SOURCE_OVERLAYS_DIR}")
        print("Please ensure your 'FINAL_ALIGNED_LEAVES_512x512/OVERLAYS' folder exists.")
        return

    # Create output directories if they don't exist
    OUTPUT_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {OUTPUT_OVERLAYS_DIR}")

    # Group all available overlay files by their main dataset
    all_files_by_dataset = defaultdict(list)
    print("\nGathering all image paths by dataset...")
    for filepath in SOURCE_OVERLAYS_DIR.iterdir():
        if filepath.is_file() and filepath.suffix == '.png':
            main_dataset, _ = parse_leaf_filename(filepath.name)
            if main_dataset:
                all_files_by_dataset[main_dataset].append(filepath)
            # else: We'll ignore unrecognized files for sampling purposes

    print("\n--- Sampling Images for Annotation ---")
    selected_files_for_annotation = []
    total_sampled_count = 0

    for dataset_prefix in DATASET_PREFIXES:
        available_files = all_files_by_dataset.get(dataset_prefix, [])
        num_available = len(available_files)

        # Determine how many to sample from this dataset
        num_to_sample = min(SAMPLES_PER_MAIN_DATASET, num_available)

        if num_available == 0:
            print(f"Warning: No files found for dataset '{dataset_prefix}'. Skipping.")
        elif num_to_sample < SAMPLES_PER_MAIN_DATASET:
            print(f"Note: Only {num_available} files available for '{dataset_prefix}'. Sampling all {num_available}.")
            selected_samples = available_files # Take all available if less than target
        else:
            # Randomly select files
            selected_samples = random.sample(available_files, num_to_sample)
            print(f"Successfully sampled {num_to_sample} files from '{dataset_prefix}'.")

        selected_files_for_annotation.extend(selected_samples)
        total_sampled_count += num_to_sample

    print(f"\nTotal images selected for annotation: {total_sampled_count}")
    print(f"Copying selected images to '{OUTPUT_OVERLAYS_DIR}'...")

    # Copy the selected files
    for src_filepath in selected_files_for_annotation:
        dest_filepath = OUTPUT_OVERLAYS_DIR / src_filepath.name
        try:
            shutil.copy(src_filepath, dest_filepath)
        except Exception as e:
            print(f"Error copying {src_filepath.name}: {e}")

    print("\n--- Sampling Complete! ---")
    print(f"Your {total_sampled_count} training overlay images are ready in: {OUTPUT_OVERLAYS_DIR}")
    print("You can now begin manually annotating the Base and Tip points on these images.")
    print("Remember to save the coordinates for each image in a simple TXT file.")

if __name__ == "__main__":
    main()