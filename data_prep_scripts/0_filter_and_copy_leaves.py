import pandas as pd
from pathlib import Path
import shutil
import os
from tqdm import tqdm # For progress bars

# --- Configuration ---

# Base directory where PIXEL_SEGMENTATION is located, and where the new folder will be created
BASE_DIR = Path(".")

# Source data root folder
PIXEL_SEGMENTATION_ROOT = BASE_DIR / "PIXEL_SEGMENTATION"

# Destination folder for filtered leaves
DEST_ROOT = BASE_DIR / "TRUE_LEAVES_FILTERED"
DEST_MASKS_DIR = DEST_ROOT / "MASKS"
DEST_RGB_CROPS_DIR = DEST_ROOT / "RGB_CROPS"
DEST_OVERLAYS_DIR = DEST_ROOT / "OVERLAYS"

# List of all destination directories to create
ALL_DEST_DIRS = [DEST_MASKS_DIR, DEST_RGB_CROPS_DIR, DEST_OVERLAYS_DIR]

# Execution log file
EXECUTION_LOG_FILE = BASE_DIR / "leaf_filtering_log_v2.txt" # New log file for this run

# --- Dataset Configurations and Filters ---
# Each entry defines a dataset, its metadata, its base path for image components, and filter criteria.
# The 'get_component_base_path_func' is crucial for handling complex structures like California.
# Filter criteria are now INCLUSIVE (i.e., these are the conditions for files to be COPIED).
DATASET_CONFIGS = {
    "ALGERIA": {
        "metadata_path": PIXEL_SEGMENTATION_ROOT / "ALGERIA_PIXEL_SEG" / "INFERENCE_OUTPUTS" / "component_metadata.csv",
        "filter_criteria": lambda df: df["vein_pixels"] >= 20000, # INVERTED: Keep if vein_pixels >= 20000
        "get_component_base_path_func": lambda dataset_row, dataset_root: \
            PIXEL_SEGMENTATION_ROOT / "ALGERIA_PIXEL_SEG" / "INFERENCE_OUTPUTS"
    },
    "WOLFSKILL": {
        "metadata_path": PIXEL_SEGMENTATION_ROOT / "WOLFSKILL_PIXEL_SEG" / "INFERENCE_OUTPUTS" / "component_metadata.csv",
        "filter_criteria": lambda df: (df["total_bbox_pixels"] >= 50000) & (df["total_bbox_pixels"] <= 2000000), # INVERTED: Keep if 50000 <= total_bbox_pixels <= 2000000
        "get_component_base_path_func": lambda dataset_row, dataset_root: \
            PIXEL_SEGMENTATION_ROOT / "WOLFSKILL_PIXEL_SEG" / "INFERENCE_OUTPUTS"
    },
    # --- CALIFORNIA DATASETS ---
    # The 'metadata_path' points to the shared California CSV.
    # The 'get_component_base_path_func' now correctly builds the path for the specific sub-dataset.
    # The 'sub_dataset_prefix' is new, to help filter rows from the California CSV.
    "CALIFORNIA_CROSSES": {
        "metadata_path": PIXEL_SEGMENTATION_ROOT / "CALIFORNIA_PIXEL_SEG" / "INFERENCE_OUTPUTS" / "component_metadata_renamed.csv",
        "filter_criteria": lambda df: df["total_bbox_pixels"] >= 300000, # INVERTED: Keep if total_bbox_pixels >= 300000
        "sub_dataset_prefix": "CROSSES_", # NEW: Prefix to filter rows in the shared California CSV
        "get_component_base_path_func": lambda dataset_row, dataset_root: \
            PIXEL_SEGMENTATION_ROOT / "CALIFORNIA_PIXEL_SEG" / "INFERENCE_OUTPUTS" / "CROSSES"
    },
    "CALIFORNIA_VINEYARDS": {
        "metadata_path": PIXEL_SEGMENTATION_ROOT / "CALIFORNIA_PIXEL_SEG" / "INFERENCE_OUTPUTS" / "component_metadata_renamed.csv",
        "filter_criteria": lambda df: df["total_bbox_pixels"] >= 300000, # INVERTED: Keep if total_bbox_pixels >= 300000
        "sub_dataset_prefix": "VINEYARDS_", # NEW
        "get_component_base_path_func": lambda dataset_row, dataset_root: \
            PIXEL_SEGMENTATION_ROOT / "CALIFORNIA_PIXEL_SEG" / "INFERENCE_OUTPUTS" / "VINEYARDS"
    },
    "CALIFORNIA_UCDAVIS": {
        "metadata_path": PIXEL_SEGMENTATION_ROOT / "CALIFORNIA_PIXEL_SEG" / "INFERENCE_OUTPUTS" / "component_metadata_renamed.csv",
        "filter_criteria": lambda df: df["total_bbox_pixels"] >= 50000, # INVERTED: Keep if total_bbox_pixels >= 50000
        "sub_dataset_prefix": "UCDAVIS_", # NEW
        "get_component_base_path_func": lambda dataset_row, dataset_root: \
            PIXEL_SEGMENTATION_ROOT / "CALIFORNIA_PIXEL_SEG" / "INFERENCE_OUTPUTS" / "UCDAVIS"
    }
}

# --- Main Logic ---
def main():
    log_messages = []
    def log_and_print(message, is_error=False):
        log_messages.append(message)
        if is_error:
            print(f"ERROR: {message}")
        else:
            print(message)

    log_and_print(f"--- Starting Leaf Filtering and Copy Process (v2) ---")
    log_and_print(f"Source root: {PIXEL_SEGMENTATION_ROOT}")
    log_and_print(f"Destination root: {DEST_ROOT}\n")

    # 1. Create destination directories
    log_and_print(f"Creating destination directories...")
    for d in ALL_DEST_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        log_and_print(f"  Created/Ensured: {d}")
    log_and_print(f"Destination directories ready.\n")

    overall_copied_count = 0
    overall_skipped_count = 0

    # Process each dataset
    for dataset_name, config in DATASET_CONFIGS.items():
        log_and_print(f"\n--- Processing Dataset: {dataset_name} ---")
        metadata_path = config["metadata_path"]
        filter_func = config["filter_criteria"]
        get_component_base_path = config["get_component_base_path_func"]
        sub_dataset_prefix = config.get("sub_dataset_prefix") # Get this if it exists (for California)

        if not metadata_path.exists():
            log_and_print(f"  ERROR: Metadata file not found for {dataset_name} at {metadata_path}. Skipping this dataset.", is_error=True)
            continue

        try:
            df_raw = pd.read_csv(metadata_path)
            log_and_print(f"  Loaded raw metadata from {metadata_path} ({len(df_raw)} rows).")

            df_metadata = df_raw.copy() # Start with the full DataFrame

            # Special handling for California datasets to filter the main CSV by prefix
            if sub_dataset_prefix:
                # Assuming component_name starts with the sub_dataset_prefix (e.g., 'CROSSES_')
                df_metadata = df_raw[df_raw['component_name'].astype(str).str.startswith(sub_dataset_prefix, na=False)].copy()
                log_and_print(f"  Filtered metadata for '{dataset_name}' using prefix '{sub_dataset_prefix}' resulted in {len(df_metadata)} relevant rows.")

        except Exception as e:
            log_and_print(f"  ERROR reading or preparing metadata for {dataset_name} from {metadata_path}: {e}. Skipping.", is_error=True)
            continue

        # Initialize counters for this dataset
        copied_count = 0
        skipped_count = 0

        # Apply the filter based on criteria
        try:
            # Ensure filter columns exist before applying the filter
            required_cols = []
            if "vein_pixels" in str(filter_func.__code__.co_consts): # crude way to check func string for column name
                 required_cols.append("vein_pixels")
            if "total_bbox_pixels" in str(filter_func.__code__.co_consts):
                 required_cols.append("total_bbox_pixels")

            missing_cols = [col for col in required_cols if col not in df_metadata.columns]
            if missing_cols:
                log_and_print(f"  ERROR: Missing required columns {missing_cols} for filter in {dataset_name} metadata. Skipping filter for this dataset.", is_error=True)
                df_filtered = pd.DataFrame() # No files will be copied
            else:
                df_filtered = df_metadata[filter_func(df_metadata)].copy()
                log_and_print(f"  {len(df_filtered)} files passed the filter criteria for {dataset_name}.")

            # Now, calculate skipped_count from the original df_metadata for this specific sub-dataset
            skipped_count = len(df_metadata) - len(df_filtered)

        except Exception as e:
            log_and_print(f"  ERROR applying filter for {dataset_name}: {e}. Skipping filter for this dataset.", is_error=True)
            df_filtered = pd.DataFrame() # No files will be copied
            skipped_count = len(df_metadata) # All rows are skipped if filter fails


        # Iterate through filtered rows and copy files
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Copying {dataset_name} files"):
            try:
                # Determine the correct base path for components (COMPONENT_MASKS, etc.)
                component_base_path = get_component_base_path(row, PIXEL_SEGMENTATION_ROOT)

                # Get filenames from metadata
                mask_filename = row.get('mask_file')
                rgb_crop_filename = row.get('rgb_crop_file')
                overlay_filename = row.get('overlay_file')

                # Define source paths
                source_mask_path = component_base_path / "COMPONENT_MASKS" / str(mask_filename) if pd.notna(mask_filename) else None
                source_rgb_crop_path = component_base_path / "COMPONENT_RGB_CROPS" / str(rgb_crop_filename) if pd.notna(rgb_crop_filename) else None
                source_overlay_path = component_base_path / "COMPONENT_OVERLAYS" / str(overlay_filename) if pd.notna(overlay_filename) else None

                # Define destination paths
                dest_mask_path = DEST_MASKS_DIR / source_mask_path.name if source_mask_path else None
                dest_rgb_crop_path = DEST_RGB_CROPS_DIR / source_rgb_crop_path.name if source_rgb_crop_path else None
                dest_overlay_path = DEST_OVERLAYS_DIR / source_overlay_path.name if source_overlay_path else None

                # Collect paths to copy and check existence
                files_to_copy_info = []
                if source_mask_path and pd.notna(mask_filename): files_to_copy_info.append((source_mask_path, dest_mask_path))
                if source_rgb_crop_path and pd.notna(rgb_crop_filename): files_to_copy_info.append((source_rgb_crop_path, dest_rgb_crop_path))
                if source_overlay_path and pd.notna(overlay_filename): files_to_copy_info.append((source_overlay_path, dest_overlay_path))

                if not files_to_copy_info:
                    log_and_print(f"    WARNING: No valid file paths found in metadata for component {row.get('component_name', 'N/A')}. Skipping.", is_error=True)
                    skipped_count += 1 # Count as skipped if no files to process
                    continue

                all_present_and_copied = True
                for src_path, dest_path in files_to_copy_info:
                    if not src_path.exists():
                        log_and_print(f"    WARNING: Source file NOT FOUND for {row.get('component_name', 'N/A')} ({src_path.name}). Skipping this component.", is_error=True)
                        all_present_and_copied = False
                        break # Skip remaining files for this component if one is missing

                    if dest_path.exists():
                        # log_and_print(f"    INFO: Destination file {dest_path.name} already exists. Skipping copy for this file.")
                        pass # Don't re-copy if already there.
                    else:
                        shutil.copy2(src_path, dest_path)

                if all_present_and_copied:
                    copied_count += 1
                else:
                    # If not all files were present/copied, it was already counted as skipped above
                    pass # Don't double count if `all_present_and_copied` became False

            except Exception as e:
                log_and_print(f"  ERROR processing row {idx} (component: {row.get('component_name', 'N/A')}): {e}. Skipping component.", is_error=True)
                skipped_count += 1 # Count as skipped for error
                continue

        log_and_print(f"\n--- Results for {dataset_name} ---")
        log_and_print(f"  Files copied: {copied_count}")
        log_and_print(f"  Files skipped (did not meet criteria or had errors): {skipped_count}")
        overall_copied_count += copied_count
        overall_skipped_count += skipped_count

    log_and_print(f"\n--- Overall Results ---")
    log_and_print(f"Total files copied across all datasets: {overall_copied_count}")
    log_and_print(f"Total files skipped across all datasets: {overall_skipped_count}")
    log_and_print(f"Process complete. Check '{DEST_ROOT}' for your filtered files and '{EXECUTION_LOG_FILE}' for details.")

    # Save the full log to file
    with open(EXECUTION_LOG_FILE, 'w') as f:
        f.writelines(line + '\n' for line in log_messages)


if __name__ == "__main__":
    main()