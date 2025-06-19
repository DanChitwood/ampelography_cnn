import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import color, filters
from skimage.measure import label, regionprops
from scipy import ndimage
import pandas as pd
import re
import json
from pathlib import Path # For robust path handling
from collections import defaultdict # For UCDAVIS lookup
import itertools # For cycling through iterators

# --- GLOBAL CONFIGURATION ---
# Base directory for all input inference images.
INFERENCE_INPUT_ROOT = "INFERENCE_INPUT"

# Paths to metadata CSVs (assumed to be in the same directory as the script)
CROSSES_KEY_CSV_PATH = "crosses_key.csv"
VINEYARDS_KEY_CSV_PATH = "vineyards_key.csv" # <--- Ensure this path is correct now!
# Directories containing UCDAVIS info CSVs (assumed to be in the same directory as the script)
MSU_DATA_DIR = "msu_data"
UCD_DATA_DIR = "ucd_data"

# Path to your best saved model checkpoint.
BEST_MODEL_PATH = "V2_best_model_vein_dice_0.7077_epoch42.pt" # <--- UPDATE THIS!

# Base directory for all prediction outputs.
# Output structure: INFERENCE_OUTPUTS/<DatasetFolder>/COMPONENT_MASKS/...
OUTPUT_PREDICTIONS_DIR = "INFERENCE_OUTPUTS"

# Path to the preprocessing config JSON (from your synthetic training data)
PREPROCESSING_CONFIG_PATH = "processed_data/config/preprocessing_config_for_synthetic.json"

# Define common image extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}

# Model parameters
NUM_SEG_CLASSES = 3 # Background, Blade, Vein
IN_CHANNELS = 11 # Corrected: 'IN_CHANNELS' (all caps)
NUM_GEO_CHANNELS = 1

# Filter parameter for connected components
MIN_BBOX_DIMENSION = 50 # Ignore components if any bounding box dimension is less than 50 pixels

# Overlay parameters (retained from WOLFSKILL script for component-level overlays)
VEIN_OVERLAY_COLORMAP_NAME = 'plasma'
VEIN_ALPHA = 1.0 # Alpha for vein overlay. Keep 1.0 for full opacity
BLADE_OVERLAY_ALPHA = 0.7 # Alpha for blade overlay on the black background

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# === LOAD PREPROCESSING CONFIGURATION ===
try:
    if not os.path.exists(PREPROCESSING_CONFIG_PATH):
        raise FileNotFoundError(f"Preprocessing configuration file not found at: {PREPROCESSING_CONFIG_PATH}\n"
                                "Please ensure your preprocessing script has been run and saved the config.")
    with open(PREPROCESSING_CONFIG_PATH, 'r') as f:
        PREPROCESSING_CONFIG = json.load(f)

    TARGET_WIDTH = PREPROCESSING_CONFIG["TARGET_SIZE"][0]
    TARGET_HEIGHT = PREPROCESSING_CONFIG["TARGET_SIZE"][1]
    TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)
    SATO_SIGMAS = PREPROCESSING_CONFIG["sato_sigmas"]
    MEIJERING_SIGMAS = PREPROCESSING_CONFIG["meijering_sigmas"]
    FRANGI_SIGMAS = PREPROCESSING_CONFIG["frangi_sigmas"]
    HESSIAN_SIGMAS = PREPROCESSING_CONFIG["hessian_sigmas"]
    ENHANCE_PERCENTILE = PREPROCESSING_CONFIG["ENHANCE_PERCENTILE"]
    print(f"Preprocessing configuration loaded from {PREPROCESSING_CONFIG_PATH}")
    print(f"Target size: {TARGET_SIZE}, Enhance Percentile: {ENHANCE_PERCENTILE}")
except Exception as e:
    print(f"Error loading preprocessing config: {e}")
    print("Please ensure PREPROCESSING_CONFIG_PATH is correct and the JSON file is valid.")
    exit() # Exit if config cannot be loaded, as it's critical for preprocessing


# --- HELPER FUNCTIONS (PREPROCESSING - copied from previous script) ---
def rotate_to_wide(image_pil):
    """Rotates an image so its width is greater than its height."""
    width, height = image_pil.size
    rotation_applied = False
    if height > width:
        image_pil = image_pil.transpose(Image.Transpose.ROTATE_270)
        rotation_applied = True
    return image_pil, rotation_applied

def rescale_and_pad_image(image_pil, target_size):
    """
    Rescales an image to fit within target_size while maintaining aspect ratio,
    then pads with white (for RGB) or black (for L) to reach target_size.
    Also returns the paste offset, scaled dimensions, and scale factor.
    """
    original_width, original_height = image_pil.size
    target_width, target_height = target_size

    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale_factor = min(scale_w, scale_h)

    if original_width == 0 or original_height == 0:
        return Image.new("RGB", target_size, (255, 255, 255)), (0, 0), (0, 0), 0.0

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    scaled_img = image_pil.resize((new_width, new_height), Image.LANCZOS)

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    padded_img = Image.new("RGB", target_size, (255, 255, 255))
    padded_img.paste(scaled_img, (paste_x, paste_y))

    return padded_img, (paste_x, paste_y), (new_width, new_height), scale_factor

def enhance_contrast(arr, percentile_val):
    """Applies contrast enhancement based on percentile."""
    if arr.size == 0:
        return np.array([])
    vmax = np.percentile(arr, percentile_val)
    if vmax == 0:
        return np.zeros_like(arr, dtype=np.float32)
    arr_clipped = np.clip(arr, 0, vmax)
    arr_rescaled = arr_clipped / vmax
    return arr_rescaled.astype(np.float32)

def apply_ridge_filters(image_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p):
    """
    Applies various ridge filters to a grayscale image and returns their enhanced outputs.
    Generates both black_ridges=True and False versions for Sato, Meijering, Frangi.
    Generates both black_ridges=True and False versions for Hessian.
    Takes a PIL Image that is already padded to TARGET_SIZE.
    """
    image_rgb_float = np.array(image_pil_padded).astype(np.float32) / 255.0

    if image_rgb_float.ndim == 3 and image_rgb_float.shape[2] == 3:
        gray_image = color.rgb2gray(image_rgb_float)
    elif image_rgb_float.ndim == 2:
        gray_image = image_rgb_float
    else:
        raise ValueError(f"Unexpected image dimensions or mode for filter application: {image_rgb_float.shape}")

    sato_br_false_raw = filters.sato(gray_image, sigmas=sato_s, black_ridges=False, mode='reflect')
    sato_br_true_raw = filters.sato(gray_image, sigmas=sato_s, black_ridges=True, mode='reflect')
    sato_br_false_processed = enhance_contrast(sato_br_false_raw, enhance_p)
    sato_br_true_processed = enhance_contrast(sato_br_true_raw, enhance_p)

    meijering_br_false_raw = filters.meijering(gray_image, sigmas=meijering_s, black_ridges=False, mode='reflect')
    meijering_br_true_raw = filters.meijering(gray_image, sigmas=meijering_s, black_ridges=True, mode='reflect')
    meijering_br_false_processed = enhance_contrast(meijering_br_false_raw, enhance_p)
    meijering_br_true_processed = enhance_contrast(meijering_br_true_raw, enhance_p)

    frangi_br_false_raw = filters.frangi(gray_image, sigmas=frangi_s, black_ridges=False, mode='reflect')
    frangi_br_true_raw = filters.frangi(gray_image, sigmas=frangi_s, black_ridges=True, mode='reflect')
    frangi_br_false_processed = enhance_contrast(frangi_br_false_raw, enhance_p)
    frangi_br_true_processed = enhance_contrast(frangi_br_true_raw, enhance_p)

    hessian_br_true_raw = filters.hessian(gray_image, sigmas=hessian_s, black_ridges=True, mode='reflect')
    hessian_br_false_raw = filters.hessian(gray_image, sigmas=hessian_s, black_ridges=False, mode='reflect')
    hessian_br_true_processed = enhance_contrast(hessian_br_true_raw, enhance_p)
    hessian_br_false_processed = enhance_contrast(hessian_br_false_raw, enhance_p)

    return (sato_br_false_processed, sato_br_true_processed,
            meijering_br_false_processed, meijering_br_true_processed,
            frangi_br_false_processed, frangi_br_true_processed,
            hessian_br_true_processed, hessian_br_false_processed) # 8 channels

def create_11channel_input(img_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p):
    """
    Creates the 11-channel input array for the UNet model.
    Channels: L, A, B, Sato_F, Sato_T, Meijering_F, Meijering_T, Frangi_F, Frangi_T, Hessian_T, Hessian_F.
    """
    img_rgb_float = np.array(img_pil_padded).astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_rgb_float)

    L_channel = img_lab[:, :, 0] / 100.0
    A_channel = (img_lab[:, :, 1] + 128) / 255.0
    B_channel = (img_lab[:, :, 2] + 128) / 255.0

    sato_f, sato_t, meijering_f, meijering_t, frangi_f, frangi_t, hessian_t, hessian_f = \
        apply_ridge_filters(img_pil_padded, sato_s, meijering_s, frangi_s, hessian_s, enhance_p)

    eleven_channels = np.stack([L_channel, A_channel, B_channel,
                                sato_f, sato_t, meijering_f, meijering_t,
                                frangi_f, frangi_t, hessian_t, hessian_f], axis=-1)

    return eleven_channels

# --- String Sanitization Function (retained) ---
def sanitize_filename_string(s):
    """
    Sanitizes a string to be suitable for use in a filename.
    Replaces problematic characters with underscores.
    """
    s = str(s).strip() # Ensure it's a string
    # Replace any character that is not a letter, number, hyphen, underscore, or dot with an underscore
    s = re.sub(r'[^\w\s\-\.]', '_', s)
    # Replace spaces with underscores
    s = s.replace(' ', '_')
    # Remove multiple consecutive underscores
    s = re.sub(r'_{2,}', '_', s)
    # Remove leading/trailing underscores
    s = s.strip('_')
    return s

# ===================== U-Net Model Definition (copied from previous script) =====================
class UNet(nn.Module):
    def __init__(self, in_channels, num_seg_classes, num_geo_channels):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(512, 256), CBR(256, 256))

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.seg_out_conv = nn.Conv2d(64, num_seg_classes, 1)
        self.geo_out_conv = nn.Sequential(
            nn.Conv2d(64, num_geo_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.up4(e4)
        if d4.shape != e3.shape:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if d3.shape != e2.shape:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape != e1.shape:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        seg_output = self.seg_out_conv(d2)
        geo_output = self.geo_out_conv(d2)

        return seg_output, geo_output


# --- METADATA LOOKUP LOADERS ---

def load_crosses_key(csv_path):
    """Loads crosses_key.csv mapping 'Scan Name' to 'Population'."""
    crosses_lookup = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                scan_name = str(row['Scan Name']).strip()
                population = str(row['Population']).strip()
                if scan_name: # Ensure scan_name is not empty
                    crosses_lookup[scan_name] = population
        except Exception as e:
            print(f"Error loading crosses_key.csv from {csv_path}: {e}")
    else:
        print(f"Warning: crosses_key.csv not found at {csv_path}. CROSSES images will use 'UNKNOWN_ID'.")
    return crosses_lookup

def load_vineyards_key(csv_path):
    """Loads vineyards.csv mapping 'sample' to concatenated fields."""
    vineyards_lookup = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            required_cols = ["sample", "rootstock", "scion", "location", "year"]
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {csv_path} is missing one or more required columns ({required_cols}). VINEYARDS images will use 'UNKNOWN_ID'.")
                return vineyards_lookup

            for _, row in df.iterrows():
                sample_name = str(row['sample']).strip()
                if sample_name:
                    # Concatenate required fields, handling potential NaNs
                    parts = [
                        str(row['rootstock']).strip() if pd.notna(row['rootstock']) else "",
                        str(row['scion']).strip() if pd.notna(row['scion']) else "",
                        str(row['location']).strip() if pd.notna(row['location']) else "",
                        str(row['year']).strip() if pd.notna(row['year']) else ""
                    ]
                    # Filter out empty parts before joining
                    concatenated_id = "_".join(filter(None, parts))
                    if not concatenated_id: # If all parts were empty
                        concatenated_id = "MISSING_VINEYARD_META"
                    vineyards_lookup[sample_name] = concatenated_id
        except Exception as e:
            print(f"Error loading vineyards.csv from {csv_path}: {e}")
    else:
        print(f"Warning: vineyards.csv not found at {csv_path}. VINEYARDS images will use 'UNKNOWN_ID'.")
    return vineyards_lookup

def build_ucdavis_lookup(msu_dir, ucd_dir):
    """
    Builds a lookup for UCDAVIS images from *_info.csv files.
    Maps full image filename (including extension) to 'species' value.
    Handles whitespace and case-insensitivity for 'factor' names.
    """
    ucdavis_lookup = {}
    info_csv_paths = []

    for base_dir in [msu_dir, ucd_dir]:
        if not os.path.exists(base_dir):
            print(f"Warning: UCDAVIS metadata directory '{base_dir}' not found. Skipping.")
            continue
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith('_info.csv'):
                    info_csv_paths.append(Path(root) / f)

    if not info_csv_paths:
        print(f"Warning: No *_info.csv files found in '{msu_dir}' or '{ucd_dir}'. UCDAVIS images will use 'UNKNOWN_ID'.")
        return ucdavis_lookup

    print(f"Building UCDAVIS metadata lookup from {len(info_csv_paths)} info CSVs...")
    for csv_path in tqdm(info_csv_paths, desc="Parsing UCDAVIS info CSVs"):
        try:
            # Use pandas for robust CSV reading, including whitespace handling
            # `skipinitialspace=True` handles spaces after delimiters
            df = pd.read_csv(csv_path, skipinitialspace=True)

            current_image_filename = None
            current_species = None

            # Ensure 'factor' and 'value' columns exist and strip any whitespace from their names
            df.columns = df.columns.str.strip()
            if 'factor' not in df.columns or 'value' not in df.columns:
                print(f"Warning: CSV {csv_path} is missing 'factor' or 'value' columns. Skipping.")
                continue

            for _, row in df.iterrows():
                # Strip whitespace and convert factor to lowercase for robust matching
                factor = str(row['factor']).strip().lower() if pd.notna(row['factor']) else None
                value = str(row['value']).strip() if pd.notna(row['value']) else None # Only strip value, retain original case

                if factor and value:
                    if factor == 'image':
                        current_image_filename = value
                    elif factor == 'species':
                        current_species = value

            if current_image_filename and current_species:
                # Add to lookup, using the original casing of the filename and species value
                ucdavis_lookup[current_image_filename] = current_species
                # TEMPORARY DEBUG PRINT: Uncomment to see what's added to lookup
                print(f"  Added to UCDAVIS lookup: Image '{current_image_filename}' -> Species '{current_species}' from {csv_path}")
            # else:
            #     print(f"  Warning: Incomplete record for image in {csv_path} (missing image/species pair).")

        except pd.errors.EmptyDataError:
            print(f"Warning: UCDAVIS info CSV {csv_path} is empty. Skipping.")
            continue
        except Exception as e:
            print(f"Error parsing UCDAVIS info CSV {csv_path}: {e}")
            continue
    print(f"UCDAVIS lookup built with {len(ucdavis_lookup)} entries.")
    return ucdavis_lookup

# --- MAIN PREDICTION SCRIPT ---
def main():
    # --- Load Metadata Lookups ---
    crosses_lookup = load_crosses_key(CROSSES_KEY_CSV_PATH)
    vineyards_lookup = load_vineyards_key(VINEYARDS_KEY_CSV_PATH)
    ucdavis_lookup = build_ucdavis_lookup(MSU_DATA_DIR, UCD_DATA_DIR)

    # --- Prepare Output Directories ---
    # Top-level dataset folders
    OUTPUT_CROSSES_DIR = Path(OUTPUT_PREDICTIONS_DIR) / "CROSSES"
    OUTPUT_UCDAVIS_DIR = Path(OUTPUT_PREDICTIONS_DIR) / "UCDAVIS"
    OUTPUT_VINEYARDS_DIR = Path(OUTPUT_PREDICTIONS_DIR) / "VINEYARDS"

    # Create all required sub-subdirectories
    for base_output_dir in [OUTPUT_CROSSES_DIR, OUTPUT_UCDAVIS_DIR, OUTPUT_VINEYARDS_DIR]:
        os.makedirs(base_output_dir / "COMPONENT_MASKS", exist_ok=True)
        os.makedirs(base_output_dir / "COMPONENT_RGB_CROPS", exist_ok=True)
        os.makedirs(base_output_dir / "COMPONENT_OVERLAYS", exist_ok=True)

    METADATA_CSV_PATH = Path(OUTPUT_PREDICTIONS_DIR) / "component_metadata.csv"

    # --- Load Model ---
    model = UNet(in_channels=IN_CHANNELS, num_seg_classes=NUM_SEG_CLASSES, num_geo_channels=NUM_GEO_CHANNELS).to(DEVICE) # <--- FIXED TYPO HERE
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading model from {BEST_MODEL_PATH}...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Error: Model checkpoint not found at {BEST_MODEL_PATH}")
        return

    # --- Resumable Prediction Logic ---
    all_component_metadata = []
    processed_original_image_paths = set()
    global_component_serial_id = 1

    if os.path.exists(METADATA_CSV_PATH):
        print(f"Metadata CSV found at {METADATA_CSV_PATH}. Loading existing data for resuming...")
        try:
            existing_df = pd.read_csv(METADATA_CSV_PATH)
            all_component_metadata = existing_df.to_dict('records')
            processed_original_image_paths = set(existing_df['original_image_path'].unique())

            if not existing_df.empty:
                max_existing_id = 0
                # Parse component names to find the highest serial ID used across all datasets
                # The pattern is <DATASET_PREFIX><_><DERIVED_ID><_><SERIAL_ID>
                for comp_name in existing_df['component_name']:
                    try:
                        parts = comp_name.rsplit('_', 1)
                        if len(parts) > 1 and parts[-1].isdigit():
                            max_existing_id = max(max_existing_id, int(parts[-1]))
                    except ValueError:
                        pass # Ignore entries not conforming to expected naming
                global_component_serial_id = max_existing_id + 1

            print(f"Found {len(processed_original_image_paths)} unique original images already processed. Resuming...")
            print(f"Next global component ID will start from: {global_component_serial_id}")
        except pd.errors.EmptyDataError:
            print("Metadata CSV is empty. Starting fresh.")
            # Ensure the CSV is recreated with headers if it was empty
            metadata_df = pd.DataFrame(columns=[
                "original_image_path", "dataset_category", "derived_id",
                "component_name",
                "component_idx_in_original_image",
                "blade_pixels", "vein_pixels", "background_pixels_internal_holes",
                "total_bbox_pixels", "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
                "mask_file", "rgb_crop_file", "overlay_file"
            ])
            metadata_df.to_csv(METADATA_CSV_PATH, index=False)
        except Exception as e:
            print(f"Error loading existing metadata CSV: {e}. Starting fresh and overwriting existing CSV (if any issues).")
            # In case of any other read error, treat as new run.
            metadata_df = pd.DataFrame(columns=[
                "original_image_path", "dataset_category", "derived_id",
                "component_name",
                "component_idx_in_original_image",
                "blade_pixels", "vein_pixels", "background_pixels_internal_holes",
                "total_bbox_pixels", "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
                "mask_file", "rgb_crop_file", "overlay_file"
            ])
            metadata_df.to_csv(METADATA_CSV_PATH, index=False)
    else:
        print(f"Metadata CSV not found. Creating a new one at {METADATA_CSV_PATH}")
        # Create an empty DataFrame with headers to ensure file exists and has correct columns
        metadata_df = pd.DataFrame(columns=[
            "original_image_path", "dataset_category", "derived_id",
            "component_name",
            "component_idx_in_original_image",
            "blade_pixels", "vein_pixels", "background_pixels_internal_holes",
            "total_bbox_pixels", "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
            "mask_file", "rgb_crop_file", "overlay_file"
        ])
        metadata_df.to_csv(METADATA_CSV_PATH, index=False)

    # Get the colormap for vein overlay
    vein_cmap = plt.colormaps.get_cmap(VEIN_OVERLAY_COLORMAP_NAME)

    # --- Discover and categorize all images ---
    crosses_images_raw = []
    ucdavis_images_raw = []
    vineyards_images_raw = []
    top_level_folders = {"CROSSES", "UCDAVIS", "VINEYARDS"}

    print(f"Scanning for images in '{INFERENCE_INPUT_ROOT}'...")
    if not os.path.exists(INFERENCE_INPUT_ROOT):
        print(f"Error: INFERENCE_INPUT_ROOT '{INFERENCE_INPUT_ROOT}' does not exist. Please create it and place your images inside.")
        return

    for root, _, files in os.walk(INFERENCE_INPUT_ROOT):
        path_obj = Path(root)

        category_name = None
        for part in path_obj.parts:
            if part in top_level_folders:
                category_name = part
                break

        if category_name is None:
            continue

        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                full_image_path = Path(root) / file
                if category_name == "CROSSES":
                    crosses_images_raw.append((full_image_path, category_name))
                elif category_name == "UCDAVIS":
                    ucdavis_images_raw.append((full_image_path, category_name))
                elif category_name == "VINEYARDS":
                    vineyards_images_raw.append((full_image_path, category_name))

    print(f"Found {len(crosses_images_raw)} CROSSES images.")
    print(f"Found {len(ucdavis_images_raw)} UCDAVIS images.")
    print(f"Found {len(vineyards_images_raw)} VINEYARDS images.")

    # Filter out already processed images
    crosses_images_to_process = [(p, c) for p, c in crosses_images_raw if str(p) not in processed_original_image_paths]
    ucdavis_images_to_process = [(p, c) for p, c in ucdavis_images_raw if str(p) not in processed_original_image_paths]
    vineyards_images_to_process = [(p, c) for p, c in vineyards_images_raw if str(p) not in processed_original_image_paths]

    total_new_images_to_process = len(crosses_images_to_process) + len(ucdavis_images_to_process) + len(vineyards_images_to_process)
    print(f"Starting processing on {total_new_images_to_process} new images.")

    # Create iterators for each category
    crosses_iter = iter(crosses_images_to_process)
    ucdavis_iter = iter(ucdavis_images_to_process)
    vineyards_iter = iter(vineyards_images_to_process)

    # Store iterators and their names for cycling
    iterators = [crosses_iter, vineyards_iter, ucdavis_iter] # Order for cycling
    category_names_cycle = ["CROSSES", "VINEYARDS", "UCDAVIS"]

    # Track exhaustion of iterators
    iterator_exhausted = [False] * len(iterators)

    # --- Main Interleaved Processing Loop ---
    with tqdm(total=total_new_images_to_process, desc="Overall Image Processing") as pbar:
        while True:
            processed_this_cycle = False
            for i, current_iter in enumerate(iterators):
                if iterator_exhausted[i]:
                    continue # Skip if this iterator is already exhausted

                try:
                    image_path, category_name = next(current_iter)
                    processed_this_cycle = True
                    str_image_path = str(image_path)

                    # Determine dataset prefix and derived ID
                    dataset_prefix = f"{category_name.upper()}_"
                    derived_id_for_component = "UNKNOWN_ID"
                    image_filename_stem = image_path.stem # Name without extension
                    image_filename_full = image_path.name # Name with extension

                    if category_name == "CROSSES":
                        # Handle _1, _2 suffixes for CROSSES images
                        match = re.match(r'^(.*?)_\d+$', image_filename_stem)
                        if match:
                            base_stem_for_lookup = match.group(1)
                        else:
                            base_stem_for_lookup = image_filename_stem
                        derived_id_for_component = crosses_lookup.get(base_stem_for_lookup, "UNKNOWN_ID")
                        if derived_id_for_component == "UNKNOWN_ID":
                            tqdm.write(f"Warning: Scan Name '{base_stem_for_lookup}' (derived from {image_path.name}) not found in {CROSSES_KEY_CSV_PATH}. Using 'UNKNOWN_ID'.", end='\r')
                    elif category_name == "VINEYARDS":
                        derived_id_for_component = vineyards_lookup.get(image_filename_stem, "UNKNOWN_ID")
                        if derived_id_for_component == "UNKNOWN_ID":
                            tqdm.write(f"Warning: Sample '{image_filename_stem}' for VINEYARDS image {image_path.name} not found in {VINEYARDS_KEY_CSV_PATH}. Using 'UNKNOWN_ID'.", end='\r')
                    elif category_name == "UCDAVIS":
                        # For UCDAVIS, lookup using the full filename including extension
                        derived_id_for_component = ucdavis_lookup.get(image_filename_full, "UNKNOWN_ID")
                        if derived_id_for_component == "UNKNOWN_ID":
                            tqdm.write(f"Warning: Image filename '{image_filename_full}' for UCDAVIS image not found in *_info.csv files. Using 'UNKNOWN_ID'.")
                            if not ucdavis_lookup:
                                tqdm.write(f"  UCDAVIS lookup table is empty. Check paths to {MSU_DATA_DIR} and {UCD_DATA_DIR} and CSV contents.")
                            elif image_filename_full not in ucdavis_lookup:
                                tqdm.write(f"  '{image_filename_full}' not found in lookup. Keys present: {list(ucdavis_lookup.keys())[:5]}... ({len(ucdavis_lookup)} total keys)")

                    sanitized_derived_id = sanitize_filename_string(derived_id_for_component)

                    try:
                        # Load original image
                        original_pil_img = Image.open(image_path).convert("RGB")

                        # Preprocess for 11-channels (matching training pipeline)
                        img_pil_preprocessed_rot, _ = rotate_to_wide(original_pil_img.copy())
                        img_pil_padded_for_model, paste_offset, scaled_dims, scale_factor_applied = \
                            rescale_and_pad_image(img_pil_preprocessed_rot, TARGET_SIZE)

                        eleven_channel_data_np = create_11channel_input(
                            img_pil_padded_for_model, SATO_SIGMAS, MEIJERING_SIGMAS, FRANGI_SIGMAS, HESSIAN_SIGMAS, ENHANCE_PERCENTILE
                        )

                        input_tensor = torch.from_numpy(eleven_channel_data_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

                        # Perform prediction
                        with torch.no_grad():
                            seg_output_logits, _ = model(input_tensor) # Discard geo_output

                        # Process Segmentation Output: Apply softmax and argmax
                        predicted_seg_mask_padded = torch.argmax(F.softmax(seg_output_logits, dim=1).squeeze(0), dim=0).cpu().numpy()

                        # --- Reverse padding and rotation to get mask in original image dimensions ---
                        # Create an empty canvas of the scaled size before rotation
                        unpadded_mask_canvas = np.zeros((scaled_dims[1], scaled_dims[0]), dtype=predicted_seg_mask_padded.dtype)

                        # Paste the relevant part of the padded mask onto this canvas
                        unpadded_mask_canvas = predicted_seg_mask_padded[
                            paste_offset[1] : paste_offset[1] + scaled_dims[1],
                            paste_offset[0] : paste_offset[0] + scaled_dims[0]
                        ]

                        # Get original dimensions for final resize
                        original_unrotated_width, original_unrotated_height = original_pil_img.size

                        # Apply reverse rotation if rotation was applied during preprocessing
                        if original_unrotated_height > original_unrotated_width: # Original image was portrait, so it was rotated 270 deg (CCW)
                            # To reverse ROTATE_270 (CCW), we need to rotate 90 degrees CCW (k=1)
                            unpadded_mask_canvas = np.rot90(unpadded_mask_canvas, k=1)
                            # Target size for final resize is the original (width, height)
                            target_mask_size_for_resize = (original_unrotated_width, original_unrotated_height)
                        else:
                            # No rotation was applied, target size is just original (width, height)
                            target_mask_size_for_resize = (original_unrotated_width, original_unrotated_height)

                        # Resize the mask to the exact original image dimensions using nearest-neighbor interpolation
                        predicted_seg_mask_original_dims = np.array(
                            Image.fromarray(unpadded_mask_canvas.astype(np.uint8)).resize(
                                target_mask_size_for_resize, Image.NEAREST
                            )
                        )

                        # --- Connected Component Analysis ---
                        combined_leaf_mask_binary = ((predicted_seg_mask_original_dims == 1) | (predicted_seg_mask_original_dims == 2)).astype(np.uint8)
                        labeled_mask = label(combined_leaf_mask_binary, connectivity=2)
                        properties = regionprops(labeled_mask)

                        current_image_components_metadata = []
                        component_idx_in_original_image = 1

                        original_rgb_np = np.array(original_pil_img)

                        for region in properties:
                            min_row, min_col, max_row, max_col = region.bbox
                            bbox_height = max_row - min_row
                            bbox_width = max_col - min_col

                            if bbox_height < MIN_BBOX_DIMENSION or bbox_width < MIN_BBOX_DIMENSION:
                                continue # Skip this component if its bounding box is too small

                            # Generate unique name for the component (GLOBAL uniqueness now)
                            # Format: <DATASET_PREFIX><DerivedID>_<GlobalSerialID>
                            component_base_name = f"{dataset_prefix}{sanitized_derived_id}_{global_component_serial_id}"

                            # Determine output paths for the current category
                            component_masks_output_dir = Path(OUTPUT_PREDICTIONS_DIR) / category_name / "COMPONENT_MASKS"
                            component_rgb_crops_output_dir = Path(OUTPUT_PREDICTIONS_DIR) / category_name / "COMPONENT_RGB_CROPS"
                            component_overlays_output_dir = Path(OUTPUT_PREDICTIONS_DIR) / category_name / "COMPONENT_OVERLAYS"

                            # Crop Images and Masks for this component
                            cropped_rgb = original_rgb_np[min_row:max_row, min_col:max_col, :]

                            # Extract the component's specific mask within its bounding box
                            # First, isolate the region from the labeled_mask
                            component_binary_mask_in_bbox_full_size = (labeled_mask == region.label).astype(np.uint8)
                            component_binary_mask_in_bbox = component_binary_mask_in_bbox_full_size[min_row:max_row, min_col:max_col]

                            # Use this binary mask to filter the original segmentation predictions
                            # This ensures we only keep blade/vein pixels that belong to *this* component
                            cropped_seg_mask_full_classes = predicted_seg_mask_original_dims[min_row:max_row, min_col:max_col]
                            cropped_component_seg_mask_filtered = cropped_seg_mask_full_classes * component_binary_mask_in_bbox

                            # Count Pixels for Metadata
                            blade_pixels = np.sum(cropped_component_seg_mask_filtered == 1)
                            vein_pixels = np.sum(cropped_component_seg_mask_filtered == 2)

                            component_binary_only_leaf = ((cropped_component_seg_mask_filtered == 1) |
                                                        (cropped_component_seg_mask_filtered == 2)).astype(bool)
                            filled_component_binary = ndimage.binary_fill_holes(component_binary_only_leaf)
                            filled_leaf_area = np.sum(filled_component_binary)

                            background_pixels_internal_holes = filled_leaf_area - (blade_pixels + vein_pixels)

                            total_pixels_in_bbox = cropped_component_seg_mask_filtered.size

                            # Save Cropped Mask
                            mask_save_path = component_masks_output_dir / f"{component_base_name}_mask.png"
                            Image.fromarray(cropped_component_seg_mask_filtered.astype(np.uint8), mode='L').save(mask_save_path)

                            # Save Cropped RGB Image
                            rgb_crop_save_path = component_rgb_crops_output_dir / f"{component_base_name}_rgb_crop.png"
                            Image.fromarray(cropped_rgb).save(rgb_crop_save_path)

                            # Generate and Save Custom Overlay (using Matplotlib for consistency with previous WOLFSKILL script)
                            plt.figure(figsize=(cropped_rgb.shape[1]/100, cropped_rgb.shape[0]/100), dpi=100)

                            # Start with a black background for the overlay image
                            overlay_output_image = np.zeros_like(cropped_rgb, dtype=np.float32)

                            blade_mask = (cropped_component_seg_mask_filtered == 1)
                            vein_mask = (cropped_component_seg_mask_filtered == 2)

                            # Apply blade pixels with a desired transparency
                            # Only pixels identified as blade should retain their original RGB color, blended.
                            overlay_output_image[blade_mask] = cropped_rgb[blade_mask].astype(np.float32) / 255.0 * BLADE_OVERLAY_ALPHA

                            # Apply vein pixels (colored by colormap)
                            if np.any(vein_mask):
                                # Extract original RGB values for veins
                                vein_original_rgb_pixels = cropped_rgb[vein_mask]
                                # Convert to grayscale intensity (0-1) to map to colormap
                                vein_grayscale_intensity = color.rgb2gray(vein_original_rgb_pixels / 255.0)
                                # Get RGB colors from colormap based on intensity
                                colored_vein_pixels_rgb = vein_cmap(vein_grayscale_intensity)[:, :3] # Take only RGB

                                # Blend vein color over what's already there (blade or black)
                                # If VEIN_ALPHA is 1.0, this completely replaces the underlying pixels in the vein area.
                                overlay_output_image[vein_mask] = (overlay_output_image[vein_mask] * (1 - VEIN_ALPHA)) + (colored_vein_pixels_rgb * VEIN_ALPHA)

                            plt.imshow(overlay_output_image)
                            plt.axis('off')
                            plt.tight_layout(pad=0)

                            overlay_save_path = component_overlays_output_dir / f"{component_base_name}_overlay.png"
                            # Save with facecolor='black' to ensure the padding area outside the tight_layout is black,
                            # though overlay_output_image should already handle the component's internal background.
                            plt.savefig(overlay_save_path, bbox_inches='tight', pad_inches=0, facecolor='black', transparent=False)
                            plt.close()

                            # Append Metadata for Current Component
                            current_image_components_metadata.append({
                                "original_image_path": str_image_path, # Full original path
                                "dataset_category": category_name,
                                "derived_id": derived_id_for_component, # The raw derived ID, not sanitized
                                "component_name": component_base_name,
                                "component_idx_in_original_image": component_idx_in_original_image,
                                "blade_pixels": blade_pixels,
                                "vein_pixels": vein_pixels,
                                "background_pixels_internal_holes": background_pixels_internal_holes,
                                "total_bbox_pixels": total_pixels_in_bbox,
                                "bbox_min_row": min_row,
                                "bbox_min_col": min_col,
                                "bbox_max_row": max_row,
                                "bbox_max_col": max_col,
                                "mask_file": str(mask_save_path.name),
                                "rgb_crop_file": str(rgb_crop_save_path.name),
                                "overlay_file": str(overlay_save_path.name)
                            })

                            global_component_serial_id += 1
                            component_idx_in_original_image += 1

                        # Append components from the current image to the overall metadata list
                        # and immediately save the updated CSV (for resumability)
                        if current_image_components_metadata:
                            all_component_metadata.extend(current_image_components_metadata)
                            metadata_df = pd.DataFrame(all_component_metadata)
                            metadata_df.to_csv(METADATA_CSV_PATH, index=False)
                        pbar.update(1) # Increment overall progress bar

                    except Exception as e:
                        tqdm.write(f"❌ Error processing {image_path.name} (full path: {str_image_path}): {e}. Skipping this image.")
                        pbar.update(1) # Still count as processed for progress, even if failed.


                except StopIteration:
                    iterator_exhausted[i] = True # Mark this iterator as exhausted
                    # tqdm.write(f"{category_names_cycle[i]} images exhausted.", end='\r') # Optional: notify exhaustion

            if not processed_this_cycle and all(iterator_exhausted):
                break # All iterators exhausted, no images processed this cycle

    print(f"\n✅ Prediction and component extraction complete! Results saved to '{OUTPUT_PREDICTIONS_DIR}' folder.")

if __name__ == "__main__":
    # Ensure all necessary paths are configured correctly by the user.
    # The script will now ONLY run the main processing logic.
    print("Starting leaf component extraction process...")
    main()