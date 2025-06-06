import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm

# ============== Configuration ==============
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "unet_final.pt"
INPUT_ROOT = "."
MASKS_OUT_ROOT = "OUTPUTS_MASKS"
RGB_OUT_ROOT = "OUTPUTS_RGB"
os.makedirs(MASKS_OUT_ROOT, exist_ok=True)
os.makedirs(RGB_OUT_ROOT, exist_ok=True)

# ============== U-Net Model ==============
class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        def CBR(in_ch, out_ch):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True)
            )

        self.enc1 = torch.nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.enc2 = torch.nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = torch.nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.pool = torch.nn.MaxPool2d(2)
        self.up2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = torch.nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = torch.nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.out_conv = torch.nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

# ============== Dataset for Inference ==============
class InferenceLeafDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.rgb_root = os.path.join(root_dir, 'images_rgb')
        self.image_paths = []
        for root, _, files in os.walk(self.rgb_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    rel_path = os.path.relpath(os.path.join(root, file), self.rgb_root)
                    self.image_paths.append(rel_path)

    def _center_crop_even(self, img):
        w, h = img.size
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return TF.crop(img, top, left, new_h, new_w)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]

        def load_img(folder):
            path = os.path.join(self.root_dir, folder, rel_path)
            img = Image.open(path).convert('L' if folder != 'images_rgb' else 'RGB')
            img = self._center_crop_even(img)
            return TF.to_tensor(img), img

        rgb_tensor, rgb_pil = load_img('images_rgb')
        sato, _ = load_img('images_sato')
        meijering, _ = load_img('images_meijering')
        frangi, _ = load_img('images_frangi')
        hessian, _ = load_img('images_hessian')

        image_tensor = torch.cat([rgb_tensor, sato, meijering, frangi, hessian], dim=0)
        return image_tensor, rgb_pil, rel_path

# ============== Utility for Overlay ==============
def apply_overlay(rgb_img, mask):
    rgb_np = np.array(rgb_img).copy()
    overlay = np.zeros_like(rgb_np)

    # Assign magenta (vein=2) and orange (blade=1)
    overlay[mask == 1] = [255, 165, 0]   # Blade: Orange
    overlay[mask == 2] = [255, 0, 255]   # Vein: Magenta

    blended = (0.6 * rgb_np + 0.4 * overlay).astype(np.uint8)
    return Image.fromarray(blended)

# ============== Run Inference ==============
def run_inference():
    dataset = InferenceLeafDataset(INPUT_ROOT)
    model = UNet(in_channels=7, out_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        for image_tensor, rgb_pil, rel_path in tqdm(dataset, desc="Predicting"):
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Create full paths
            mask_out_path = os.path.join(MASKS_OUT_ROOT, rel_path)
            rgb_out_path = os.path.join(RGB_OUT_ROOT, rel_path)
            os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
            os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)

            # Save raw mask
            Image.fromarray(pred_mask).save(mask_out_path)

            # Save overlay
            overlay_img = apply_overlay(rgb_pil, pred_mask)
            overlay_img.save(rgb_out_path)

    print(f"\n✅ Done! Results saved to '{MASKS_OUT_ROOT}' and '{RGB_OUT_ROOT}'")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    run_inference()
