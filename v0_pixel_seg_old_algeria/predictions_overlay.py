import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ============== CONFIG ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
MODEL_PATH = "unet_final.pt"
DATA_ROOT = "."  # Assumes current folder contains images_* and masks
SAVE_DIR = "visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Dataset Definition ==========
class MultiChannelLeafDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_names = sorted(os.listdir(os.path.join(root_dir, 'images_rgb')))
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        def load_img(folder):
            path = os.path.join(self.root_dir, folder, name)
            img = Image.open(path).convert('L' if folder != 'images_rgb' else 'RGB')
            img = self.transform(img)
            return img

        rgb = load_img('images_rgb')
        sato = load_img('images_sato')
        meijering = load_img('images_meijering')
        frangi = load_img('images_frangi')
        hessian = load_img('images_hessian')

        image = torch.cat([rgb, sato, meijering, frangi, hessian], dim=0)

        mask_path = os.path.join(self.root_dir, 'masks', name)
        mask = Image.open(mask_path)
        mask = TF.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask, name

# ============= Model Definition =============
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
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

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

# =========== Visualization Function ===========
def visualize(model, dataset, num_samples=10):
    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            image, mask, name = dataset[i]
            image_input = image.unsqueeze(0).to(DEVICE)
            output = model(image_input)
            pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
            mask = mask.numpy()

            # Convert first 3 channels to RGB image for display
            rgb_image = image[:3].permute(1, 2, 0).numpy()

            # Plot and save
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(rgb_image)
            axs[0].set_title("RGB")
            axs[1].imshow(mask, cmap="tab10", vmin=0, vmax=2)
            axs[1].set_title("Ground Truth")
            axs[2].imshow(pred, cmap="tab10", vmin=0, vmax=2)
            axs[2].set_title("Prediction")

            for ax in axs:
                ax.axis("off")

            plt.tight_layout()
            out_path = os.path.join(SAVE_DIR, f"{os.path.splitext(name)[0]}_pred.png")
            plt.savefig(out_path)
            plt.close()

    print(f"✅ Visualizations saved to '{SAVE_DIR}/'.")

# =================== MAIN ====================
if __name__ == "__main__":
    print("🚀 Loading model and dataset...")
    dataset = MultiChannelLeafDataset(DATA_ROOT)
    model = UNet(in_channels=7, out_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    visualize(model, dataset, num_samples=25)

