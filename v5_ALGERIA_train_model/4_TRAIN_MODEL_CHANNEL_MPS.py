import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm

# ===================== CONFIG =====================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============== Dataset Definition ===============
class MultiChannelLeafDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_names = sorted(os.listdir(os.path.join(root_dir, 'images_rgb')))

    def __len__(self):
        return len(self.image_names)

    def _center_crop_even(self, img):
        """Crop image so height and width are even (divisible by 2)"""
        w, h = img.size
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return TF.crop(img, top, left, new_h, new_w)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        def load_img(folder):
            path = os.path.join(self.root_dir, folder, name)
            img = Image.open(path).convert('L' if folder != 'images_rgb' else 'RGB')
            img = self._center_crop_even(img)
            return TF.to_tensor(img)

        rgb = load_img('images_rgb')
        sato = load_img('images_sato')
        meijering = load_img('images_meijering')
        frangi = load_img('images_frangi')
        hessian = load_img('images_hessian')

        image = torch.cat([rgb, sato, meijering, frangi, hessian], dim=0)

        mask_path = os.path.join(self.root_dir, 'MASKS', name)
        mask = Image.open(mask_path)
        mask = self._center_crop_even(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# ===================== U-Net =====================
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

        # Add output_padding=1 for mismatch fix
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2, output_padding=0)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2, output_padding=0)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.out_conv = nn.Conv2d(64, out_channels, 1)

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

# ===================== Training =====================
def train():
    dataset = MultiChannelLeafDataset(".")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = UNet(in_channels=7, out_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"unet_epoch{epoch}.pt"))

    torch.save(model.state_dict(), "unet_final.pt")
    print("✅ Training complete. Final model saved as unet_final.pt")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()


