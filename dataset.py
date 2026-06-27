import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PrescriptionDataset(Dataset):

    def __init__(self, df, img_dir, char_to_idx, transform=None):

        self.df = df
        self.img_dir = img_dir
        self.char_to_idx = char_to_idx

        # ======================
        # DEFAULT TRANSFORM (if none provided)
        # ======================
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 512)),
                transforms.Grayscale(num_output_channels=1),

                # 🔥 Data Augmentation
                transforms.RandomRotation(2),
                transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)),
                transforms.ColorJitter(0.2, 0.2, 0.2),

                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # ======================
        # Load image + label
        # ======================
        img_name = self.df.iloc[idx, 0]
        text = self.df.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("L")

        # ======================
        # Apply transform
        # ======================
        if self.transform:
            image = self.transform(image)

        # ======================
        # Encode text → numbers
        # ======================
        target = torch.IntTensor(
            [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        )

        return image, target