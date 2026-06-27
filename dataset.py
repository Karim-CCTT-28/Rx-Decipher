import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class PrescriptionDataset(Dataset):
    
            #   train,"RxHandBD-ML/Train_Set",char_to_idx,transform
    def __init__(self, df, img_dir, char_to_idx, transform=None):

        self.df = df
        self.img_dir = img_dir
        self.char_to_idx = char_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_name = self.df.iloc[idx, 0]
        text = self.df.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # target = []

        # for c in text:
        #     target.append(self.char_to_idx[c]) 
        target = torch.IntTensor(
        [self.char_to_idx[c] for c in text])

        target = torch.IntTensor(target)

        return image, target