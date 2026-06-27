import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import PrescriptionDataset
from model import CRNN
import time




import torch

def collate_fn(batch):

    images = []
    targets = []
    target_lengths = []

    for image, target in batch:

        images.append(image)

        targets.append(target)

        target_lengths.append(len(target))

    images = torch.stack(images)

    targets = torch.cat(targets)

    target_lengths = torch.IntTensor(target_lengths)

    return images, targets, target_lengths






train_df = pd.read_csv('data/RxHandBD-ML/Train_Label.csv')
# print(train_df.head())
# print(train_df.shape)
# print(train_df.columns)
# print(train_df.isnull().sum())
print("Samples:", len(train_df))

train_df["length"] = train_df["Text"].str.len()

# print("Max length:", train_df["length"].max())
# print("Average length:", train_df["length"].mean())



all_text = ''.join(train_df["Text"].astype(str))
chars = sorted(set(all_text))

# plt.text(0.5, 0.5, chars, fontsize=12, ha='center')
# print("Number of chars:", len(chars))



# # # # # # # # # # # # # # # # # 
# Transform from Char to Number #
#  الحاج موسى , موسى الحاج    #
# # # # # # # # # # # # # # # # # 

char_to_idx = {}
# idx_to_char = {}

for idx, char in enumerate(chars):
    text = char + " " + str(idx + 1) + '\n'
    # print(text)
    char_to_idx[char] = idx + 1



# for idx ,char in enumerate(chars):
#     text = str(idx + 1) + " " + char + "\n"
#     # print(text)
#     idx_to_char[idx + 1] = char











transform = transforms.Compose([
    transforms.Resize((128, 512)),
    transforms.ToTensor()
])


dataset = PrescriptionDataset(
    train_df,
    "data/RxHandBD-ML/Train_Set",
    char_to_idx,
    transform
)




loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn

)


print("Classes:", len(char_to_idx) + 1)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using:", device)
model = CRNN(num_classes=len(char_to_idx)+1).to(device)




def train():

    criterion = nn.CTCLoss(blank=0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )
    model.train()

    epochs = 10

    start = time.time()

    best_loss = float("inf")

    for epoch in range(epochs):

        total_loss = 0

        for images, targets, target_lengths in loader:

            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            output = model(images)

            output = output.log_softmax(2)

            output = output.permute(1, 0, 2)

            input_lengths = torch.full(
                (images.size(0),),
                output.size(0),
                dtype=torch.long,
                device=device
            )

            loss = criterion(
                output,
                targets,
                input_lengths,
                target_lengths
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)


        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("checkpoints", exist_ok=True)

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )

        print(f"Epoch {epoch+1} Loss={avg_loss:.4f}")

    



    end = time.time()

    print("Time:", end-start)






# if __name__ == "__main__":
#     train()