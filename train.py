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
from sklearn.model_selection import train_test_split
from EarlyStopping import EarlyStopping


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



# ======================
# Load dataset
# ======================
train_df = pd.read_csv('data/RxHandBD-ML/Train_Label.csv') 
print(train_df.head()) 
print(train_df.shape) 
print(train_df.columns) 
print(train_df.isnull().sum()) 
print("Samples:", len(train_df)) 

train_df["length"] = train_df["Text"].str.len() 

print("Max length:", train_df["length"].max()) 
print("Average length:", train_df["length"].mean()) 
print(train_df["Text"].str.len().describe()) 


# ======================
# Char mapping
# ======================
all_text = ''.join(train_df["Text"].astype(str)) 
chars = sorted(set(all_text)) 

char_to_idx = {} 

for idx, char in enumerate(chars): 
    char_to_idx[char] = idx + 1 



# ======================
# Train / Val split
# ======================
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)


# ======================
# Transforms
# ======================
train_transform = transforms.Compose([ 
    transforms.Resize((128, 512)), 
    transforms.Grayscale(num_output_channels=1), 

    transforms.RandomRotation(2), 
    transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)), 
    transforms.ColorJitter(0.2, 0.2, 0.2), 

    transforms.ToTensor() 
]) 

test_transform = transforms.Compose([ 
    transforms.Resize((128, 512)), 
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor() 
]) 


# ======================
# Datasets
# ======================
train_dataset = PrescriptionDataset(
    train_df,
    "data/RxHandBD-ML/Train_Set",
    char_to_idx,
    train_transform
)

val_dataset = PrescriptionDataset(
    val_df,
    "data/RxHandBD-ML/Train_Set",
    char_to_idx,
    test_transform
)


# ======================
# Loaders
# ======================
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


# ======================
# Model
# ======================
print("Classes:", len(char_to_idx) + 1)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)

model = CRNN(num_classes=len(char_to_idx)+1).to(device)


# ======================
# Train function
# ======================
def train():

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    epochs = 100
    start = time.time()

    os.makedirs("checkpoints", exist_ok=True)
    early_stopping = EarlyStopping(patience=5, verbose=True, path='checkpoints/best_model.pth')

    for epoch in range(epochs):

        # ======================
        # TRAINING PHASE
        # ======================
        model.train()
        total_loss = 0

        for images, targets, target_lengths in train_loader:

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ======================
        # VALIDATION PHASE
        # ======================
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for images, targets, target_lengths in val_loader:

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

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)

        # ======================
        # Print & Early Stopping
        # ======================
        print(f"Epoch {epoch+1}")
        print(f"Train Loss = {avg_train_loss:.4f}")
        print(f"Val Loss   = {avg_val_loss:.4f}")
        
        early_stopping(avg_val_loss, model)
    
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered. Training stopped!")
            break

    print("✅ Done! Your best model is saved inside 'checkpoints/' folder")

    end = time.time()
    print("Time:", end-start)

# ======================
# Run training
# ======================
# train()