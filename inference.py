import torch
import json
from PIL import Image
from torchvision import transforms
from model import CRNN

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Load vocab
# ======================
with open("checkpoints/vocab.json", "r") as f:
    char_to_idx = json.load(f)

idx_to_char = {int(v): k for k, v in char_to_idx.items()}

blank = 0

# ======================
# Model
# ======================
model = CRNN(num_classes=len(char_to_idx) + 1)
model.load_state_dict(
    torch.load("checkpoints/best_model.pth", map_location=device)
)
model.to(device)
model.eval()

# ======================
# Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((128, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# ======================
# Image
# ======================
img_path = "data/RxHandBD-ML/Train_Set/P1116.jpg"

image = Image.open(img_path).convert("L")
image = transform(image).unsqueeze(0).to(device)

# ======================
# Inference
# ======================
with torch.no_grad():
    output = model(image)

# output shape = (1,128,69)
output = output.squeeze(0)
output = output.log_softmax(dim=1)

print("Output Shape:", output.shape)

# ======================
# Greedy CTC Decoder
# ======================
preds = torch.argmax(output, dim=1)

text = ""
prev = blank

for p in preds.tolist():

    if p != blank and p != prev:
        text += idx_to_char.get(p, "")

    prev = p

print("Predicted Text:", text)