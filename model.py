import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.lstm = nn.LSTM(
            # 64 channels × 32 height = 2048
            input_size=64 * 32,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        x = self.cnn(x)

        batch_size, channels, height, width = x.size()

        x = x.permute(0, 3, 1, 2)

        x = x.contiguous().view(
            batch_size,
            width,
            channels * height
        )

        x, _ = self.lstm(x)

        x = self.fc(x)

        return x