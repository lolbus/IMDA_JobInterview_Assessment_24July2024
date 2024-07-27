# Simple CNN architecture used to develop a character analyzer
import torch.nn as nn
import torch
from torchvision import transforms

class SimpleCNNArchitecture(nn.Module):
    def __init__(self):
        super(SimpleCNNArchitecture, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 36)  # 26 letters + 10 digits
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # x = self.dropout(x)
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

mean, std = 0.6701, 0.4443
torch_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
