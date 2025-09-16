# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CHARS = len(CHARS)
MAX_LEN = 6

class BreakerCNN(nn.Module):
    def __init__(self, num_chars=NUM_CHARS, max_len=MAX_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # feature map size for input 1x60x160 -> after 3 pools: roughly 1x7x20 => flatten
        self.fc = nn.Linear(128*7*20, 512)
        # one classifier per time-step (fixed max_len)
        self.classifiers = nn.ModuleList([nn.Linear(512, num_chars+1) for _ in range(max_len)])  # +1 for blank/pad

    def forward(self, x):
        # x: B x 1 x 60 x 160
        f = self.conv(x)
        f = f.view(f.size(0), -1)
        f = F.relu(self.fc(f))
        outs = [clf(f) for clf in self.classifiers]  # list of B x (num_chars+1)
        out = torch.stack(outs, dim=1)  # B x max_len x (num_chars+1)
        return out

class UsabilityMLP(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128,32), nn.ReLU(),
            nn.Linear(32,1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
