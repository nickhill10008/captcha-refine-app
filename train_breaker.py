# train_breaker.py
import torch, os, random
from torch.utils.data import DataLoader
from dataset_utils import CaptchaDataset
from models import BreakerCNN, CHARS
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    # pad/truncate to MAX_LEN
    MAX_LEN = 6
    padded = []
    for l in labels:
        arr = l[:MAX_LEN] + [len(CHARS)]*(MAX_LEN - len(l))  # index len(CHARS) as PAD
        padded.append(arr)
    labels = torch.tensor(padded, dtype=torch.long)
    return imgs, labels

def train():
    ds = CaptchaDataset(transform=None)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreakerCNN().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=len(CHARS))
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(dl):
            imgs = imgs.mean(dim=1, keepdim=True).to(device)  # grayscale
            labels = labels.to(device)
            out = model(imgs)  # B x max_len x C
            loss = sum(criterion(out[:,i,:], labels[:,i]) for i in range(out.size(1)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss {total_loss/len(dl)}")
        torch.save(model.state_dict(), "breaker.pth")

if __name__ == "__main__":
    train()
