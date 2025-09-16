# dataset_utils.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CaptchaDataset(Dataset):
    def __init__(self, data_dir="data/captchas", labels_file="data/captchas/labels.txt", transform=None, max_len=6):
        self.data_dir = data_dir
        with open(labels_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.items = [l.split() for l in lines]
        self.transform = transform or T.Compose([T.Grayscale(), T.Resize((60,160)), T.ToTensor()])
        self.chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char_to_idx = {c:i for i,c in enumerate(self.chars)}
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def encode_label(self, s):
        # simple fixed-length encoding, pad with index -1 (we'll ignore in loss)
        res = [self.char_to_idx[c] for c in s]
        return res

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        img = Image.open(os.path.join(self.data_dir, fname)).convert("RGB")
        img = self.transform(img)
        lbl = self.encode_label(label)
        return img, lbl, label
