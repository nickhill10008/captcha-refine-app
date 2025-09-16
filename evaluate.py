# evaluate.py
import torch
from dataset_utils import CaptchaDataset
from torch.utils.data import DataLoader
from models import BreakerCNN, CHARS
import numpy as np

def evaluate(model_path="breaker.pth", n=500):
    ds = CaptchaDataset()
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreakerCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    correct = 0
    total = 0
    for i,(img, lbl, raw) in enumerate(dl):
        if i>=n: break
        inp = img.mean(dim=1, keepdim=True).to(device)
        with torch.no_grad():
            out = model(inp)
        preds = out.argmax(dim=-1)[0].cpu().numpy()
        pred_text = ''.join(CHARS[p] for p in preds if p < len(CHARS))
        if pred_text.startswith(raw[0]):  # simple compare prefix
            correct += 1
        total += 1
    print(f"Acc (prefix match) {correct}/{total} = {correct/total:.3f}")

if __name__ == "__main__":
    evaluate()
