
# refine_loop.py
import os, random
from generate_captchas import generate_one, random_text
from models import BreakerCNN, CHARS
import torch
from PIL import Image, ImageFilter   # <-- FIX HERE
import numpy as np
import cv2


DATA_DIR = "data/captchas"
LABELS = os.path.join(DATA_DIR, "labels.txt")

def load_breaker(path="breaker.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreakerCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

def predict_text_from_output(out):
    # out: 1 x max_len x (C+1) logits
    idxs = out.argmax(dim=-1)[0].cpu().numpy().tolist()
    res = ''.join(CHARS[i] if i < len(CHARS) else '' for i in idxs)
    return res

def adversary_confidence(out):
    # average softmax confidence for predicted chars
    probs = torch.softmax(out, dim=-1)
    top_probs, top_idx = probs.max(dim=-1)
    return top_probs.mean().item()

def simulate_usability(img):  # quick heuristic: less noisy images -> easier
    arr = np.array(img.convert("L")).astype(np.float32)
    # normalized edge magnitude as proxy for clutter
    import cv2
    edges = cv2.Laplacian(arr, cv2.CV_32F)
    score = np.mean(np.abs(edges))
    # map to 0-1 inverse (higher edges = less usable)
    val = 1.0 - (score / 50.0)
    return float(max(0.0, min(1.0, val)))

def refine_loop(iterations=200, batch=10):
    model, device = load_breaker()
    # parameters controlling generator aggressiveness
    noise_level = 0.02
    blur = 0.6
    success_thresh = 0.2  # adversary conf threshold below this considered safe

    for it in range(iterations):
        adv_succ = 0
        usability_acc = 0
        for b in range(batch):
            text = random_text(5)
            fname = f"temp_{it}_{b}.png"
            # temporarily call generate_one with randomization influenced by noise_level, blur
            # For simplicity, adjust generate_one internals by temporarily changing module-level params
            # We'll re-generate using generate_one and then post-process
            path = os.path.join("data/captchas", fname)
            generate_one(path, text=text)
            img = Image.open(path)
            # apply extra noise/blur proportional to parameters
            if random.random() < noise_level:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur * 2))
            # predict
            inp = torch.tensor(np.array(img.convert("L"))/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            out = model(inp)
            conf = adversary_confidence(out)
            pred = predict_text_from_output(out)
            # simulate human usability
            us = simulate_usability(img)
            usability_acc += us
            if conf > success_thresh:
                adv_succ += 1
        adv_rate = adv_succ / batch
        avg_us = usability_acc / batch
        print(f"Iter {it}: adv_rate={adv_rate:.2f}, avg_us={avg_us:.2f}, noise={noise_level:.3f}, blur={blur:.2f}")
        # refine rules
        if adv_rate > 0.2:
            noise_level = min(0.5, noise_level + 0.02)
            blur = min(3.0, blur + 0.1)
        elif avg_us < 0.5:
            noise_level = max(0.0, noise_level - 0.02)
            blur = max(0.0, blur - 0.1)
        # cleanup temporary files
        for b in range(batch):
            try:
                os.remove(os.path.join("data/captchas", f"temp_{it}_{b}.png"))
            except:
                pass

if __name__ == "__main__":
    refine_loop(100, batch=8)
