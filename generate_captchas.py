# generate_captchas.py
import random, os, string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

OUT_DIR = "data/captchas"
FONT_PATHS = []  # leave empty to use default PIL font; add ttf paths for variety

os.makedirs(OUT_DIR, exist_ok=True)

def random_text(length=5):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def add_noise(draw, w, h, amount=0.05):
    for _ in range(int(w*h*amount)):
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)
        draw.point((x,y), fill=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))

def distort(img):
    # simple affine warp
    arr = np.array(img)
    h, w = arr.shape[:2]
    dx = w * 0.03
    dy = h * 0.03
    x1 = int(random.uniform(-dx, dx))
    y1 = int(random.uniform(-dy, dy))
    return img.transform(img.size, Image.AFFINE, (1, 0.2 * random.uniform(-1,1), x1, 0.2 * random.uniform(-1,1), 1, y1), resample=Image.BILINEAR)

def generate_one(save_path, text=None, width=160, height=60):
    if text is None:
        text = random_text(5)
    # background
    bg_color = tuple(random.randint(200,255) for _ in range(3))
    img = Image.new('RGB', (width,height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # text font
    font_size = int(height * 0.6)
    if FONT_PATHS:
        font = ImageFont.truetype(random.choice(FONT_PATHS), font_size)
    else:
        font = ImageFont.load_default()

    # draw some lines
    for _ in range(random.randint(1,3)):
        x1,y1 = random.randint(0,width), random.randint(0,height)
        x2,y2 = random.randint(0,width), random.randint(0,height)
        draw.line((x1,y1,x2,y2), fill=(random.randint(50,150),)*3, width=random.randint(1,3))

    # render text with jitter
    spacing = width // (len(text)+1)
    for i,ch in enumerate(text):
        x = spacing*(i+1) + random.randint(-8,8)
        y = int((height-font_size)/2) + random.randint(-6,6)
        draw.text((x,y), ch, font=font, fill=(random.randint(0,120),random.randint(0,120),random.randint(0,120)))

    # noise and blur
    add_noise(draw, width, height, amount=0.02)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,1.2)))
    img = distort(img)

    img.save(save_path)
    return text

if __name__ == "__main__":
    # generate a dataset
    N = 2000
    labels_file = os.path.join(OUT_DIR, "labels.txt")
    with open(labels_file, "w") as f:
        for i in range(N):
            text = random_text(5)
            fname = f"cap_{i:05d}.png"
            generate_one(os.path.join(OUT_DIR, fname), text=text)
            f.write(f"{fname} {text}\n")
    print("Generated", N, "captchas in", OUT_DIR)
