import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def generate_number_dataset(count=50000):
    output_dir = "dataset/numbers/train"
    os.makedirs(output_dir, exist_ok=True)
    labels = []
    
    fonts = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\verdana.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf",
        "C:\\Windows\\Fonts\\times.ttf"
    ]
    
    print(f"Master Veri Seti Üretimi Başladı: {count} Örnek...")
    
    for i in range(count):
        # Generate number (0-99)
        num = random.randint(0, 99)
        
        # 1. Canvas (Match ROI size approximately)
        width, height = 120, 50
        img = Image.new('RGB', (width, height), color=(240, 243, 248))
        draw = ImageDraw.Draw(img)
        
        # 2. Draw Number
        font_path = random.choice(fonts)
        font_size = random.randint(28, 34)
        font = ImageFont.truetype(font_path, font_size)
        
        text = str(num)
        t_w, t_h = draw.textbbox((0, 0), text, font=font)[2:]
        draw.text(((width - t_w)//2 + random.randint(-5,5), (height - t_h)//2 + random.randint(-3,3)), 
                  text, font=font, fill=(0, 0, 0))
        
        # 3. EXTREME NOISE (Circles)
        for _ in range(random.randint(4, 7)):
            cx, cy = random.randint(0, width), random.randint(0, height)
            r = random.randint(10, 25)
            color = (random.randint(50,200), random.randint(50,200), random.randint(50,200))
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=random.randint(1,2))
            
        # 4. Convert to Binary (Simulate the Solver's Denoising)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological Jitter
        kernel = np.ones((2,2), np.uint8)
        if random.random() > 0.5:
            thresh = cv2.dilate(thresh, kernel, iterations=1)
        if random.random() > 0.5:
            thresh = cv2.erode(thresh, kernel, iterations=1)
            
        binary = cv2.bitwise_not(thresh)
        
        # Save
        fname = f"train_{i}.png"
        cv2.imwrite(os.path.join(output_dir, fname), binary)
        
        tens = num // 10 if num >= 10 else 10
        units = num % 10
        labels.append({'filename': fname, 'tens': tens, 'units': units})
        
        if i % 10000 == 0:
            print(f"Üretilen: {i}...")

    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
    print("Üretim Tamamlandı!")

if __name__ == "__main__":
    generate_number_dataset(100000)
