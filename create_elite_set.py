import os
import cv2
import numpy as np
import pandas as pd
from plus_locator import find_plus_sign
from inject_real_data import EQUATIONS

def create_elite_set():
    train_dir = 'dataset/numbers/train'
    os.makedirs(train_dir, exist_ok=True)
    # Clean prev files
    for f in os.listdir(train_dir):
        try: os.remove(os.path.join(train_dir, f))
        except: pass

    labels = []
    idx = 0
    for fname, (n1, n2) in EQUATIONS.items():
        path = os.path.join('image', fname)
        if not os.path.exists(path):
            print(f"Atlandiyor: {fname}")
            continue
        
        # Use imdecode for unicode paths
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: continue
        
        # Locate plus and crop
        px = find_plus_sign(img)
        crops = [(img[:, :px-2], n1), (img[:, px+6:], n2)]
        
        for crop_img, val in crops:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            for _ in range(300):
                h, w = thresh.shape
                # Random slight transformations
                M = np.float32([[1,0,np.random.randint(-1,2)],[0,1,np.random.randint(-1,2)]])
                aug = cv2.warpAffine(thresh, M, (w, h))
                
                out_name = f'elite_{idx}.png'
                cv2.imwrite(os.path.join(train_dir, out_name), cv2.bitwise_not(aug))
                labels.append({
                    'filename': out_name, 
                    'tens': val//10 if val>=10 else 10, 
                    'units': val%10
                })
                idx += 1
                
    pd.DataFrame(labels).to_csv(os.path.join(train_dir, 'labels.csv'), index=False)
    print(f"Elite Toplanti Bitti: {idx} Uzman Ornek.")

if __name__ == "__main__":
    create_elite_set()
