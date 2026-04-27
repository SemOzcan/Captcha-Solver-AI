import os
import cv2
import pandas as pd
import numpy as np
import re

# Ground Truth Equations from the user screenshots
EQUATIONS = {
    'islem.jpg': (26, 5),
    'Ekran görüntüsü 2026-04-13 174619.png': (36, 6),
    'Ekran görüntüsü 2026-04-13 174816.png': (23, 6),
    'Ekran görüntüsü 2026-04-13 174834.png': (29, 7),
    'Ekran görüntüsü 2026-04-13 174850.png': (66, 3),
    'Ekran görüntüsü 2026-04-13 174855.png': (94, 4),
    'Ekran görüntüsü 2026-04-13 174901.png': (65, 6),
    'Ekran görüntüsü 2026-04-13 174906.png': (89, 3),
    'Ekran görüntüsü 2026-04-13 174910.png': (55, 8),
    'Ekran görüntüsü 2026-04-13 174915.png': (50, 2),
    'Ekran görüntüsü 2026-04-13 174921.png': (10, 8),
    'Ekran görüntüsü 2026-04-13 174928.png': (67, 7),
    'Ekran görüntüsü 2026-04-13 174934.png': (91, 8),
    'Ekran görüntüsü 2026-04-13 174939.png': (76, 6),
    'Ekran görüntüsü 2026-04-13 174943.png': (19, 7),
    'Ekran görüntüsü 2026-04-21 171128.png': (19, 3),
    'Ekran görüntüsü 2026-04-21 171314.png': (69, 4),
    'Ekran görüntüsü 2026-04-21 171319.png': (63, 0),
    'Ekran görüntüsü 2026-04-21 171325.png': (26, 0),
    'Ekran görüntüsü 2026-04-21 171330.png': (11, 0),
    'Ekran görüntüsü 2026-04-21 171334.png': (11, 0),
    'Ekran görüntüsü 2026-04-21 171338.png': (40, 2),
    'Ekran görüntüsü 2026-04-21 171345.png': (81, 1),
    'Ekran görüntüsü 2026-04-22 093534.png': (81, 1),
    'Ekran görüntüsü 2026-04-22 093545.png': (98, 8),
    'Ekran görüntüsü 2026-04-22 093553.png': (23, 5),
    'resim1.png': (81, 1),
    'resim2.png': (40, 2),
    'resim3.png': (23, 5),
    'resim4.png': (98, 8),
    'resim5.png': (81, 1),
    'Ekran görüntüsü 2026-04-24 102917.png': (25, 6),
    'Ekran görüntüsü 2026-04-24 102925.png': (69, 1),
    'Ekran görüntüsü 2026-04-24 102932.png': (14, 4),
    'Ekran görüntüsü 2026-04-24 102945.png': (14, 4),
    'Ekran görüntüsü 2026-04-24 102955.png': (73, 7),
    'yeni1.png': (92, 7),
    'yeni2.png': (28, 3),
    'yeni3.png': (12, 4),
    'yeni4.png': (94, 7),
    'yeni5.png': (74, 1),
    'yeni6.png': (87, 4),
    'yeni7.png': (46, 7),
    'yeni8.png': (93, 3),
    'yeni9.png': (10, 3),
    'yeni10.png': (38, 7)
}

def inject():
    from captcha_locator import locate_captcha
    from plus_locator import find_plus_sign
    
    output_dir = "dataset/numbers/train"
    labels_path = os.path.join(output_dir, "labels.csv")
    
    # 1. Clean existing 'real_' entries and physical files
    if os.path.exists(labels_path):
        df = pd.read_csv(labels_path)
        df = df[~df['filename'].str.startswith('real_')]
    else:
        df = pd.DataFrame(columns=['filename', 'tens', 'units'])
        
    for f in os.listdir(output_dir):
        if f.startswith('real_'):
            try: os.remove(os.path.join(output_dir, f))
            except: pass

    new_rows = []
    print(f"Injecting {len(EQUATIONS)} Golden Samples with ASCII-Safe filenames...")
    
    img_idx = 0
    for img_name, (n1, n2) in EQUATIONS.items():
        path = os.path.join("image", img_name)
        if not os.path.exists(path): continue
        
        # Load image via numpy to avoid encoding issues with imread
        img_array = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Consistent ROI Pre-processing
        if image.shape[1] > 300:
            loc = locate_captcha(image)
            if loc:
                x, y, w, h = loc
                pad = 3
                image = image[y+pad:y+h-pad, x+pad:x+w-pad]
        
        px = find_plus_sign(image)
        l_crop = image[:, :px-2]
        r_crop = image[:, px+6:]
        
        img_idx += 1
        
        def process_and_save(crop, num, side, base_id):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            # Denoising
            refined = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
            binary = cv2.bitwise_not(refined)
            
            # Boost weight by 100x copies
            tens = num // 10 if num >= 10 else 10
            units = num % 10
            
            for copy_idx in range(100):
                # ASCII STRICT FILENAME
                fname = f"real_sam_{base_id}_{side}_{copy_idx}.png"
                cv2.imwrite(os.path.join(output_dir, fname), binary)
                new_rows.append({'filename': fname, 'tens': tens, 'units': units})

        process_and_save(l_crop, n1, "L", img_idx)
        process_and_save(r_crop, n2, "R", img_idx)

    final_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    final_df.to_csv(labels_path, index=False)
    print(f"Master injection complete! Total rows: {len(final_df)}")

if __name__ == "__main__":
    inject()
