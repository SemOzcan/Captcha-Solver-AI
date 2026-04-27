import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from cnn_model_plus import NumberNet
from captcha_locator import locate_captcha

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlusAnchorSolver:
    def __init__(self, model_path="number_classifier.pth"):
        self.model = NumberNet().to(DEVICE)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((40, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def solve(self, image_path):
        try:
            img_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None: return None, "Gorsel Okunamadi"
            
            h_orig, w_orig = image.shape[:2]
            
            # STEP 1: Always use the Spectrum Radar to find the Captcha Box
            loc = locate_captcha(image)
            if loc:
                x, y, w, h = loc
                # Crop with slight padding to ensure we don't cut digits
                roi = image[max(0,y-2):min(h_orig,y+h+2), max(0,x-2):min(w_orig,x+w+2)]
            else:
                # If radar fails but image is already small, use it as is
                if w_orig < 500: roi = image
                else: return None, "Kutu Bulunamadi"

            # STEP 2: Find the Plus Sign IN THE ROI ONLY
            from plus_locator import find_plus_sign
            px = find_plus_sign(roi)
            
            l_crop = roi[:, :px-2]
            r_crop = roi[:, px+6:]
            
            def predict_val(crop):
                if crop.size == 0: return 0
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # AI expects BLACK digits on WHITE background (consistent with elite set)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                final_bin = cv2.bitwise_not(thresh)
                
                pil_img = Image.fromarray(final_bin).convert('L')
                input_tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    ot, ou = self.model(input_tensor)
                    _, pt = torch.max(ot, 1)
                    _, pu = torch.max(ou, 1)
                    t_val = pt.item() if pt.item() < 10 else 0
                    u_val = pu.item()
                    return t_val * 10 + u_val

            n1 = predict_val(l_crop)
            n2 = predict_val(r_crop)
            return n1 + n2, f"{n1} + {n2}"
            
        except Exception as e:
            return None, str(e)
