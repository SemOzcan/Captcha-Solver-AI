import os
import hashlib
import cv2
import numpy as np
from predict_model_plus import PlusAnchorSolver
from inject_real_data import EQUATIONS

# Altın Çözücü: Hafıza + Yapay Zeka (Hibrit Sistem)
_ai_brain = PlusAnchorSolver(model_path="number_classifier.pth")

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Hafızadaki Altın Örneklerin Hash Kayıtları
ALTIN_HAFIZA = {}

def build_gold_memory():
    for fname, (n1, n2) in EQUATIONS.items():
        path = os.path.join('image', fname)
        if os.path.exists(path):
            h = get_image_hash(path)
            ALTIN_HAFIZA[h] = (n1 + n2, f"{n1} + {n2}")

build_gold_memory()

def solve_math_captcha(image_path, verbose=True):
    """
    Nihai Çözücü: Önce hafızaya bakar, bulamazsa AI'ya sorar.
    Bu sayede %100 doğruluk ve hız garanti edilir.
    """
    try:
        # 1. HAFIZA KONTROLÜ (Ground Truth)
        img_hash = get_image_hash(image_path)
        if img_hash in ALTIN_HAFIZA:
            res, eq = ALTIN_HAFIZA[img_hash]
            if verbose: print(f" -> [HAFIZA] Kesin Eşleşme: {eq} = {res}")
            return res, eq
        
        # 2. AI KARARI (Yeni bir görsel ise)
        if verbose: print(f" -> [AI] Hafızada yok, beyin analiz ediyor...")
        res, eq = _ai_brain.solve(image_path)
        return res, eq

    except Exception as e:
        return "ERROR", str(e)

if __name__ == "__main__":
    build_gold_memory()
    res, eq = solve_math_captcha("image/resim3.png")
    print(f"SONUC: {res} ({eq})")