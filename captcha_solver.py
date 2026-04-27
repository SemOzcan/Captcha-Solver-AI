import os
import sys
import hashlib
import cv2
import numpy as np

# 1. Dizin Sabitleme: RPA'nın dışarıdan çalıştırması durumunda yolların şaşmaması için
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)

from predict_model_plus import PlusAnchorSolver
from inject_real_data import EQUATIONS

# 2. Beyin ve Hafıza Hazırlığı (Warm-up)
_ai_brain = PlusAnchorSolver(model_path="number_classifier.pth")
ALTIN_HAFIZA = {}

def get_image_hash(image_path):
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def build_gold_memory():
    """Hafızadaki Altın Örneklerin Hash Kayıtlarını oluşturur."""
    for fname, (n1, n2) in EQUATIONS.items():
        path = os.path.join('image', fname)
        if os.path.exists(path):
            h = get_image_hash(path)
            if h:
                ALTIN_HAFIZA[h] = (n1 + n2, f"{n1} + {n2}")

# İlk yüklemede hafızayı doldur
build_gold_memory()

def solve_math_captcha(image_path, verbose=True):
    try:
        # A. HAFIZA KONTROLÜ (Anında Çözüm)
        img_hash = get_image_hash(image_path)
        if img_hash and img_hash in ALTIN_HAFIZA:
            res, eq = ALTIN_HAFIZA[img_hash]
            if verbose: print(f" -> [HAFIZA] Kesin Eşleşme: {eq} = {res}")
            return res
        
        # B. AI ANALİZİ (Görsel yeniyse)
        if verbose: print(f" -> [AI] Analiz ediliyor...")
        res, eq = _ai_brain.solve(image_path)
        return res

    except Exception as e:
        if verbose: print(f"Hata: {e}")
        return "ERROR"

if __name__ == "__main__":
    # C# (Nodesy AI) Tarafından Çağrılma Mantığı
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        # ÖNEMLİ: C# sadece bu print'i okuyacak, verbose False olmalı
        print(solve_math_captcha(target_path, verbose=False))
    else:
        # Manuel Geliştirici Testi
        test_path = os.path.join("image", "yeni1.png")
        print(f"Test Başlatıldı: {test_path}")
        result = solve_math_captcha(test_path, verbose=True)
        print(f"SONUC: {result}")