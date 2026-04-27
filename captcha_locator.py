import cv2
import numpy as np

def locate_captcha(image):
    """
    Spectrum Radar: Locates the captcha by finding the rectangular region with the 
    highest color variance (ignoring grayscale UI elements).
    """
    # 1. Measure color 'saturation' or variance at each pixel
    # Simple way: max(R,G,B) - min(R,G,B)
    b, g, r = cv2.split(image)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    diff = max_rgb - min_rgb
    
    # 2. Threshold the 'colorfulness'
    _, color_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    
    # 3. Find Rectangles in the original grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    H, W = image.shape[:2]
    candidates = []
    
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        # Geometrical filter (Standard Captcha Rectangles)
        if 2.5 <= aspect_ratio <= 7.0 and 100 <= w <= 500 and 20 <= h <= 120:
            # Check for colorfulness inside this box
            roi_mask = color_mask[y:y+h, x:x+w]
            color_score = np.sum(roi_mask > 0) / (w * h)
            
            # Captchas MUST have circles (color)
            if color_score > 0.02: # At least 2% colorful pixels
                candidates.append((x, y, w, h, color_score))
    
    if not candidates:
        return None
        
    # Pick the one with the most colorfulness (DNA of the circles)
    candidates.sort(key=lambda x: x[4], reverse=True)
    return candidates[0][:4]

if __name__ == "__main__":
    img = cv2.imread('image/resim3.png')
    if img is not None:
        print(f"Captcha Found at: {locate_captcha(img)}")
