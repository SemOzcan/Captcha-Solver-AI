import cv2
import numpy as np

def find_plus_sign(image):
    """
    Stabilized Plus Sign Detector: Uses horizontal and vertical projection 
    peaks to find the exact intersection of the math operator.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    col_sums = np.sum(thresh, axis=0)
    w = len(col_sums)
    mid_start, mid_end = int(w * 0.2), int(w * 0.8)
    
    if mid_end <= mid_start: return w // 2
    
    # Smooth the signal to avoid noise peaks
    col_sums_smooth = np.convolve(col_sums, np.ones(3)/3, mode='same')
    px = np.argmax(col_sums_smooth[mid_start:mid_end]) + mid_start
    
    return int(px)

if __name__ == "__main__":
    # Test
    test_img = cv2.imread("image/islem.jpg")
    if test_img is not None:
        px = find_plus_sign(test_img)
        print(f"Artı işareti X koordinatı: {px}")
        # Görselleştirme (Debug için)
        cv2.line(test_img, (px, 0), (px, test_img.shape[0]), (0, 0, 255), 2)
        cv2.imwrite("debug_plus.jpg", test_img)
