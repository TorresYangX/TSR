from cv2 import cvtColor, COLOR_BGR2HSV, COLOR_HSV2BGR
import numpy as np

# cv2 BGR standard

def log2_sat(x):
    return np.log2(x + 1)

def sqrt_sat(x):
    return np.sqrt(x)

def quad_comp(x):
    return x * x

class HSVEnhancer:
    def __init__(self, saturate=sqrt_sat, compress=quad_comp):
        self.saturate = saturate
        self.compress = compress

    def enhance(self, img):
        rb = img.copy()
        g = img.copy()
        rb[..., 1] = 0
        g[..., [0, 2]] = 0
        hsv_rb = cvtColor(rb, COLOR_BGR2HSV)
        hsv_g = cvtColor(g, COLOR_BGR2HSV)
        hsv_rb[..., 2] = self.saturate(hsv_rb[..., 2] / 255) * 255
        hsv_g[..., 2] = self.compress(hsv_g[..., 2] / 255) * 255
        result = cvtColor(hsv_rb, COLOR_HSV2BGR) + cvtColor(hsv_g, COLOR_HSV2BGR)
        return result
