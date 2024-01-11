import cv2
import numpy as np

class MSER:          
    def preprocessing(self, img):
        tot = np.sum(img, axis=2)
        gray = np.maximum(img[..., 0] / tot, img[..., 2] / tot)
        gray = (gray * 255).astype(np.uint8)
        
        mser = cv2.MSER_create(delta=5, min_area=5,min_diversity = 0, max_variation=0.5)
        _, boxes = mser.detectRegions(gray)
        height, width = img.shape[0] // 128, img.shape[1] // 128
        boxes[:, 0] -= width // 2
        boxes[:, 1] -= height // 2
        boxes[:, 2] += width
        boxes[:, 3] += height
            
        mask = np.zeros_like(img)
        for box in boxes:
            x, y, w, h = box
            start_point = (x, y)
            end_point = (x+w, y+h)
            points = np.array([[start_point, (start_point[0], end_point[1]), end_point, (end_point[0], start_point[1])]])
            cv2.fillPoly(mask, points, (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        MSER_img = cv2.bitwise_and(img, img, mask=mask)
        return MSER_img
    
    
    
        
        
        
        