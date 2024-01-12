import os
import cv2
import numpy as np
from HSVEnhancer import HSVEnhancer
from MSER import MSER
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
    
    
class DataSaver:
    def saveImg(self, img, destPath, filename):
        if not os.path.exists(destPath):
            os.makedirs(destPath)
        filename = filename.split('.')[0] + '.png'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave(os.path.join(destPath, filename), img)
        

def main(args):
    datasaver = DataSaver()
    mser = MSER()
    hsvEnhancer = HSVEnhancer()
    parts = ['train', 'test']
    for part in parts:
        srcPath = os.path.join(args.srcRoot, part)
        destPath = os.path.join(args.destRoot, part)
        for filename in tqdm(os.listdir(srcPath)):
            imgpath = os.path.join(srcPath, filename)
            img = cv2.imread(imgpath)
            original = img.copy()
            if args.method == 'HSV':
                img = hsvEnhancer.enhance(img)
            elif args.method == 'MSER':
                img = mser.preprocessing(img)
            else:
                img = hsvEnhancer.enhance(img)
                img = mser.preprocessing(img)
            img = (img * 0.8 + original * 0.2).astype(np.uint8) 
            datasaver.saveImg(img, destPath, filename)
        

if __name__ == '__main__':
    print('ing')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srcRoot', type=str, required=True)
    parser.add_argument('-d', '--destRoot', type=str, required=True)
    parser.add_argument('-m', '--method', type=str, default='MSER', choices=['MSER', 'HSV', 'MIX'], required=True)
    args = parser.parse_args()
    main(args)