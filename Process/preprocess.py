import os
import cv2
from HSVEnhancer import HSVEnhancer
from MSER import MSER
import matplotlib.pyplot as plt
import argparse


class Dataloader:   
    def loadImgs(self, path):
        imgFilenames = []
        for filename in os.listdir(path):
            imgFilenames.append(filename)
        return imgFilenames
    
    
class DataSaver:
    def saveImg(self, img, destPath, filename):
        if not os.path.exists(destPath):
            os.makedirs(destPath)
        filename = filename.split('.')[0] + '.png'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave(os.path.join(destPath, filename), img)
        

def main(args):
    dataloader = Dataloader()
    datasaver = DataSaver()
    mser = MSER()
    hsvEnhancer = HSVEnhancer()
    parts = ['train', 'test']
    for part in parts:
        srcPath = os.path.join(args.srcRoot, part)
        destPath = os.path.join(args.destRoot, part)
        imgFilenames = dataloader.loadImgs(srcPath)
        for filename in imgFilenames:
            imgpath = os.path.join(srcPath, filename)
            img = cv2.imread(imgpath)
            if args.method == 'HSV':
                img = hsvEnhancer.enhance(img)
                datasaver.saveImg(img, destPath, filename)
            elif args.method == 'MSER':
                img = mser.preprocessing(img)
                datasaver.saveImg(img, destPath, filename)
            else:
                img = hsvEnhancer.enhance(img)
                img = mser.preprocessing(img)
                datasaver.saveImg(img, destPath, filename)
        
        

if __name__ == '__main__':
    print('ing')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srcRoot', type=str, required=True)
    parser.add_argument('-d', '--destRoot', type=str, required=True)
    parser.add_argument('-m', '--method', type=str, default='MSER', choices=['MSER', 'HSV', 'MIX'], required=True)
    args = parser.parse_args()
    main(args)