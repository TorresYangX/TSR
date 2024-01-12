from multiprocessing import  Process
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


def preprocess(srcPath, destPath, method, filenames):
    datasaver = DataSaver()
    mser = MSER()
    hsvEnhancer = HSVEnhancer()
    for filename in tqdm(filenames):
        img = cv2.imread(os.path.join(srcPath, filename))
        original = img.copy()
        if method == 'HSV':
            img = hsvEnhancer.enhance(img)
        elif method == 'MSER':
            img = mser.preprocessing(img)
        elif method == 'MIX':
            img = hsvEnhancer.enhance(img)
            img = mser.preprocessing(img)
        else:
            print('No such a method!')
            return
        img = (img * 0.8 + original * 0.2).astype(np.uint8) 
        datasaver.saveImg(img, destPath, filename)
    

if __name__ == '__main__':
    proc_num = 12
    processes = []
    srcRoot = '/mnt/e/dataset/original/images/train'
    destRoot = '/mnt/e/dataset/processed/mix/images/train'
    filenames = list(os.listdir(srcRoot))
    seg_size = len(filenames) // proc_num
    for i in range(proc_num):
        print(f'start process {i}')
        seg_filenames = filenames[i * seg_size: min((i + 1) * seg_size, len(filenames))]
        proc = Process(target=preprocess, args=(srcRoot, destRoot, 'MIX', seg_filenames))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    print('')