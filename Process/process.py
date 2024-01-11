import os
from shutil import copy
import json


class DataLoader():
    def build_dir(self, path):
        path = path.rstrip('/')
        dirs = ['', '/images', '/labels', '/images/train', '/images/test', '/labels/train', '/labels/test']
        for dir in dirs:
            dir = path + dir
            if not os.path.exists(dir):
                os.makedirs(dir)
    
    def load(self, src, dest):
        pass


class tt100k_2021_Loader(DataLoader):
    def load(self, src, dest):
        src = src.rstrip('/')
        dest = dest.rstrip('/')
        
        super().build_dir(dest)
        
        annots = {}
        with open(f'{src}/annotations_all.json') as file:
            annots = json.load(file)

        types = {}
        for i, type in enumerate(annots['types']):
            types[type] = i

        def _load(path):
            path = path.rstrip('/')
            with open(f'{src}/{path}/ids.txt') as file:
                ids = file.read().split()
            
            for id in ids:
                if id not in annots['imgs']:
                    continue
                annot = annots['imgs'][id]
                objects = annot['objects']
                copy(f'{src}/{path}/{id}.jpg', f'{dest}/images/{path}')
                with open(f'{dest}/labels/{path}/{id}.txt', 'w') as file:
                    for object in objects:
                        xmin, ymin, xmax, ymax = map(int, object['bbox'].values())
                        category = types[object['category']]
                        file.write(f'{category} {(xmin + xmax) / 2 / 2048} {(ymin + ymax) / 2 / 2048} {(xmax - xmin) / 2048} {(ymax - ymin) / 2048}\n')

        _load('train')
        _load('test')


dataloader = {
    'tt100k_2021': tt100k_2021_Loader,
}
dataset_name = 'tt100k_2021'

if __name__ == '__main__':
    print(f'Loading {dataset_name}')
    loader = dataloader[dataset_name]()
    loader.load('../../../dataset/tt100k_2021', './dataset')
    
