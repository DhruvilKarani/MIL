'''
    --Preprocess and make patches
'''

import sys
import os
import numpy as np 
from PIL import Image
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Patch:

    def __init__(self, path, patch_size, img_names, labels, transform=None):
        self.path = path
        self.patch_size = patch_size
        self.img_names = img_names
        self.labels = labels
        self.transform = transform

    @staticmethod
    def patches(img_array, patch_size):
        '''
        --returns patches of shape patch_size*patch_size for an image array

        --params:
            -img_array > np array of shape h,w,c (height, width, channel), Ideally h should be equal to w and c=3
            -patch_size > integer (divisible by h)
        
        --returns
            -torch tensor of shape (num_patches, c, patch_size, patch_size)
             where num_patches = (h/patch_size) **2
        '''
        h, w, c = img_array.shape
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patches.append(img_array[i:i+patch_size,j:j+patch_size,:])
        patches = torch.Tensor(patches).view(-1, c, patch_size, patch_size)
        return patches    

    def load_image(self, img_name):
        '''
        --loads image from path and img_name

        --params:
            -img_name > image filename
        '''
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        return img

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        --generator object to generate patches and corresponding labels

        --params:
            -img_names > list of filenames of images
            -labels > corresponding labels (to img_names)
            -shuffle > shuffle img_name, label pair. default = True

        --returns
            -(patches (according to 'patches' method), label)
        '''
        name, label = self.img_names[idx], self.labels[idx]
        img = self.load_image(name+'.jpg')
        
        if self.transform:
            return self.transform(img), label
        
        return img, label
    

class Patchify(object):

    def __init__(self, patch_dim):
        self.patch_dim = patch_dim

    def __call__(self, img):
        assert len(img.shape) == 3, "Check img shape"
        c, h, w = img.shape
        patches = []
        for i in range(0, h, self.patch_dim):
            for j in range(0, w, self.patch_dim):
                patches.append(torch.unsqueeze(img[:, i:i+self.patch_dim,j:j+self.patch_dim],0))
        patches = torch.cat(patches, 0)
        assert patches.shape[1:] == (c, self.patch_dim, self.patch_dim), "{0}".format(patches.shape)
        return patches 


if __name__ == '__main__':
    PATH = '/home/chamelyon/Documents/Gautam/harvard_original_processed/all/'
    df = pd.read_csv('/home/chamelyon/Documents/Gautam/harvard_original_processed/unique.csv')
    PATCH_SIZE = 620

    transform_scheme = transforms.Compose([
            transforms.ToTensor(),
            Patchify(PATCH_SIZE)
        ])

    names = list(df['name'])
    labels= list(df['label'])
    PatchDataset = Patch(PATH, PATCH_SIZE, names, labels, transform = transform_scheme)

    PatchLoader = DataLoader(PatchDataset, batch_size=2, shuffle=True, num_workers=8)

    for i, (whole_img, label) in enumerate(PatchLoader):
        print("Main", whole_img.shape, label)