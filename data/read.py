'''
    --Get image names of train and test having unique gleason rating
'''

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

class ImageProcess:

    def __init__(self, mask_path):
        self.mask_path = mask_path

    def img_names(self):
        '''
        --returns list of img_names from path specified
        '''
        return os.listdir(self.mask_path)
    
    def load_image(self, img_name):
        '''
        --loads an image as a numpy array

        --params
            -img_name > filename of the image

        --returns
            -np array
        '''
        img_path = os.path.join(self.mask_path, img_name)
        img = Image.open(img_path)
        img_array = np.asarray(img)
        return img_array

    def is_unique(self, img_name):
        '''
        --checks of an image has more than 2 unique pixel values (apart from white). Used to filter out images with multiple gleason ratings

        --params
            img_name > filename of the image

        --output
            if unique > label, True
            else > -1, False
        '''
        img_array = self.load_image(img_name)
        unique = np.unique(img_array)
        label = [item for item in unique if item!=4]
        if len(unique)>=2:
            return -1, False
        return label[0], True

    def unique_imgs(self):
        '''
        --generator object that returns image filename and corresponding label of images with unique gleason rating

        --params
            -none

        --returns
            -filename, label
        '''
        img_names = self.img_names()
        for name in img_names:
            label, unique = self.is_unique(name)
            name = name[5:-4]
            if unique:
                yield name, label



if __name__ == '__main__':
    PATH = 'C:/Users/Dhruvil/Desktop/Data_Sets/Harvard_Original/Gleason_masks_train'
    img_process = ImageProcess(PATH)
    unique_files = list(img_process.unique_imgs())
    print(len(unique_files))
    names = [name for name,_ in unique_files]
    labels = [label for _, label in unique_files]
    data_dict = {
        'name':names,
        'label':labels
    }
    df = pd.DataFrame(data_dict)
    df.to_csv('C:/Users/Dhruvil/Desktop/Data_Sets/Harvard_Original/unique.csv')
