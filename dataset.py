from __future__ import print_function
import torch.utils.data as data
from PIL import Image
from skimage import io
import os
import os.path
import numpy as np


class Digits(data.Dataset):
    def __init__(self, list, transform=None):

        f_l = open(list, 'r')
        self.list = f_l.readlines()
        if list.find('train') != -1:
            self.dtype = 'train'
        else:
            self.dtype = 'test'
        self.transform = transform


    def __getitem__(self, index):
        image_name = self.list[index]
        image_name = image_name[0:-1]

        if self.dtype == 'train':
            l_image = io.imread(image_name)
            image = io.imread(image_name)
            if len(image.shape) < 3:
                l_image = np.repeat(l_image[:, :, np.newaxis], 3, 2)
                image = np.repeat(image[:, :, np.newaxis], 3, 2)
            image = self.transform(Image.fromarray(image, mode='RGB'))
            l_image = self.transform(Image.fromarray(l_image, mode='RGB'))
            return image, l_image
        else:
            image = io.imread(image_name)
            if len(image.shape) < 3:
                image = np.repeat(image[:, :, np.newaxis], 3, 2)
            image = self.transform(Image.fromarray(image, mode='RGB'))
            return image, image_name

    def __len__(self):
        return len(self.list)
