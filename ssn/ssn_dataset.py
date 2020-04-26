import sys
sys.path.append("..")

import os
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image
import time
import random
from skimage.transform import resize
import matplotlib.pyplot as plt
from params import params
import numbergen as ng
import imagen as ig

class To_Normalized_Img(object):
    """Convert PIL image to [0,1] numpy"""

    def __call__(self, img):
        img = np.array(img)
        if img.dtype == np.uint8:
            img = img/255.0
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, is_transpose=True):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if is_transpose:
            img = img.transpose((2, 0, 1))
        return torch.Tensor(img)

class IBL_Transform(object):
    """ IBL transforms before training"""

    # input is PIL img, output is numpy
    def __call__(self, img):
        normalize_transform = To_Normalized_Img()
        img = normalize_transform(img)
        
        return img[:,:,1:2]

class Mask_Transform(object):
    """ Mask transforms before training
        Input: PIL image
        Output: numpy
    """
    def __call__(self, img):
        normalize_transform = To_Normalized_Img()
        img = normalize_transform(img)
        h,w,c = img.shape
        return img[:,:,0:1]
    
class SSN_Dataset(Dataset):
    def __init__(self, csv_meta_file, is_training):
        start = time.time()
        
        # # of samples in each group
        # magic number here
        self.ibl_group_size = 16
        
        parameter = params().get_params()
        self.meta_data = pd.read_csv(csv_meta_file, header=None).to_numpy()

        self.is_training = is_training
        self.to_tensor = ToTensor()
        self.mask_transfrom = Mask_Transform()
        self.ibl_transform = IBL_Transform()
        
        end = time.time()
        print("Dataset initialize spent: {} ms".format(end - start))

        # fake random
        np.random.seed(19950220)
        np.random.shuffle(self.meta_data)
        self.training_num = (len(self.meta_data) - int(len(self.meta_data) / 10))
        
        if parameter.small_ds:
            self.training_num = self.training_num//30
        
        self.ibl_num = parameter.ibl_num
        self.scale_ibl = parameter.scale_ibl
        self.ibl_shape = [16, 32, 1]
        self.shadow_shape = [256, 256, 1]
        self.ibl_pattern_generator = ig.Composite(operator=np.add,
                                                  generators=[ig.Gaussian(size=0.15,
                                                                          x=ng.UniformRandom(seed=i+1)-0.5,
                                                                          y=ng.UniformRandom(seed=i+2)-0.5,
                                                                          orientation=np.pi*ng.UniformRandom(seed=i+3))
                                                                for i in range(10)])
    
    def __len__(self):
        if self.is_training:
            return self.training_num
        else:
            # return len(self.meta_data) - self.training_num
            return self.training_num//10

        # exp_one_data
        # return 1

    def __getitem__(self, idx):
        if self.is_training and idx > self.training_num:
            print("error")
        # offset to validation set
        if not self.is_training:
            idx = self.training_num + idx
        
        # random ibls
        mask_path, shadow_path = self.meta_data[idx]
        mask_img, shadow_bases = np.expand_dims(np.load(mask_path),2), np.load(shadow_path)
        shadow_img, light_img = self.render_new_shadow(shadow_bases)
        del shadow_bases
        # print('mask: {}, shadow: {}, light: {}'.format(mask_img.shape, shadow_img.shape, light_img.shape))
        mask_img, shadow_img, light_img = self.to_tensor(mask_img), self.to_tensor(shadow_img),self.to_tensor(light_img)
        
        return mask_img, light_img, shadow_img
    
    def get_prefix(self, path):
        folder = os.path.dirname(path)
        basename = os.path.basename(path)
        return os.path.join(folder, basename[:basename.find('_')])
    
    def check_light(self, light_img):
        return np.max(light_img) != 0.0
    
    def statistics(self, key):
        self.stats_keys[key] += 1

    def get_statistics(self):
        return self.stats_keys
    
    def render_new_shadow(self, shadow_bases):
        h, w, iw, ih = shadow_bases.shape
        pattern_img = self.ibl_pattern_generator()
        pattern_img = resize(pattern_img, (ih, iw))
        shadow = np.tensordot(shadow_bases, pattern_img, axes=([2,3], [1,0]))
        pattern_img = np.expand_dims(resize(pattern_img, (16,32)), 2)

        return np.expand_dims(shadow, 2), pattern_img
    
    def get_min_max(self, batch_data, name):
        print('{} min: {}, max: {}'.format(name, np.min(batch_data), np.max(batch_data)))