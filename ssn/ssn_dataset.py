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
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize

class To_Normalized_Img(object):
    """Convert PIL image to [0,1] numpy"""

    def __call__(self, img):
        img = np.array(img)/255.0
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = img.transpose((2, 0, 1))
        return torch.Tensor(image)

class IBL_Transform(object):
    """ IBL transforms before training"""

    # input is PIL img, out put is numpy
    def __call__(self, img):
        normalize_transform = To_Normalized_Img()
        img = normalize_transform(img)
        img = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3.0
        # print('1 min: {} max: {} shape: {}'.format(np.min(img),np.max(img), img.shape))
        img = gaussian_filter(img, sigma=20)
        # print('2 min: {} max: {} shape: {}'.format(np.min(img),np.max(img), img.shape))
        img = resize(img,(16,32))
        img = img/np.max(img)
        # print('3 min: {} max: {} shape: {}'.format(np.min(img),np.max(img), img.shape))
        return img.reshape(16,32,1)

class Mask_Transform(object):
    """ Mask transforms before training
        Input: PIL image
        Output: numpy
    """
    def __call__(self, img):
        normalize_transform = To_Normalized_Img()
        img = normalize_transform(img)
        h, w, c = img.shape
        img = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3.0
        img = resize(img, (256, 256, 1))
        ret = np.zeros(img.shape, dtype=np.float)
        ret[np.where(img > 0.9)] = 1.0
        return ret


class SSN_Dataset(Dataset):
    def __init__(self, csv_meta_file, is_training):
        start = time.time()

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
        self.training_num = len(self.meta_data) - int(len(self.meta_data) / 10)

    def __len__(self):
        if self.is_training:
            return self.training_num
        else:
            return len(self.meta_data) - self.training_num

        # exp_one_data
        # return 1

    def __getitem__(self, idx):
        if self.is_training and idx > self.training_num:
            print("error")

        # offset to validation set
        if not self.is_training:
            idx = self.training_num + idx
        
        path_list = self.meta_data[idx]
        
        mask_path, light_path, shadow_path = path_list[1], path_list[3], path_list[2]
        
        # convert image to [0.0, 1.0]
        mask_img = self.mask_transfrom(Image.open(mask_path))
        light_img = self.ibl_transform(Image.open(light_path))
        shadow_img = 1.0 - self.mask_transfrom(Image.open(shadow_path))

        mask_img, shadow_img, light_img = self.to_tensor(mask_img), self.to_tensor(shadow_img),torch.clamp(self.to_tensor(light_img),0.0,1.0)

        return mask_img, light_img, shadow_img
    
    def get_prefix(self, path):
        return path[0:path.find('_')]

    # def downsample_light(self, img):
    #
    #     img = gaussian_filter(img, sigma=20)
    #     img = resize(img,(16,32))
    #     h, w, c = img.shape
    #     return img[:,:,0].reshape(h,w,1)
    
    def check_light(self, light_img):
        return np.max(light_img) != 0.0
    
    def statistics(self, key):
        self.stats_keys[key] += 1

    def get_statistics(self):
        return self.stats_keys