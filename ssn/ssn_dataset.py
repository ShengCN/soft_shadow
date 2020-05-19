import sys
sys.path.append("..")

import os
from os.path import join
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import random
import matplotlib.pyplot as plt
import cv2
from params import params
from .random_pattern import random_pattern

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
    def __init__(self, ds_dir, is_training):
        start = time.time()
        
        # # of samples in each group
        # magic number here
        parameter = params().get_params()
        # self.meta_data = pd.read_csv(csv_meta_file, header=None).to_numpy()
        self.meta_data = self.init_meta(ds_dir)
        self.meta_data = self.meta_data
        self.is_training = is_training
        self.to_tensor = ToTensor()
        self.mask_transfrom = Mask_Transform()
        self.ibl_transform = IBL_Transform()
        self.flip = parameter.flip

        end = time.time()
        print("Dataset initialize spent: {} ms".format(end - start))

        # fake random
        np.random.seed(19950220)
        np.random.shuffle(self.meta_data)
        self.training_num = len(self.meta_data) - len(self.meta_data) // 10
        print('training: {}, validation: {}'.format(self.training_num, self.training_num//10))
        
        if parameter.small_ds:
            self.training_num = self.training_num//20
        
        self.random_pattern_generator = random_pattern()
        
        self.thread_id = os.getpid()
        self.need_log = parameter.log
        self.seed = os.getpid()

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
        
        is_log = False
        if idx % 10 == 0 and self.thread_id == os.getpid() and self.need_log:
            is_log = True
        
        cur_seed = idx * 1234 + os.getpid() + time.time()
        random.seed(cur_seed)
        # random ibls
        if is_log:
            s = time.time()
        shadow_path, mask_path = self.meta_data[idx]  
        mask_img = plt.imread(mask_path)
        mask_img = mask_img[:,:,0]
        if mask_img.dtype == np.uint8:
            mask_img = mask_img/ 255.0

        if self.flip:
            mask_img, shadow_bases = np.expand_dims(mask_img, axis=2), 1.0 - np.load(shadow_path)
        else:
            mask_img, shadow_bases = np.expand_dims(mask_img, axis=2), np.load(shadow_path)
        
        if is_log:
            elapsed = time.time() - s
            log_info = '{} loading file time: {} \n'.format(idx, elapsed)

        if is_log:
            s = time.time()  
        shadow_img, light_img = self.render_new_shadow(shadow_bases, cur_seed)
        if is_log:
            elapsed = time.time() - s
            log_info += '{} rendering file time: {} \n'.format(idx, elapsed)
            self.log(log_info)
            
#         print('mask: {}, shadow: {}, light: {}'.format(mask_img.shape, shadow_img.shape, light_img.shape))
        mask_img, shadow_img, light_img = self.to_tensor(mask_img), self.to_tensor(shadow_img),self.to_tensor(light_img)
        
        return mask_img, light_img, shadow_img
    
    def init_meta(self, ds_dir):
        base_folder = join(ds_dir, 'base')
        mask_folder = join(ds_dir, 'cache/mask')
        model_list = [f for f in os.listdir(base_folder) if os.path.isdir(join(base_folder, f))]
        metadata = []
        for m in model_list:
            shadow_folder, cur_mask_folder = join(base_folder, m), join(mask_folder, m)
            shadows = [f for f in os.listdir(shadow_folder) if f.find('_shadow.npy')!=-1]
            for s in shadows:
                prefix = s[:s.find('_shadow')]
                metadata.append((join(shadow_folder, s), join(cur_mask_folder, prefix + '_mask.png')))
        
        return metadata

    def get_prefix(self, path):
        folder = os.path.dirname(path)
        basename = os.path.basename(path)
        return os.path.join(folder, basename[:basename.find('_')])
    
    def statistics(self, key):
        self.stats_keys[key] += 1

    def get_statistics(self):
        return self.stats_keys
    
    def render_new_shadow(self, shadow_bases, seed):
        h, w, iw, ih = shadow_bases.shape
        # is_bias = random.random() < 0.5
        # if is_bias:
        #     low, high = 0, 6
        # else:
        #     low, high = 0, 50
        num = random.randint(0, 50)
        pattern_img = self.random_pattern_generator.get_pattern(num=num, size=0.1, mitsuba=False, seed=int(seed))
        
        # flip to mitsuba ibl
        pattern_img = self.normalize_energy(cv2.flip(cv2.resize(pattern_img, (iw, ih)), 0))
        shadow = np.tensordot(shadow_bases, pattern_img, axes=([2,3], [1,0]))
        pattern_img = np.expand_dims(cv2.resize(pattern_img, (32,16)), 2)

        return np.expand_dims(shadow, 2), pattern_img
    
    def get_min_max(self, batch_data, name):
        print('{} min: {}, max: {}'.format(name, np.min(batch_data), np.max(batch_data)))

    def log(self, log_info):
        with open('log.txt', 'a+') as f:
            f.write(log_info)

    def normalize_energy(self, ibl, energy=30.0):
        if np.sum(ibl) < 1e-3:
            return ibl
        return ibl * energy / np.sum(ibl)
