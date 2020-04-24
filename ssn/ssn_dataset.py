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
        
        # random group ids
        self.ibl_group_id_list = [i for i in range(8)]
        
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
        self.first_init()
        
        self.training_num = (len(self.meta_data) - int(len(self.meta_data) / 10))
        
        if parameter.small_ds:
            self.training_num = self.training_num//30
        
        
        self.ibl_num = parameter.ibl_num
        self.scale_ibl = parameter.scale_ibl
        self.ibl_shape = [16, 32, 1]
        self.shadow_shape = [256, 256, 1]

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
        seed = idx * 1234 + os.getpid() + time.time()
        random.seed(seed)
        
        # random_ibl_num = random.randint(1,self.ibl_num)
        
        key = self.get_key(self.meta_data[idx])
        # random_lists = random.choices(self.mappings[key],k=random_ibl_num)
        group_lists = random.sample(self.ibl_group_id_list, k=3)
        random_lists = []
        for g in group_lists:
            sample_group = min(len(self.mappings[key][g]), self.ibl_group_size)
            random_lists += random.sample(self.mappings[key][g], k=sample_group)
        
        random_ibl_num = len(random_lists)
        shadows, lights = np.zeros(([random_ibl_num] + self.shadow_shape)), np.zeros(([random_ibl_num] + self.ibl_shape))
    
        for i in range(random_ibl_num):
            lights[i], shadows[i] = self.get_data(random_lists[i])
            
        light_img, shadow_img = self.render_new_shadow(lights, shadows)
        
        mask_img,_,_ = self.get_data(random_lists[0], True)
        mask_img, shadow_img, light_img = self.to_tensor(mask_img), self.to_tensor(shadow_img),self.to_tensor(light_img)
        
        return mask_img, light_img, shadow_img, random_ibl_num
    
    def get_prefix(self, path):
        folder = os.path.dirname(path)
        basename = os.path.basename(path)
        return os.path.join(folder, basename[:basename.find('_')])
    
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
    
    def first_init(self):
        """ Initialize a hash map: obj_type -> data list"""
        
        tmp_list = []
        # move those non-human out of current dataset
        for r in self.meta_data:
            key = os.path.basename(r[0]) 
            if key.find("simulated")==-1:
                continue
            tmp_list.append(r)
        
        self.meta_data = tmp_list
        self.mappings = dict()
        for r in self.meta_data:
            key = self.get_key(r)
            if not key in self.mappings.keys():
                self.mappings[key] = {group_id:[] for group_id in range(len(self.ibl_group_id_list))}
                
            group_num = r[8]
            self.mappings[key][group_num].append(r)
    
    def render_new_shadow(self, ibls, shadows):
        shadow_num = shadows.shape[0]
        
        if self.scale_ibl:
            scale_factor = np.random.rand(shadow_num)
        else:
            scale_factor = np.ones(shadow_num)
            
        new_ibl = np.tensordot(ibls, scale_factor, ([0],[0]))
        new_shadow = np.tensordot(shadows, scale_factor, ([0],[0]))

        return new_ibl, new_shadow
    
    def get_min_max(self, batch_data, name):
        print('{} min: {}, max: {}'.format(name, np.min(batch_data), np.max(batch_data)))
        
    def get_key(self, r):
        # model, rotation, camera position
        key = (os.path.basename(r[0]), r[5],r[4]) 
        return key
    
    def get_data(self, metadata_row, is_mask=False):
        mask_path, light_path, shadow_path = metadata_row[1], metadata_row[7], metadata_row[2]
        prefix = self.get_prefix(metadata_row[1])
        mask_path, light_path, shadow_path = prefix + '_mask.npy', prefix + '_light.npy', prefix + '_shadow.npy'
        # convert image to [0.0, 1.0] numpy 
        if is_mask:
            # mask_img = self.mask_transfrom(Image.open(mask_path))
            # light_img = self.ibl_transform(Image.open(light_path))
            # shadow_img = self.mask_transfrom(Image.open(shadow_path))
            mask_img = np.expand_dims(np.load(mask_path),2)/255.0
            shadow_img = np.expand_dims(np.load(shadow_path),2)/255.0
            light_img = np.expand_dims(np.load(light_path),2)
            return mask_img, light_img, 1.0-shadow_img
        else:
            # light_img = self.ibl_transform(Image.open(light_path))
            shadow_img = np.expand_dims(np.load(shadow_path),2)/255.0
            light_img = np.expand_dims(np.load(light_path),2)
            return light_img, 1.0-shadow_img