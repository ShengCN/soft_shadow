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
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
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

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = img.transpose((2, 0, 1))
        return torch.Tensor(image)

class IBL_Transform(object):
    """ IBL transforms before training"""

    # input is PIL img, output is numpy
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
        self.first_init()
        
        self.training_num = (len(self.meta_data) - int(len(self.meta_data) / 10))
        
        parameter = params().get_params()
        self.ibl_num = parameter.ibl_num

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
        
        random_range = (0, self.__len__())
        # offset to validation set
        if not self.is_training:
            idx = self.training_num + idx
            random_range = (self.training_num , self.training_num + self.__len__())
            
        def get_data(metadata_row):
            mask_path, light_path, shadow_path = metadata_row[1], metadata_row[3], metadata_row[2]
            # convert image to [0.0, 1.0] numpy 
            mask_img = self.mask_transfrom(Image.open(mask_path))
            light_img = self.ibl_transform(Image.open(light_path))
            shadow_img = self.mask_transfrom(Image.open(shadow_path))
            return mask_img, light_img, shadow_img
        
        path_list = self.meta_data[idx]
        mask_img, light_img, shadow_img = get_data(path_list)
        shadow_list, light_list = [shadow_img], [light_img]
        
        # random ibls
        seed = idx * 1234 + os.getpid()
        random.seed(seed)
        # random_ibl_num = random.randint(0,2)
        
        # random_ibl_num = random.randint(0,self.ibl_num)
        random_ibl_num = self.ibl_num
        key = (os.path.basename(self.meta_data[idx][0]), self.meta_data[idx][-2])
        random_lists = random.choices(self.mappings[key],k=random_ibl_num)
        
        for new_data in random_lists:
            _,light,shadow = get_data(new_data)
            shadow_list.append(shadow)
            light_list.append(light)
        
        light_img, shadow_img = self.render_new_shadow(light_list, shadow_list)
        mask_img, shadow_img, light_img = self.to_tensor(mask_img), self.to_tensor(shadow_img),self.to_tensor(light_img)

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
            key = (os.path.basename(r[0]),r[-2]) 
            if key in self.mappings.keys():
                self.mappings[key].append(r)
            else:
                self.mappings[key] = []
                self.mappings[key].append(r)
        # import pdb; pdb.set_trace()
        
    def render_new_shadow(self, ibls, shadows):
        assert len(ibls) == len(shadows)
        
        ibl_num = len(ibls)
        if ibl_num == 1:
            return ibls[0], shadows[0]
        
        ibl_num = float(ibl_num)
        new_ibl = ibls[0]
        for i in range(1, int(ibl_num)):
            new_ibl += ibls[i]
        
        new_shadow = shadows[0]
        for i in range(1, int(ibl_num)):
            new_shadow += shadows[i]
        
        return new_ibl, new_shadow