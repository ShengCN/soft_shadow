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
        self.first_init()
        
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
        random_ibl_num = random.randint(0,10)
        random_lists = random.choices(self.meta_data[random_range[0]:random_range[1]],k=random_ibl_num)
        
        for new_data in random_lists:
            _,light,shadow = get_data(new_data)
            shadow_list.append(shadow)
            light_list.append(light)
        
        light_img, shadow_img = self.render_new_shadow(light_list, shadow_list)
        mask_img, shadow_img, light_img = self.to_tensor(mask_img), self.to_tensor(shadow_img),torch.clamp(self.to_tensor(1.0 - light_img),0.0,1.0)

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
        key = self.meta_data[0][0]
        tmp = []
        for row in self.meta_data:
            if row[0] == key:
                tmp.append(row)
            else:
                break
        self.meta_data = tmp
    
    def render_new_shadow(self, ibls, shadows):
        assert len(ibls) == len(shadows)
        
        ibl_num = len(ibls)
        if ibl_num == 1:
            return ibls[0], shadows[0]
        
        ibl_num = float(ibl_num)
        new_ibl = ibls[0]/ibl_num
        for i in range(1, int(ibl_num)):
            new_ibl += ibls[i]/ibl_num
        
        new_shadow = shadows[0]/ibl_num
        for i in range(1, int(ibl_num)):
            new_shadow += shadows[i]/ibl_num
        
        return new_ibl, new_shadow
            