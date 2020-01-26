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

        self.first_init()
        end = time.time()
        print("Dataset initialize spent: {} ms".format(end - start))

        # fake random
        np.random.seed(19950220)
        np.random.shuffle(self.keys)
        self.training_num = len(self.keys) - int(len(self.keys) / 10)

        # statistics
        self.stats_keys = {k: 0 for k in self.keys}

    def __len__(self):
        if self.is_training:
            return self.training_num
        else:
            return len(self.keys) - self.training_num

        # exp_one_data
        # return 1

    def __getitem__(self, idx):
        """ randomly select two pairs """

        if self.is_training and idx > self.training_num:
            print("error")

        # offset to validation set
        if not self.is_training:
            idx = self.training_num + idx

        # exp_one_data
        # idx = 0
        self.statistics(self.keys[idx])

        # random select one pair
        data_list = self.hash[self.keys[idx]]
        mask_path, shadow_path, light_path = random.choice(data_list)
        
        counter = 0
        # in case the same as idx
        while True:
            counter += 1
            nov_mask_path, nov_shadow_path, nov_light_path = random.choice(data_list)
            if counter > 100:
                print('please check meta data file')
            # print('old random: {} new random: {}'.format(mask_path,nov_mask_path))
            if nov_mask_path != mask_path:
                break

        # convert image to [0.0, 1.0]
        mask_img = self.mask_transfrom(Image.open(mask_path))
        shadow_img = 1.0 - self.mask_transfrom(Image.open(shadow_path))
        light_img = self.ibl_transform(Image.open(light_path))

        nov_mask_img = self.mask_transfrom(Image.open(nov_mask_path))
        nov_shadow_img = 1.0 - self.mask_transfrom(Image.open(nov_shadow_path))
        nov_light_img = self.ibl_transform(Image.open(nov_light_path))

        mask_img, nov_mask_img = self.to_tensor(mask_img), self.to_tensor(nov_mask_img)
        light_img, nov_light_img = torch.clamp(self.to_tensor(light_img),0.0,1.0),  torch.clamp(self.to_tensor(nov_light_img),0.0,1.0)
        shadow_img, nov_shadow_img = self.to_tensor(shadow_img), self.to_tensor(nov_shadow_img)

        return mask_img, light_img, shadow_img, nov_mask_img, nov_light_img, nov_shadow_img

    def first_init(self):
        """ construct hash map """
        """ key(camera pos, human rotation) -> list(prefix) """

        self.hash = dict()
        self.keys = set()

        for data in self.meta_data:
            filename = os.path.basename(data[1])
            prefix = self.get_prefix(filename)
            camera_position = data[4]
            human_rot = data[5]
            
            file_type = data[0]
            # key = (file_type,camera_position, human_rot)
            key = (file_type, camera_position, human_rot)

            # initialize the dict value
            if key not in self.hash.keys():
                self.hash[key] = []

            mask_path = data[1]
            shadow_path = data[2]
            light_path = data[3]
            self.hash[key].append((mask_path, shadow_path, light_path))
            self.keys.add(key)

        self.keys = list(self.keys)
        

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