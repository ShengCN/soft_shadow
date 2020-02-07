import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
from tqdm import tqdm
import numpy as np
import os
import math
from PIL import Image
from ssn.ssn_dataset import Mask_Transform, ToTensor, IBL_Transform
from ssn.ssn import Relight_SSN
from utils.net_utils import save_model, get_lr, set_lr
from utils.utils_file import create_folder
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from params import params as options, parse_params
import multiprocessing
from multiprocessing import set_start_method
import random

# params = parse_params()

params = options().get_params()
print(params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device: ", device)
model = Relight_SSN(1,1)
weight_file = os.path.join('weights', 'bilinear_groupnorm_05-February-12-30-PM.pt')
checkpoint = torch.load(weight_file, map_location=device)    
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

def to_one_batch(img_tensor):
    c,h,w = img_tensor.size()
    return img_tensor.view(1,c,h,w)

def get_trnsf():
    img_trnsf = transforms.Compose([
        Mask_Transform(),
        ToTensor()
    ])
    
    ibl_trnsf = transforms.Compose([
        IBL_Transform(),
        ToTensor()
    ])
    
    return img_trnsf, ibl_trnsf

def mask_to_rgb(mask_np):
    dimension = mask_np.shape
    w,h = dimension[0], dimension[1]
    rgb_np = np.zeros((w,h,3))
    rgb_np[:,:,0],rgb_np[:,:,1],rgb_np[:,:,2] = np.squeeze(mask_np),np.squeeze(mask_np),np.squeeze(mask_np)
    
    return rgb_np
    
def save_results(img_batch, out_path):
    # check if folder exist
    folder = os.path.dirname(out_path)
    create_folder(folder)

    batch, c, h, w = img_batch.size()
    for i in range(batch):
        cur_batch = img_batch[i]
        batch_np = cur_batch.detach().cpu().numpy().transpose((1,2,0))
        mask, shadow = batch_np[:,:,0], batch_np[:,:,1]
        # shadow[np.where(mask != 0)] = 0.0
        fig, axs = plt.subplots(1,2)
        for ax, cur_img, title in zip(axs, [mask, shadow], ['mask', 'shadow']):
            ax.imshow(cur_img, interpolation='nearest', cmap='gray')
            ax.set_title(title)
        plt.savefig(out_path)

def predict(img, ibl_img):
    """ Predict results for a numpy img(png image using alpha channel to represent mask) + ibl numpy img
        img: w x h x 3 image
        ibl_img: [0,1.0]
    """
    img_trnsf = transforms.Compose([
        Mask_Transform(),
        ToTensor()
    ])
    ibl_trnsf = transforms.Compose([
        # IBL_Transform(),
        ToTensor()
    ])

    ibl_tensor = ibl_trnsf(ibl_img)
    c,h,w =ibl_tensor.size()
    ibl_tensor = ibl_tensor.view(1, c, h, w)
    # print('ibl: ', ibl_tensor.size())

    img_tensor = img_trnsf(img)
    c,h,w = img_tensor.size()
    img_tensor = img_tensor.view(1, c, h, w)
    model.eval()
    with torch.no_grad():
        I_s = img_tensor.to(device)
        L_t = ibl_tensor.to(device)
        predicted_img = model(I_s, L_t)
        predicted_img = predicted_img[0].detach().cpu().numpy()
        predicted_img = predicted_img[0].transpose((1,2,0))
        return predicted_img
    
def predict_testing_results(input_folder, output_folder):
    """ Given a testing set folder(real humans), predict results from that folder"""
    
#     img_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
#     print('Predict {} files'.format(len(img_files)))

#     def real_to_mask(img):
#         # print(np.max(img))
#         h, w, c = img.shape
#         mask = np.zeros((h, w, 3), dtype=np.uint8)
#         mask[:, :, 0], mask[:, :, 1], mask[:, :, 2] = img[:, :, 3], img[:, :, 3], img[:, :, 3]

#         # print(np.max(mask))

#         return mask

#     testing_ibl_file = os.path.join('/home/ysheng/Dataset/soft_shadow/train/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/imgs','00000000_light.png')
#     testing_ibl = Image.open(testing_ibl_file)
#     testing_ibl_tensor = to_one_batch(ibl_trnsf(testing_ibl)).to(device)

#     model.eval()
#     with torch.no_grad():
#         for file in img_files:
#             # print(file)
#             # print(input_folder)
#             file_path = os.path.join(input_folder, file)
#             img = np.array(Image.open(file_path))
#             img = real_to_mask(img)
#             # import pdb; pdb.set_trace()
#             mask_tensor = to_one_batch(img_trnsf(img))
#             I_s = torch.cat((mask_tensor, mask_tensor), dim=1).to(device)
#             L_t = testing_ibl_tensor
#             predicted_img, predicted_ibl = model(I_s, L_t)
#             save_results(predicted_img, os.path.join(output_folder, os.path.splitext(file)[0] + ".png"))
    print('not modified yet')
    
def render_animation(target_mask_np, output_folder, light_folder=""):
    """ Given a mask(w x h x 3, uint8), render a sequence of images for making an animation"""
    def get_first_ibl():
        tmp_np = np.zeros((256,512,1))
        tmp_np[0, 0] = 1.0
        tmp_np = gaussian_filter(tmp_np, 20)
        tmp_np = resize(tmp_np, (16,32))
        tmp_np = tmp_np/np.max(tmp_np) 
        return tmp_np
    
    def rotate_ibl(img_np, axis=1, step=1):
        """ rotate ibl along one axis for one pixel """
        new_np = np.copy(img_np)
        if axis == 1:
            # ---> direction
            new_np[:,step:] = img_np[:,:-step]
            new_np[:,0:step] = img_np[:,-step:]
        else:
            # | direction
            new_np[step:,:] = img_np[:-step,:]
            new_np[:step,:] = img_np[-step,:]

        return new_np
    
    def to_mask(target_mask_np):
        target_mask_np = (target_mask_np[:,:,0] + target_mask_np[:,:,1] + target_mask_np[:,:,2])/3.0
        target_mask_np = target_mask_np/np.max(target_mask_np)
        target_mask_np = resize(target_mask_np, (256,256,1))
        return target_mask_np
    
    def compute_ibl(i,j, w=512, h=256):
        """ given width, height, (i,j) compute the 16x32 ibls """
        ibl = np.zeros((h,w,1))
        ibl[j,i] = 1.0
        ibl = gaussian_filter(ibl, 20)
        ibl = resize(ibl, (16,32))
        ibl = ibl/np.max(ibl)
        return ibl
    
    def render_save(target_mask_np, ibl, predict_fname):
        mask = np.copy(target_mask_np)
        mask = np.squeeze(to_mask(mask))
    
        pred_shadow = predict(target_mask_np, ibl)
        saving_result = pred_shadow
        saving_result[:16,:32] = 1.0 - ibl
        saving_result[mask != 0] = 1.0
        
        np.clip(saving_result, 0.0, 1.0, out=saving_result)
        plt.imsave(predict_fname, mask_to_rgb(saving_result))
    
    # cur_ibl = get_first_ibl()
    counter, prefix = 0, 0
    w,h = 512,256 
    i_range, j_range = w, 1
    row_ind = int(256 * 0.6);
    
    with tqdm(total= i_range * j_range) as tbar:
        for j in range(j_range):
            for i in range(i_range):
                ibl = compute_ibl(i, j + row_ind)
                ibl = ibl + compute_ibl(w-1-i, j * 2 + row_ind)
                np.clip(ibl, 0.0, 1.0, out=ibl)
                out_fname = '{:07d}.png'.format(prefix)
                predict_fname = os.path.join(output_folder, out_fname)
                    
                render_save(target_mask_np, ibl, predict_fname)
                prefix += 1
                tbar.update()
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='testing images target folder')
    parser.add_argument('--out_folder', type=str, help='output folder')

    params = parser.parse_args()
    print('Params: {}'.format(params))
    predict_testing_results(params.folder, params.out_folder)

    # testing_ibl_file = os.path.join('/home/ysheng/Dataset/soft_shadow/single_human/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A','00000000_light.png')
    # testing_img = '/home/ysheng/Dataset/soft_shadow/real_human_testing_set/warrior-2-1.png'
    # img, ibl_img = plt.imread(testing_img), plt.imread(testing_ibl_file)
    # shadow_img = predict(img, ibl_img)
    # plt.imsave("testing.png",shadow_img)