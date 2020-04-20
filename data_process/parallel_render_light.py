import csv
import numpy as np
import os
from shadow_render import render_shadow
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from PIL import Image

def get_vector(vec_str):
    """input: x_y_z
       output: np array
    """
    x_ind = vec_str.find('_')
    x = float(vec_str[:x_ind])
    y_ind = vec_str[x_ind + 1:].find('_')
    
    x = float(vec_str[:x_ind])
    y = float(vec_str[x_ind + 1:x_ind + 1 + y_ind])
    z = float(vec_str[x_ind + 1 + y_ind + 1:])
    return np.array([x,y,z])

def crop_ibl(ibl_np):
    """input  256x512 ibl
       output 16x32 cropp
    """
    cropped_row = 256 - (30/90 + 1.0) * 0.5 * 256
    cropped = ibl_np[:int(cropped_row),:] 
    return np.array(Image.fromarray(cropped).resize((32,16), Image.BILINEAR))

def render_ibl(alpha, beta):
    ori_ibl_h,ori_ibl_w = 256, 512
    i,j = 0.5 * (alpha + np.pi)/np.pi ,(beta + 0.5*np.pi)/np.pi 
    i,j = int(i * ori_ibl_w), ori_ibl_h-int(j * ori_ibl_h)
    ibl_img = np.zeros((ori_ibl_h, ori_ibl_w))
    ibl_img[j, i] = 1.0
    return crop_ibl(ibl_img)

def render_worker(path_relative_vec):
    mask_path = path_relative_vec[0]
    relative_vector_str = path_relative_vec[1]
    
    relative_vec = get_vector(relative_vector_str)
    relative_vec = relative_vec/np.linalg.norm(relative_vec, 2)
    beta = np.arcsin(relative_vec[1])
    alpha = np.arctan2(relative_vec[2], relative_vec[0])
        
    base_name, folder = os.path.basename(mask_path), os.path.dirname(mask_path)
    out_file = os.path.join(folder, base_name[:base_name.find('_')] + '_light.npy')
    saved_np = render_ibl(alpha, beta)
    
    np.save(out_file, saved_np)
    
def parallel_render():
    dataset_folder = '/home/ysheng/Dataset/soft_shadow/train'
    out_file = os.path.join(dataset_folder, "metadata.csv")
    with open(out_file) as f:
        csv_read = csv.reader(f, delimiter=',')
        mask_path_list, rel_vec_list = [],[]
        for r in csv_read:
            mask_path_list.append(r[1])
            rel_vec_list.append(r[6])

    task_num = len(mask_path_list) 
    print(task_num)    
    processor_num = 512
    input_params = zip(mask_path_list, rel_vec_list)
    with multiprocessing.Pool(processor_num) as pool:
        # working_fn = partial(batch_working_process, src_folder, out_folder)
        for i,_ in enumerate(pool.imap_unordered(render_worker, input_params), 1):
            print("Finished: {} \r".format(float(i)/task_num), flush=True, end='')

def resize_worker(path):
    mask, shadow = path
    img = Image.open(shadow)
    img = img.resize((256,256))
    img.save(shadow)
    
    img = Image.open(mask)
    img = img.resize((256,256))
    img.save(mask)    
            
def parallel_resize():
    dataset_folder = '/home/ysheng/Dataset/soft_shadow/train'
    out_file = os.path.join(dataset_folder, "metadata.csv")
    with open(out_file) as f:
        csv_read = csv.reader(f, delimiter=',')
        mask_path_list = []
        shadow_path_list = []
        for r in csv_read:
            mask_path_list.append(r[1])
            shadow_path_list.append(r[2])

    task_num = len(mask_path_list) 
    print(task_num)    
    processor_num = 512
    
    input_list = zip(mask_path_list, shadow_path_list)
    with multiprocessing.Pool(processor_num) as pool:
        # working_fn = partial(batch_working_process, src_folder, out_folder)
        for i, _ in enumerate(pool.imap_unordered(resize_worker, input_list), 1):
            print("Finished: {} \r".format(float(i)/task_num), flush=True, end='')
            
if __name__ == '__main__':
    print('begin light')
    parallel_render()
    print('end light')
    
    print('begin resize')
    parallel_resize()
    print('end resize')
    