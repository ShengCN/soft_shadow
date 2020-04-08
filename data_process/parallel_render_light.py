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

"""input: x_y_z
   output: np array
"""
def get_vector(vec_str):
    x_ind = vec_str.find('_')
    x = float(vec_str[:x_ind])
    y_ind = vec_str[x_ind + 1:].find('_')
    
    x = float(vec_str[:x_ind])
    y = float(vec_str[x_ind + 1:x_ind + 1 + y_ind])
    z = float(vec_str[x_ind + 1 + y_ind + 1:])
    return np.array([x,y,z])

def render_worker(path_relative_vec):
    mask_path = path_relative_vec[0]
    relative_vector_str = path_relative_vec[1]
    
    relative_vec = get_vector(relative_vector_str)
    folder = os.path.dirname(mask_path)
    prefix = os.path.basename(mask_path)[:os.path.basename(mask_path).find("_")]
    light_path = os.path.join(folder,"{}_light.png".format(prefix))
    img = render_shadow(relative_vec)
    
    img = img/255.0
    img = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3.0
    
    # print('1 min: {} max: {} shape: {}'.format(np.min(img),np.max(img), img.shape))
    img = gaussian_filter(img, sigma=20)
    # print('2 min: {} max: {} shape: {}'.format(np.min(img),np.max(img), img.shape))
    img = resize(img,(16,32,1))
    img = img/np.max(img)
    
    plt.imsave(light_path, img)
    
def parallel_render():
    dataset_folder = '/home/ysheng/Dataset/soft_shadow/train'
    out_file = os.path.join(dataset_folder, "metadata.csv")
    with open(out_file) as f:
        csv_read = csv.reader(f, delimiter=',')
        mask_path_list, rel_vec_list = [],[]
        for r in csv_read:
            mask_path_list.append(r[1])
            rel_vec_list.append(r[-1])

    task_num = len(mask_path_list) 
    print(task_num)    
    processor_num = 48
    with multiprocessing.Pool(processor_num) as pool:
        # working_fn = partial(batch_working_process, src_folder, out_folder)
        for i, _ in enumerate(pool.imap_unordered(render_worker, zip(mask_path_list, rel_vec_list)), 1):
            print("Finished: {} \r".format(float(i)/task_num), flush=True, end='')

def resize_worker(path):
    img = Image.open(path)
    img.resize((256,256))
    img.save(path)
            
def parallel_resize():
    dataset_folder = '/home/ysheng/Dataset/soft_shadow/train'
    out_file = os.path.join(dataset_folder, "metadata.csv")
    with open(out_file) as f:
        csv_read = csv.reader(f, delimiter=',')
        mask_path_list = []
        for r in csv_read:
            mask_path_list.append(r[1])

    task_num = len(mask_path_list) 
    print(task_num)    
    processor_num = 48
    with multiprocessing.Pool(processor_num) as pool:
        # working_fn = partial(batch_working_process, src_folder, out_folder)
        for i, _ in enumerate(pool.imap_unordered(resize_worker, mask_path_list), 1):
            print("Finished: {} \r".format(float(i)/task_num), flush=True, end='')
            
if __name__ == '__main__':
    # parallel_render()
    parallel_resize()