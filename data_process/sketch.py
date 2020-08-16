import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from tqdm import tqdm

def get_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def get_folders(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

def sketch(normal, depth):
    normal_img, depth_img = cv2.imread(normal), cv2.imread(depth)
    normal_edge = cv2.Canny(normal_img, 0, 200)
    depth_edge = cv2.Canny(depth_img, 0, 10)
    alpha = 0.3
    merged_edge = normal_edge * (1.0-alpha) + depth_edge * alpha
    return merged_edge

def render_sketches(ds_folder, out_folder):
    def get_prefix_list(files):
        prefix_list = set()
        for f in files:
            if f.find('_mask') != -1:
                fname = os.path.basename(f)
                prefix_list.add(fname[:fname.find('_mask')])
        return list(prefix_list)

    ds_folder = '/home/ysheng/Dataset/general_dataset/'
    cache_folder = join(ds_folder, 'cache/shadow_output')
    obj_list = get_folders(cache_folder)
    print('there are {} files'.format(len(obj_list)))
    for m in tqdm(obj_list):
        files = get_files(m)
        prefix_list = get_prefix_list(files)
        
        model_name = os.path.basename(m)
        out_sketch_folder = join(out_folder, model_name)
        os.makedirs(out_sketch_folder, exist_ok=True)
        
        for prefix in prefix_list:
            normal_img, depth_img = join(m,'{}_normal.png'.format(prefix)), join(m,'{}_depth.png'.format(prefix))
            sketch_img = sketch(normal_img, depth_img)
            
            plt.imsave(join(out_sketch_folder, '{}_sketch.png'.format(prefix)),sketch_img)
        
        
if __name__ == '__main__':
    render_sketches('/home/ysheng/Dataset/general_dataset/', '/home/ysheng/Dataset/general_dataset/sketch')