import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import multiprocessing
from functools import partial
from PIL import Image
import time

def base_compute(input):
    x, y, shadow_list = input
    ret_np = np.zeros((256,256))
    for shadow_path in shadow_list:
        ret_np += plt.imread(shadow_path)[:,:,0]

    return x,y, ret_np

def multithreading_post_process(folder, base_size=16):
    path = folder
    output_folder = os.path.join(path, 'base')
    os.makedirs(output_folder, exist_ok=True)
    gt_file = os.path.join(path, 'ground_truth.txt')
    lines = []

    with open(gt_file) as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            lines.append(r)

    print('there are {} lines'.format(len(lines)))

    group_data = {}
    for l in tqdm(lines):
        prefix = l[0]
        ibl = (int(l[1]), int(l[2]))
        camera_pos = (l[3], l[4], l[5])
        rot = l[6]
        target_center = (l[7], l[8], l[9])
        light_pos = (l[10], l[11], l[12])

        key = (camera_pos, rot)
        if key not in group_data.keys():
            group_data[key] = dict()

        ibl_key = ibl
        group_data[key][ibl_key] = prefix

    print('keys: ', len(group_data.keys()))
    print('keys: ', group_data.keys())
    img_folder = os.path.join(path, 'imgs')
    x_begin, y_begin = 0, 170

    for key_id, key in enumerate(group_data.keys()):
        # prepare mask
        prefix = group_data[key][(x_begin,y_begin)]
        mask_np = plt.imread(os.path.join(img_folder, '{}_mask.png'.format(prefix)))
        mask_output = os.path.join(output_folder, '{:03d}_mask.npy'.format(key_id))
        np.save(mask_output, mask_np)

        # prepare shadow
        input_list = []
        x_iter, y_iter = 512//base_size, (256-y_begin) // base_size
        group_np = np.zeros((256,256, x_iter, y_iter))
        for xi in tqdm(range(x_iter)):
            for yi in range(y_iter):
               # share all shadow results
                tuple_input = [xi, yi]
                shaodw_list = [os.path.join(img_folder,
                                            '{}_shadow.png'.format(group_data[key][(xi * base_size + i, y_begin + yi * base_size + j)]))
                               for i in range(base_size)
                               for j in range(base_size)]
                tuple_input.append(shaodw_list)
                input_list.append(tuple_input)
        processer_num, task_num = 128, len(input_list)
        with multiprocessing.Pool(processer_num) as pool:
            for i, base in enumerate(pool.imap_unordered(base_compute, input_list), 1):
                x,y, base_np = base[0], base[1], base[2]
                group_np[:,:,x,y] = base_np
                print("Finished: {} \r".format(float(i) / task_num), flush=True, end='')


        output_path = os.path.join(output_folder, '{:03d}'.format(key_id))
        np.save(output_path, group_np)
        del group_np

def memory_post_process(folder, base_size=16):
    path = folder
    output_folder = os.path.join(path, 'base')

    os.makedirs(output_folder, exist_ok=True)

    gt_file = os.path.join(path, 'ground_truth.txt')
    lines = []

    with open(gt_file) as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            lines.append(r)

    print('there are {} lines'.format(len(lines)))

    group_data = {}
    for l in tqdm(lines):
        prefix = l[0]
        ibl = (int(l[1]), int(l[2]))
        camera_pos = (l[3], l[4], l[5])
        rot = l[6]
        target_center = (l[7], l[8], l[9])
        light_pos = (l[10], l[11], l[12])

        key = (camera_pos, rot)
        if key not in group_data.keys():
            group_data[key] = dict()

        ibl_key = ibl
        group_data[key][ibl_key] = prefix

    print('keys: ', len(group_data.keys()))
    print('keys: ', group_data.keys())

    # h x w x x x y
    x_begin, y_begin = 0, 170
    h, w, x, y = 256, 256, 512, 256 - y_begin
    print(h * w * x * y * 4 / 1024 / 1024 / 1024)
    all_data = np.zeros((h, w, x, y))

    info_str,new_prefix  = "", 0
    img_folder = os.path.join(path, 'imgs')
    meta_str, new_base_prefix = "", 0

    for key_id, key in enumerate(group_data.keys()):
        # loading data
        for (ibl, prefix) in tqdm(group_data[key].items()):
            x, y = ibl
            shadow_path = os.path.join(img_folder, prefix + '_shadow.png')
            shadow_np = plt.imread(shadow_path)

            cur_x, cur_y = x - x_begin, y - y_begin
            all_data[:, :, cur_x, cur_y] = shadow_np[:, :, 0]

        # saving new dataset
        x_range, y_range = base_size, base_size
        x_iter, y_iter = 512 // x_range, (256 - y_begin) // y_range

        output_bases = np.zeros((256, 256, x_iter, y_iter))
        for xi in tqdm(range(x_iter)):
            for yi in range(y_iter):
                x_pos, y_pos = xi * base_size, yi * base_size
                new_ibl = np.sum(all_data[:, :, x_pos:x_pos + x_range, y_pos:y_pos + y_range], axis=(2, 3))

                output_bases[:,:,xi, yi] = new_ibl
        np_output_path = os.path.join(output_folder, '{:03d}.npy'.format(key_id))
        np.save(np_output_path, output_bases)
        del output_bases

    print('finsihed')

if __name__ == '__main__':
    begin = time.time()
    folder =  '/home/ysheng/Dataset/soft_shadow/new_dataset/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A'
    # memory_post_process(folder)
    multithreading_post_process(folder)
    end = time.time()
    print('time: {}'.format(end-begin))