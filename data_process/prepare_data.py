import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import multiprocessing
from functools import partial
from PIL import Image
import time

def get_folders(folder):
    return [join(folder, f) for f in os.listdir(folder) if os.path.isdir(join(folder, f))] 

def get_files(folder):
    return [join(folder, f) for f in os.listdir(folder) if os.path.isfile(join(folder, f))] 

def mask_transform(param):
    mask_path, base_folder_path, base_file = param
    dname = os.path.dirname(mask_path)
    fname = os.path.splitext(os.path.basename(mask_path))[0]
    out_fname = join(dname, fname + ".npy") 
    np.save(out_fname, plt.imread(mask_path)[:,:,0])

    base_np = 1.0 - np.load(base_file)
    np.save(base_file, base_np)
    return dname, fname[:fname.find('_mask')], base_folder_path

def multithreading_post_process(folder, output_path):
    cache_folder = join(folder, 'cache')
    mask_folder = join(cache_folder, 'mask')
    model_folders = get_folders(mask_folder)
    base_folder = join(folder, 'base')
    print('models: ', len(model_folders))

    out_folder = join(folder, 'base')
    os.makedirs(out_folder, exist_ok=True)

    meta_data = ''
    files_list, output_folder_list, base_file_list = [], [], []

    for folder in tqdm(model_folders):
        files = [join(folder, f) for f in os.listdir(folder) if (os.path.isfile(join(folder, f))) and (f.find('png')!=-1)] 
        files_list += files
        
        fname = os.path.basename(folder)
        cur_out_folder = join(out_folder, fname)
        output_folder_list += [cur_out_folder] * len(files)

        cur_base_folder = join(base_folder, fname)
        base_files = get_files(cur_base_folder)
        base_file_list += base_files

    task_num = len(files_list)
    print('total task num: ', task_num)

    input_params = zip(files_list, output_folder_list, base_file_list)
    processor_num = 256
    with multiprocessing.Pool(processor_num) as pool:
        # working_fn = partial(batch_working_process, src_folder, out_folder)
        for i,(mask_dname, out_prefix, cur_base_folder) in enumerate(pool.imap_unordered(mask_transform, input_params), 1):
            meta_data += '{},{}\n'.format(join(cur_base_folder, out_prefix + '_shadow.npy'), join(mask_dname, out_prefix + '_mask.npy'))
            print("Finished: {} \r".format(float(i)/task_num), flush=True, end='')

    print(meta_data)
    with open(output_path,'w+') as f:
        f.write(meta_data)

if __name__ == '__main__':
    begin = time.time()
    dataset_folder = '/home/ysheng/Dataset/new_dataset'
    output_path = join(dataset_folder, 'meta_data.csv')

    multithreading_post_process(dataset_folder, output_path)

    end = time.time()
    print('time: {}'.format(end-begin))