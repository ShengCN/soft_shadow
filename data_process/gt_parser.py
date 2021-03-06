import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir,"utils")) 
from utils_file import get_all_folders
import csv
from tqdm import tqdm
import numpy as np
import multiprocessing
from functools import partial

""" Parse the GT file to get result dict
    (prefix) -> {values}
"""
def parse(file_path):
    if not os.path.exists(file_path):
        print("cannot find file ", file_path)
        return {}
    
    line_list = []
    with open(file_path) as gt_f:
        for i, line in enumerate(gt_f):
            # print("{} {}".format(i, line))
            line_list.append(line)
    
    seg_num = 6
    data_num = int(len(line_list)/seg_num)
    # print("There are: {} data".format(data_num))
    gt_dict = {}
    
    # slice the gt data file for each data
    for i in range(data_num):
        prefix = line_list[seg_num * i + 0]
        group_num = line_list[seg_num * i + 1][-2]
        
        camera_pos_key, camera_pos_value = line_list[seg_num * i + 2].split()[0], line_list[seg_num * i + 2].split()[1:]
        human_rot_key,human_rot_value = line_list[seg_num * i + 3].split()[0], line_list[seg_num * i + 3].split()[1:]
        human_pos_key, human_pos_value = line_list[seg_num * i + 4].split()[0], line_list[seg_num * i + 4].split()[1:]
        
        light_info = line_list[seg_num * i + 5].split()
        light_pos_key, light_pos_value = light_info[0], light_info[1:]
        
        gt_dict[prefix] = {
            'group_num':group_num,
            camera_pos_key:camera_pos_value,
            human_rot_key:human_rot_value, 
            human_pos_key:human_pos_value, 
            light_pos_key:light_pos_value
        }

    return gt_dict


def parse_worker(cur_folder):
    gt_file = os.path.join(cur_folder, "ground_truth.txt")
    gt_dict = parse(gt_file)
    lines = ""
    for key, value in gt_dict.items():
        prefix = "imgs/{:07d}".format(int(key))
        mask_file = os.path.join(cur_folder, prefix.split()[0] + "_mask.png")
        shadow_file = os.path.join(cur_folder, prefix.split()[0] + "_shadow.png")
        light_file = os.path.join(cur_folder, prefix.split()[0] + "_light.png")
        light_np_file = os.path.join(cur_folder, prefix.split()[0] + "_light.npy")

        human_pos,light_pos = value['human_position'],value['light_position']
        human_pos = np.array([float(human_pos[0]), float(human_pos[1]), float(human_pos[2])])
        light_pos = np.array([float(light_pos[0]), float(light_pos[1]),float(light_pos[2])])
        relative_pos = light_pos - human_pos
        relative_pos_str = '{}_{}_{}'.format(relative_pos[0], relative_pos[1], relative_pos[2])
        camera_pos = '{}_{}_{}'.format(value['camera_position'][0], value['camera_position'][1], value['camera_position'][2])
        human_rot = value['human_rotation_alpha']
        group_num = value['group_num']
        lines += '{},{},{},{},{},{},{},{},{}\n'.format(cur_folder, 
                                                       mask_file, 
                                                       shadow_file, 
                                                       light_file,
                                                       camera_pos, 
                                                       human_rot,
                                                       relative_pos_str, 
                                                       light_np_file, 
                                                       group_num)
    return lines

def parse_folder(folder, out_file, processor_num = 256):
    """ parse sub-folders from folder and generate a metadata csv file """
    folders = get_all_folders(folder)
    path_folders = [os.path.join(folder, f) for f in folders ]
    
    lines = ""
    task_num = len(path_folders)
    with multiprocessing.Pool(processor_num) as pool:
        for i, line in enumerate(pool.imap_unordered(parse_worker, path_folders), 1):
            lines += line
            print("Finished: {} \r".format(float(i)/task_num), flush=True, end='')
    
    with open(out_file, "w+") as f:
        f.write(lines)
    
def get_final_path(dir, prefix):
    return os.path.join(dir, prefix.split()[0] + "_final.png")

def get_mask_path(dir, prefix):
    return os.path.join(dir, prefix.split()[0] + "_mask.png")

def get_shadow_path(dir, prefix):
    return os.path.join(dir, prefix.split()[0] + "_shadow.png")