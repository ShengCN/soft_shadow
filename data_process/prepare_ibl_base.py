import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import multiprocessing
from functools import partial
from PIL import Image

def get_files(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

if __name__ == '__main__':
    path = 'D:/HPCG/Adobe/labels/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A'
    gt_file = os.path.join(path, 'ground_truth.txt')
    lines = []

    with open(gt_file) as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            lines.append(r)

    print('there are {} lines'.format(len(lines)))
    
    # construct the map
    # [camera, human_rot] -> [[i, j]->[prefix]]
    # oss << cur_prefix << ",";
    # oss << light_pixel_pos.to_string() << ",";
    # oss << to_string(manager.cur_camera->_position) << ",";
    # oss << target_rot << ",";
    # oss << to_string(render_target->compute_world_center()) << ",";
    # oss << to_string(manager.m_lights[0]->m_verts[0]) << std::endl;
    # gt_str.push_back(oss.str());
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

    print(len(group_data.keys()))
    # h x w x x x y 
    h, w, x, y = 256,256, 256-176, 512 
    all_data = np.zeros((h, w, x, y))
    

    info_str = ""
    new_prefix = 0
    for key in group_data.keys():
        # 176~256 x 0~512
        # 16x16
        rows = (256-176)//16
        cols = 512//16
        for i in range(rows):
            for j in range(cols):
                prefix = group_data[key][(176 + i * 16, j * 16)]
                # compute base
