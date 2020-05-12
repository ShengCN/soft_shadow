import numpy as np
import json
import pdb
import os
from os.path import join
import html
from tqdm import tqdm
import argparse

def get_files(folder):
    return [join(folder, f) for f in os.listdir(folder) if os.path.isfile(join(folder, f))]

def get_folders(folder):
    return [join(folder, f) for f in os.listdir(folder) if os.path.isdir(join(folder, f))]

vis_img_folder = 'imgs'
def eval_gen(webpage, output_folder, is_pattern=True):
    img_folders = join(output_folder, 'imgs')
    folders = get_folders(img_folders)
    print("There are {} folders".format(len(folders)))
    
    for model in tqdm(folders):
        cur_model_relative = join(vis_img_folder,os.path.basename(model))

        if is_pattern:
            ibl_relative = join(cur_model_relative, 'pattern')
        else:
            ibl_relative = join(cur_model_relative, 'real')
        
        # import pdb; pdb.set_trace()
        ibl_folders = get_folders(ibl_relative)
        ibl_folders.sort()
        for ibl in ibl_folders:
            cur_ibl_relative = join(ibl_relative,os.path.basename(ibl))
            ibl_name = os.path.basename(ibl)

            ibl = join(cur_ibl_relative, 'ibl.png')
            # mitsuba_shadow = join(cur_ibl_relative, 'mitsuba_shadow.png')
            mitsuba_shadow = join(cur_ibl_relative, 'mitsuba_shaddow_flipped.png')
            net_gt = join(cur_ibl_relative, 'gt.png')
            net_predict = join(cur_ibl_relative, 'predict.png')
            mitsuba_final = join(cur_ibl_relative, 'composite.png')
            
            # print(ibl_name)
            ims, txts, links = [ibl,mitsuba_shadow, net_gt, net_predict, mitsuba_final], ['{}'.format(ibl_name),'mitsuba shadow', 'net GT', 'net predict', 'mitsuba final'], [ibl, mitsuba_shadow, net_gt, net_predict, mitsuba_final]
            webpage.add_images(ims, txts, links)

vis_pattern_folder = '/home/ysheng/Documents/vis_pattern'
vis_real_folder = '/home/ysheng/Documents/vis_real'

def vis_files_in_folder():
    webpage = html.HTML(vis_pattern_folder, 'pattern evaluation', reflesh=1)    
    eval_gen(webpage, vis_pattern_folder)
    webpage.save()

    webpage = html.HTML(vis_real_folder, 'real evaluation', reflesh=1)    
    eval_gen(webpage, vis_real_folder, False)
    webpage.save()

    print('finished')

    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='evaluatoin pipeline')
    parser.add_argument('-p','--pattern', action='store_true', help='pattern?')

    options = parser.parse_args()

    if options.pattern:
        webpage = html.HTML(vis_pattern_folder, 'pattern evaluation', reflesh=1)    
        eval_gen(webpage, vis_pattern_folder)
        webpage.save()
    else:
        webpage = html.HTML(vis_real_folder, 'real evaluation', reflesh=1)    
        eval_gen(webpage, vis_real_folder, False)
        webpage.save()