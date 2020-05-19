import numpy as np
import json
import pdb
import os
from os.path import join
import html
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def get_files(folder):
    return [join(folder, f) for f in os.listdir(folder) if os.path.isfile(join(folder, f))]

def get_folders(folder):
    return [join(folder, f) for f in os.listdir(folder) if os.path.isdir(join(folder, f))]

vis_img_folder = 'imgs'
vis_eval_img_folder = 'eval_imgs'
def eval_gen(webpage, output_folder, is_pattern=True):
    def get_file(files, key_world):
        mitsuba_shadow = ''
        for f in files:
            if f.find(key_world) != -1:
                mitsuba_shadow = f
                break
        return mitsuba_shadow

    def flip_shadow(img_file):
        dirname, fname = os.path.dirname(img_file), os.path.splitext(os.path.basename(img_file))[0]
        if img_file == '':
            print('find one zero')
            mts_shadow_np = np.zeros((256,256,3))
        else:
            mts_shadow_np = plt.imread(img_file)
        
        save_path = join(dirname, fname + '_flip.png')
        plt.imsave(save_path, 1.0-mts_shadow_np)
        return save_path

    img_folders = join(output_folder, 'imgs')
    folders = get_folders(img_folders)
    print("There are {} folders".format(len(folders)))
    
    for model in tqdm(folders):
        cur_model_relative = join(vis_img_folder, os.path.basename(model))
        evl_cur_model_relative = join(vis_eval_img_folder, os.path.basename(model))

        if is_pattern:
            ibl_relative = join(cur_model_relative, 'pattern')
        else:
            ibl_relative = join(cur_model_relative, 'real')
        
        # import pdb; pdb.set_trace()
        ibl_folders = get_folders(ibl_relative)
        ibl_folders.sort()
        for ibl in ibl_folders:
            cur_ibl_relative = join(ibl_relative,os.path.basename(ibl))
            gt_files = get_files(cur_ibl_relative)
            mts_shadow = get_file(gt_files, '_shadow.png')

            ibl_name = os.path.basename(ibl)
            ibl = join(cur_ibl_relative, ibl_name + '.png')
           
            mitsuba_shadow = flip_shadow(mts_shadow)

            cur_eval_folder = join(evl_cur_model_relative, join('pattern', ibl_name))
            net_predict = get_file(get_files(cur_eval_folder), 'predict.png')

            # mitsuba_final = join(cur_ibl_relative, 'composite.png')
            # pred_final = join(cur_ibl_relative, 'composite_pred.png')
            
            # print(ibl_name)
            ims, txts, links = [ibl,mitsuba_shadow, net_predict], ['ibl','mitsuba', 'predict'], [ibl,mitsuba_shadow, net_predict]

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