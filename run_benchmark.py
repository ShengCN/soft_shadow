import os
from os.path import join
import glob
from ssn.ssn_touch import SSN_Touch 
from evaluation.exp_predict import get_mask_ibl, ssn_touch_pred
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def run_prediction(model, benchmark_folder, out_folder, baseline=True):
    def run_one_folder(model, folder, out_folder):
        model_folders = glob.glob(join(folder, '*'))
        human_general = os.path.basename(folder)
        if human_general == 'general':
            is_general = True
        else:
            is_general = False

        for mf in tqdm(model_folders):
            # create output folder
            mf_base = os.path.basename(mf)
            cur_out_folder = join(out_folder, mf_base)
            os.makedirs(cur_out_folder, exist_ok=True)

            gts = glob.glob(join(mf, '*.png'))
            for gt in gts:
                mask, ibl = get_mask_ibl(gt, is_general) 
                img = ssn_touch_pred(model, mask, ibl, device, baseline)
                img = np.repeat(img, 3, axis=2)
                
                cv2.normalize(img, img, 0.0,1.0, cv2.NORM_MINMAX)
                
                opath = join(cur_out_folder, os.path.basename(gt))
                plt.imsave(opath,img)

                np_save_fname = join(cur_out_folder,os.path.splitext(os.path.basename(gt))[0] + ".npy")
                np.save(np_save_fname, img)

    general_out, human_out = join(out_folder, 'general'), join(out_folder, 'human')
    os.makedirs(general_out,exist_ok=True)
    os.makedirs(human_out,exist_ok=True)

    run_one_folder(model, join(benchmark_folder, 'general'), general_out)
    run_one_folder(model, join(benchmark_folder, 'human'), human_out)

def run_metric(gt_folder, pred_folder):
    pass

if __name__ == '__main__':
    model = SSN_Touch()
    model.to(device)
    weight_file = 'weights/new_arch_baseline.pt'
    checkpoint = torch.load(weight_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    benchmark_folder = '/home/ysheng/Dataset/benchmark_ds'
    exp_name = 'new_arch_general_baseline'
    out_folder = join(benchmark_folder, exp_name)
    os.makedirs(out_folder, exist_ok=True)

    run_prediction(model, join(benchmark_folder,'shadow_gt'), out_folder, baseline=True)