import sys
sys.path.append("..")

import os 
from os.path import join
import argparse
import time
from tqdm import tqdm
from ssn.ssn import Relight_SSN
from ssn.ssn_dataset import ToTensor
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='evaluatoin pipeline')
parser.add_argument('-f', '--file', type=str, help='input model file')
parser.add_argument('-m', '--mask', type=str, help='mask file')
parser.add_argument('-i', '--ibl', type=str, help='ibl file')
parser.add_argument('-o', '--output', type=str, help='output folder')
parser.add_argument('-w', '--weight', type=str, help='weight of current model', default='../weights/new_pattern_07-May-06-05-PM.pt')
parser.add_argument('-v', '--verbose', action='store_true', help='output file name')

options = parser.parse_args()
print('options: ', options)

device = torch.device("cpu")
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = Relight_SSN(1,1)
weight_file = options.weight
checkpoint = torch.load(weight_file, map_location=device)    
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
to_tensor = ToTensor()

cam_world_dict = {}
def get_files(folder):
    return [join(folder, f) for f in os.listdir(folder) if os.path.isfile(join(folder, f))]

def parse_cam_world_str(lines):
    cam, world = [], ''
    cam = [lines[0], lines[1], lines[2]]
    world_str = lines[3] + ',' + lines[4] + ',' + lines[5] + ',' + lines[6]
    world_elements = world_str.split(',')

    world = ''
    for w in world_elements:
        world += w + ' '
    return cam, world

def parse_camera_world(update=True):
    cam_world_file = './cam_world_dict.pkl'
    cam_world_dict = dict()
    if not update and os.path.exists(cam_world_file):
        with open(cam_world_file, 'rb') as f:
            cam_world_dict = pickle.load(f)
    else:
        cam_world_folder = '/home/ysheng/Dataset/mts_params/'
        folders = [join(cam_world_folder, f) for f in os.listdir(cam_world_folder) if os.path.isdir(join(cam_world_folder, f))]
        print('there are {} folders'.format(len(folders)))
        for f in folders:
            basename = os.path.basename(f)
            if basename not in cam_world_dict.keys():
                cam_world_dict[basename] = dict()

            cam_world_files = get_files(f)
            for cam_world in cam_world_files:
                lines = []
                with open(cam_world) as f:
                    for l in f:
                        lines.append(l.rstrip('\n'))
                fname = os.path.splitext(os.path.basename(cam_world))[0]
                cam_world_dict[basename][fname] = parse_cam_world_str(lines)
        
        with open(cam_world_file, 'wb') as handle:
            pickle.dump(cam_world_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cam_world_dict

def to_net_ibl(ibl_file):
    """ input:  32 x 16
        output: 32 x 5
    """
    ibl = cv2.imread(ibl_file) / 255.0
    return ibl[:5,:,0]

mts_final_xml,mts_shadow_xml = 'mts_final.xml', 'mts_shadow.xml'
def mitsuba_render(model_file, ibl_file, final_out_file, shadow_out_file, real_ibl=True):
    """ Input: mitsuba rendering related resources
        Output: rendered_gt, saved shadow image 
    """
    final_out_folder, shadow_out_folder = os.path.dirname(final_out_file), os.path.dirname(shadow_out_file)
    # parse camera parameters, human matrix
    cam_world_dict = parse_camera_world()
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    model_cam_world_dict = cam_world_dict.get(model_name)
    cam, world = model_cam_world_dict[list(model_cam_world_dict.keys())[0]]

    # ground plane model path
    ground_path = '"/home/ysheng/Dataset/models/ground/ground.obj'

    samples = 256
    # prepare an xml into this folder that has parameter for model file and ibl file, output_file
    shadow_cmd = 'mitsuba {} -Dw=256 -Dh=256 -Dsamples={} -Dori=\"{}\" -Dtarget=\"{}\" -Dup=\"{}\" -Dibl=\"{}\" -Dground={}\" -Dmodel=\"{}\" -Dworld=\"{}\" -o \"{}\"'.format(
        mts_shadow_xml, samples, cam[0], cam[1], cam[2], ibl_file, ground_path, model_file, world, shadow_out_file)

    final_cmd = 'mitsuba {} -Dw=256 -Dh=256 -Dsamples={} -Dori=\"{}\" -Dtarget=\"{}\" -Dup=\"{}\" -Dibl=\"{}\" -Dground={}\" -Dmodel=\"{}\" -Dworld=\"{}\" -o \"{}\"'.format(
        mts_final_xml, samples, cam[0], cam[1], cam[2], ibl_file, ground_path, model_file, world, final_out_file)
    
    # with open('test.txt','w+') as f:
    #     f.write(shadow_cmd)
    #     f.write('\n')
    #     f.write(final_cmd)

    os.system(shadow_cmd)
    mts_util_tonemapping_cmd = 'mtsutil tonemap {}'.format(shadow_out_file)
    os.system(mts_util_tonemapping_cmd)

    os.system(final_cmd)
    if real_ibl:
        tone_scale = 5.0
    else:
        tone_scale = 80
    mts_util_tonemapping_cmd = 'mtsutil tonemap -m {} {}'.format(tone_scale, final_out_file)
    os.system(mts_util_tonemapping_cmd)


def net_gt(mask_file, ibl_file, out_file):
    # given mask, get bases
    dirname = os.path.basename(os.path.dirname(mask_file))
    pitch_rot = os.path.splitext(os.path.basename(mask_file))[0]
    pitch_rot = pitch_rot[:pitch_rot.find('_mask')]

    ibl_base_folder = join('/home/ysheng/Dataset/new_dataset/base/', dirname)
    shadow_base_file = join(ibl_base_folder, pitch_rot + '_shadow.npy')

    # use ibl to compute the gt shadow
    shadow_bases = np.load(shadow_base_file)
    h, w, iw, ih = shadow_bases.shape
    
    # remember, this ibl should always be 80 x 512
    # ibl = np.load(ibl_file)
    ibl = to_net_ibl(ibl_file)
    ibl = cv2.flip(ibl, 0)

    # ibl = cv2.resize(ibl, (iw, ih), interpolation=cv2.INTER_NEAREST)
    shadow = np.tensordot(shadow_bases, ibl, axes=([2,3], [1,0]))

    # save
    np.save(out_file, shadow)
    print('net gt finish')
    dirname, fname = os.path.dirname(out_file), os.path.splitext(os.path.basename(out_file))[0]
    png_output = os.path.join(dirname, fname + '.png')
    cv2.normalize(shadow, shadow, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imsave(png_output, shadow, cmap='gray')

def net_render(mask_file, ibl_file, out_file):
    s = time.time()
    ibl = cv2.flip(to_net_ibl(ibl_file), 0)
    mask, ibl = to_tensor(np.expand_dims(np.load(mask_file), axis=2)), to_tensor(np.expand_dims(cv2.resize(ibl, (32,16), interpolation=cv2.INTER_NEAREST), axis=2))
    with torch.no_grad():
        I_s, L_t = torch.unsqueeze(mask.to(device),0), torch.unsqueeze(ibl.to(device),0)

        predicted_img, predicted_src_light = model(I_s, L_t)

    shadow_predict = np.squeeze(predicted_img[0].detach().cpu().numpy().transpose((1,2,0)))
    np.save(out_file, shadow_predict)
    
    dirname, fname = os.path.dirname(out_file), os.path.splitext(os.path.basename(out_file))[0]
    png_output = os.path.join(dirname, fname + '.png')
    cv2.normalize(shadow_predict, shadow_predict, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imsave(png_output, shadow_predict, cmap='gray')

    print('net predict {} finished, time: {}s'.format(out_file, time.time() -s))

def merge_result(rendered_img, mask_file, shadow_img, out_file):
    pass

def evaluate(model_file, mask_file, ibl_file, output, real_ibl=True):
    """ output/mitsuba_final.png
        output/mitsuba_shadow.png
        output/mitsuba_merge.png
        output/prediction_shadow.npy
        output/net_gt_shadow.npy
        output/predcition_merge.npy
    """
    mitsuba_final = join(output, 'mitsuba_final.exr')
    mitsuba_shadow_output, mitsuba_merge = join(output, 'mitsuba_shadow.exr'), join(output, 'mitsuba_merge.png')
    net_shadow_output, net_merge = join(output, 'prediction_shadow.npy'), join(output, 'prediction_merge.png')
    net_gt_output, net_gt_merge = join(output, 'net_gt_shadow.npy'), join(output, 'net_gt_merge.npy')

    # call mitsuba render 
    # mitsuba_render(model_file, ibl_file, mitsuba_final, mitsuba_shadow_output, real_ibl)

    # call net render result
    net_render(mask_file, ibl_file, net_shadow_output)
    net_gt(mask_file, ibl_file, net_gt_output)
    
    # merge result
    # merge_result(mitsuba_final, )

if __name__ == '__main__':
    # evaluate(options.file, options.mask, options.ibl, options.output)
    model_file = '/Data_SSD/models/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj'
    mask_file = '/Data_SSD/new_dataset/cache/mask/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/pitch_15_rot_0_mask.npy'
    # ibl_file = '/home/ysheng/Dataset/ibls/real/20060430-01_hd.hdr'
    # ibl_file = '../test_pattern.png'
    
    output = 'dbg/'
    os.makedirs(output, exist_ok=True)
    ibl_file = '../test_pattern.png'
    evaluate(model_file, mask_file, ibl_file, output, False)