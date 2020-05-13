import sys
sys.path.append("..")

import os 
from os.path import join
import argparse
import time
from tqdm import tqdm
from ssn.ssn import Relight_SSN
import pickle

parser = argparse.ArgumentParser(description='evaluatoin pipeline')
parser.add_argument('-f', '--file', type=str, help='input model file')
parser.add_argument('-m', '--mask', type=str, help='mask file')
parser.add_argument('-i', '--ibl', type=str, help='ibl file')
parser.add_argument('-o', '--output', type=str, help='output folder')
parser.add_argument('-w', '--weight', type=str, help='weight of current model', default='../weights/new_pattern_05-May-02-14-AM.pt')
parser.add_argument('-v', '--verbose', action='store_true', help='output file name')

options = parser.parse_args()
print('options: ', options)

# device = torch.device("cpu")
# # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# print("Device: ", device)
# model = Relight_SSN(1,1)
# weight_file = options.weight
# checkpoint = torch.load(weight_file, map_location=device)    
# model.to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
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

def parse_camera_world():
    cam_world_file = './cam_world_dict.pkl'
    cam_world_dict = dict()
    if os.path.exists(cam_world_file):
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

    return cam_world_dict

mts_final_xml,mts_shadow_xml = 'mts_final.xml', 'mts_shadow.xml'
def mitsuba_render(model_file, ibl_file, final_out_file, shadow_out_file):
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

    # prepare an xml into this folder that has parameter for model file and ibl file, output_file
    shadow_cmd = 'mitsuba {} -Dw=256 -Dh=256 -Dsamples={} -Dori=\"{}\" -Dtarget=\"{}\" -Dup=\"{}\" -Dibl=\"{}\" -Dground={}\" -Dmodel=\"{}\" -Dworld=\"{}\" -o \"{}\"'.format(
        mts_shadow_xml, 4, cam[0], cam[1], cam[2], ibl_file, ground_path, model_file, world, shadow_out_file)

    final_cmd = 'mitsuba {} -Dw=256 -Dh=256 -Dsamples={} -Dori=\"{}\" -Dtarget=\"{}\" -Dup=\"{}\" -Dibl=\"{}\" -Dground={}\" -Dmodel=\"{}\" -Dworld=\"{}\" -o \"{}\"'.format(
        mts_final_xml, 4, cam[0], cam[1], cam[2], ibl_file, ground_path, model_file, world, final_out_file)
    with open('test.txt','w+') as f:
        f.write(shadow_cmd)
        f.write('\n')
        f.write(final_cmd)

    # print('shadow cmd: \n', shadow_cmd)
    # print('final cmd: \n', final_cmd)

    os.system(shadow_cmd)
    mts_util_tonemapping_cmd = 'mtsutil tonemap {}'.format(shadow_out_file)
    os.system(mts_util_tonemapping_cmd)

    os.system(final_cmd)
    mts_util_tonemapping_cmd = 'mtsutil tonemap -m {} {}'.format(1, final_out_file)
    os.system(mts_util_tonemapping_cmd)


def net_render(mask_file, ibl_file, out_file):
    pass

def merge_result(rendered_img, mask_file, shadow_img, out_file):
    pass

def evaluate(model_file, mask_file, ibl_file, output):
    mitsuba_final = join(output, 'mitsuba_final.exr')
    mitsuba_shadow_output, mitsuba_merge = join(output, 'mitsuba_shadow.exr'), join(output, 'mitsuba_merge.png')
    net_shadow_output, net_merge = join(output, 'prediction_shadow.png'), join(output, 'prediction_merge.png')

    # call mitsuba render 
    mitsuba_render(model_file, ibl_file, mitsuba_final, mitsuba_shadow_output)

    # call net render result
    net_render(mask_file, ibl_file, net_shadow_output)

    # merge result
    merge_result(mitsuba_final, )

if __name__ == '__main__':
    # evaluate(options.file, options.mask, options.ibl, options.output)
    model_file = '/Data_SSD/models/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj'
    mask_file = '/Data_SSD/new_dataset/cache/mask/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/pitch_15_rot_0_mask.png'
    ibl_file = '/home/ysheng/Dataset/ibls/real/20060430-01_hd.hdr'
    # ibl_file = '../test_pattern.png'
    output = 'dbg/'
    os.makedirs(output, exist_ok=True)
    evaluate(model_file, mask_file, ibl_file, output)