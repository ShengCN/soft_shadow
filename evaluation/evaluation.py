import sys
sys.path.append("..")

import os 
from os.path import join
import argparse
import time
from tqdm import tqdm
from ssn.ssn import Relight_SSN

parser = argparse.ArgumentParser(description='evaluatoin pipeline')
parser.add_argument('-f', '--file', type=str, help='input model file')
parser.add_argument('-m', '--mask', type=str, help='mask file')
parser.add_argument('-i', '--ibl', type=str, help='ibl file')
parser.add_argument('-o', '--output', type=str, help='output folder')
parser.add_argument('-w', '--weight', type=str, help='weight of current model', default='../weights/new_pattern_05-May-02-14-AM.pt')
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

mts_final_xml,mts_shadow_xml = 'mts_final.xml', 'mts_shadow.xml'
def mitsuba_render(model_file, ibl_file, final_out_file, shadow_out_file):
    """ Input: mitsuba rendering related resources
        Output: rendered_gt, saved shadow image 
    """
    final_out_folder, shadow_out_folder = os.path.dirname(final_out_file), os.path.dirname(shadow_out_file)
    # parse camera parameters, human matrix

    # ground plane model path

    # prepare an xml into this folder that has parameter for model file and ibl file, output_file
    shadow_cmd = 'mitsuba {} -Dsamples={} -Dori={} -Dtarget={} -Dup={} -Dibl={} -Dground={} -Dmodel={} -Dworld={} -o={}'.format(mts_final_xml, 4, )
    final_cmd = 'mitsuba {} -Dsamples={} -Dori={} -Dtarget={} -Dup={} -Dibl={} -Dground={} -Dmodel={} -Dworld={} -o={}'.format(mts_final_xml, 4, )
    os.system(shadow_cmd)
    os.system(final_cmd)
    mts_util_tonemapping_cmd = 'mtsutil tonemap {}/*.exr'.format(final_out_folder)
    os.system(mts_util_tonemapping_cmd)


def net_render(mask_file, ibl_file, out_file):
    pass

def merge_result(rendered_img, mask_file, shadow_img, out_file):
    pass

def evaluate(model_file, mask_file, ibl_file, output):
    mitsuba_final = join(output, 'mitsuba_final.png')
    mitsuba_shadow_output, mitsuba_merge = join(output, 'mitsuba_shadow.png'), join(output, 'mitsuba_merge.png')
    net_shadow_output, net_merge = join(output, 'prediction_shadow.png'), join(output, 'prediction_merge.png')

    # call mitsuba render 
    mitsuba_render(model_file, ibl_file, mitsuba_final, mitsuba_shadow_output)

    # call net render result
    net_render(mask_file, ibl_file, net_shadow_output)

    # merge result
    merge_result(mitsuba_final, )

if __name__ == '__main__':
    evaluate(options.file, options.mask, options.ibl, options.output)