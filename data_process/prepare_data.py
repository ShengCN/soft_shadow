from parallel_render_light import parallel_render
import gt_parser
import os

if __name__ == '__main__':
    # prepare metadata
    dataset_folder = '/home/ysheng/Dataset/soft_shadow/train'
    out_file = os.path.join(dataset_folder, "metadata.csv")
    gt_parser.parse_folder(dataset_folder, out_file)
    
    # render lights
    parallel_render()