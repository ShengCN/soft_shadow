import matplotlib.pyplot as plt
import os
from utils.time_utils import get_time_stamp
from PIL import Image
import numpy as np

def visualize_folder(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    num_files = len(files)
    cols = 5
    rows = int(num_files/cols)
    print(rows,' ', cols)
    fig = plt.figure(figsize=(50,20))
    for i, file in enumerate(files):
        file_path = os.path.join(folder, file)
        img = np.array(Image.open(file_path))

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img, cmap='gray', interpolation='nearest')
        # ax.set_title(file)

    fig.savefig('{}.png'.format(get_time_stamp()))



if __name__ == '__main__':
    folder = 'results'
    visualize_folder(folder)