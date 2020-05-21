import imageio 
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def composite(mask, cut, back):
    return mask * cut + (1.0-mask) * back

if __name__ == '__main__':
    test_mask, test_cutout = '001_mask.png', '001_cutout.png'
    bg = 'x.jpg'

    bg_np = imageio.imread(bg) / 255.0
    bg_np = np.repeat(bg_np[:,:,np.newaxis], 3, axis=2)
    print(bg_np.shape)

    mask = imageio.imread(test_mask)/255.0
    cutout = imageio.imread(test_cutout)/255.0
    
    bg_np = bg_np[:,:,:3]
    bg_np = cv2.resize(bg_np, (256,256))
    mask = mask[:,:,:3]
    cutout = cutout[:,:,:3]

    comp = composite(mask, cutout, bg_np)

    plt.figure()
    plt.imshow(comp)
    plt.show()