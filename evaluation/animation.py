import numpy as np
import imagen as ig
import cv2
import random

# gs = ig.Composite(operator=np.add,
#             generators=[ig.Gaussian(
#                         size=size*ng.UniformRandom(seed=seed+i+4),
#                         scale=scale*(ng.UniformRandom(seed=seed+i+5)+1e-3),
#                         x=ng.UniformRandom(seed=seed+i+1)-0.5,
#                         y=(ng.UniformRandom(seed=seed+i+2)-0.5)*factor,
#                         aspect_ratio=0.7,
#                         orientation=np.pi*ng.UniformRandom(seed=seed+i+3),
#                         ) for i in range(num)],
#                 position=(0, 0), 
#                 xdensity=512)

class base_ibl_animator(object):
    def __init__(self, num):
        cur_pos = [(256/2,512/2)] * num
        
        random.seed(19920208)
        vec = [(random.random(), random.random()) for i in range(num)]



    # interface 
    def animate_ibl(self, iteration, max_iter):
        # 24 * 30 = 720 frames
        # 24 * 15 = 360, just move


        # 24 * 15 = 360, scale the blob size
        pass