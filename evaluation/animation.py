import numpy as np
import numbergen as ng
import imagen as ig
import cv2
import random
from tqdm import tqdm

class base_ibl_animator(object):
    def __init__(self, num, size,verbose=True):
        random.seed(19920208)
        
        self.num = num
        self.verbose = verbose
        self.cur_pos = []
        self.cur_vec = []
        self.cur_size = np.array([size] * num) 
        self.cur_resize = np.array([random.random() for i in range(num)]) # < 0.5, smaller

        for i in range(num):
            self.cur_pos.append((512/2, 80/2))
            self.cur_vec.append((random.random() * 2.0 - 1.0, random.random() * 2.0 - 1.0))
        
        self.cur_pos = np.array(self.cur_pos)
        self.cur_vec = np.array(self.cur_vec)


        # normalize vector
        for i in range(num):
             self.cur_vec[i] =  self.cur_vec[i] / np.linalg.norm(self.cur_vec[i], 2)

        if self.verbose:
            print('pos: \n ', self.cur_pos)
            print('vec: \n ', self.cur_vec)
            print('size:\n ',self.cur_resize)

    def move_advance(self):
        for i, p in enumerate(self.cur_pos):
            self.cur_pos[i] += self.cur_vec[i] * 2.0

            factor = 0.8
            self.cur_vec[i][0] = factor * self.cur_vec[i][0] + (1.0-factor) * (random.random() * 2.0 - 1.0)
            self.cur_vec[i][1] = factor * self.cur_vec[i][1] + (1.0-factor) * (random.random() * 2.0 - 1.0)
            self.cur_vec[i] = self.cur_vec[i]/np.linalg.norm(self.cur_vec[i], 2)

            if self.cur_pos[i][0] >= 511 or self.cur_pos[i][0]<=1:
                self.cur_vec[i][0] = -self.cur_vec[i][0]
            
            if self.cur_pos[i][1] >= 79 or self.cur_pos[i][1] <=1:
                self.cur_vec[i][1] = -self.cur_vec[i][1]


    def resize_advance(self):
        for i, p in enumerate(self.cur_size):
            if self.cur_size[i] < 0.002:
                self.cur_resize[i] = 0.7

            if self.cur_size[i] > 0.19:
                self.cur_resize[i] = 0.3

            if self.cur_resize[i] <= 0.5:
                self.cur_size[i] = self.cur_size[i] * 0.95
            else:
                self.cur_size[i] = self.cur_size[i] * 1.05
        
        self.cur_size = np.clip(self.cur_size, 0.001, 0.2) 

    # compute current ibl pattern given position and size list
    def get_cur_ibl(self):
        gs = ig.Composite(operator=np.add,
                generators=[ig.Gaussian(
                            size=self.cur_size[i],
                            scale=1.0,
                            x=(self.cur_pos[i][0]/512)-0.5,
                            y=(1.0 - self.cur_pos[i][1]/256)-0.5,
                            aspect_ratio=1.0,
                            ) for i in range(self.num)],
                    xdensity=512)
        return gs()

    # interface 
    def animate_ibl(self, iteration, max_iter):

        # 24 * 30 = 720 frames
        # 24 * 15 = 360, just move
        if iteration < 360:
            self.move_advance()
            return self.get_cur_ibl()

        # 24 * 5 = 360 + 120 = 480, scale the blob size
        if iteration >= 360 and iteration < 480: 
            self.resize_advance()
            return self.get_cur_ibl()

        self.move_advance()
        return self.get_cur_ibl()
