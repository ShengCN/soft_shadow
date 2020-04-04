import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize

class base_ibl_animator:
    def __init__(self, ibl_num=1):
        self.ibl_num = ibl_num
        self.r = 200
        
    def get_ibl_num(self):
        return self.ibl_num
    
    def compute_ibl(self, i,j, w=512, h=256):
        """ given width, height, (i,j) compute the 16x32 ibls """
        ibl = np.zeros((h,w,1))
        ibl[j,i] = 1.0
        ibl = gaussian_filter(ibl, 20)
        ibl = resize(ibl, (16,32))
        ibl = ibl/np.max(ibl)
        return ibl
    
    # interface 
    def animate_ibl(self, iteration, max_iter):
        fract = iteration / max_iter
        i = int(512 * fract)
        return self.compute_ibl(i, self.r)

class three_ibl_animator:
    def __init__(self, ibl_num=3):
        super().__init__(ibl_num)
        self.r = 200
        
    
    def animate_ibl(self, iteration, max_iter):
        pass