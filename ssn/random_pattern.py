import random
import time 
import numbergen as ng
import imagen as ig
from skimage.transform import resize
import numpy as np

class random_pattern():
    def __init__(self, maximum_blob=50):
#         self.generator_list = []
        
#         start = time.time()
#         for i in range(maximum_blob):
#             self.generator_list.append(ig.Gaussian(size=))
#         print('random pattern init time: {}s'.format(time.time()-start))
        
        pass

    def get_pattern(self, num=50, scale=1.0, size=0.1):
        seed = random.randint(0,19920208)
        if num == 0:
            return np.zeros((256,256))
            
        gs = ig.Composite(operator=np.add,
                          generators=[ig.Gaussian(
                                      size=size,
                                      scale=scale,
                                      x=ng.UniformRandom(seed=seed+i+1)-0.5,
                                      y=(ng.UniformRandom(seed=seed+i+2)-0.5) * 2.0,
                                      orientation=np.pi*ng.UniformRandom(seed=seed+i+3),
                                      aspect_ratio=1.0/0.71) for i in range(num)])
        # ret = np.power(gs(), 3)
        return gs()