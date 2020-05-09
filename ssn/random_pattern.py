import random
import time 
import numbergen as ng
import imagen as ig
import numpy as np
import cv2

class random_pattern():
    def __init__(self, maximum_blob=50):
#         self.generator_list = []
        
#         start = time.time()
#         for i in range(maximum_blob):
#             self.generator_list.append(ig.Gaussian(size=))
#         print('random pattern init time: {}s'.format(time.time()-start))
        
        pass

    def y_transform(self, y):
        # y = []
        pass

    def get_pattern(self, num=50, scale=3.0, size=0.1, energy=3500, mitsuba=False, seed=None):
        if seed is None:
            seed = random.randint(0,19920208)
        else:
            seed = seed + int(time.time())

        if num == 0:
            return np.zeros((80,512))

        factor = 80/256
        gs = ig.Composite(operator=np.add,
                        generators=[ig.Gaussian(
                                    size=size*ng.UniformRandom(seed=seed+i+4),
                                    scale=scale*(ng.UniformRandom(seed=seed+i+5)+1e-3),
                                    x=ng.UniformRandom(seed=seed+i+1)-0.5,
                                    y=(ng.UniformRandom(seed=seed+i+2)-0.5)*factor,
                                    aspect_ratio=0.7,
                                    orientation=np.pi*ng.UniformRandom(seed=seed+i+3),
                                    ) for i in range(num)],
                            position=(0, 0), 
                            xdensity=512)
        ibl = self.normalize(gs()[80:160,:], energy) 

        if mitsuba:
            return ibl, self.to_mts_ibl(np.copy(ibl))
        else:
            return ibl


    def to_mts_ibl(self, ibl):
        """ Input: 80 x 512 pattern generated ibl 
            Output: the ibl in mitsuba ibl
        """
        cur_ibl = np.repeat(ibl[:,:,np.newaxis], 3, axis=2)
        ret_ibl = np.zeros((256,512, 3))
        ret_ibl[:80,:, :] = cur_ibl
        return ret_ibl

    def normalize(self, ibl, energy=3500):
        total_energy = np.sum(ibl)
        if total_energy < 1e-3:
            print('small energy: ', total_energy)
            return np.zeros((80,512))

        ibl = ibl * energy / total_energy

        return ibl