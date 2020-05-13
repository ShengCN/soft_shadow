import numpy as np
import math

def rmse(I1, I2):
    """ Compute root min square errors between I1 and I2
    """
    diff = I1 - I2
    num_pixels = float(diff.size)
    return np.sqrt(np.sum(np.square(diff)))/num_pixels



if __name__ == '__main__':
    pass