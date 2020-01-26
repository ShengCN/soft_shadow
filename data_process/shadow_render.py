import numpy as np

def render_shadow(relative_vec, w=512, h=256):
    img = np.zeros((h,w,3))
    x,y,z = relative_vec[0], relative_vec[1], relative_vec[2]
    # compute alpha, beta
    alpha = np.arctan2(z,x)
    beta = np.arctan2(y, np.sqrt(x**2 + z**2))
    
    alpha = (np.pi + alpha)/np.pi * 0.5 # [0,1]
    beta = (np.pi + beta)/np.pi * 0.5
    
    coord_x = (int)(beta * (h-1))
    coord_y = (int)(alpha *(w-1))
    img[coord_x, coord_y, :] = 1.0
    
    return img

if __name__ == '__main__':
    testing_vec = np.array([0.0, 0.0, -1.0])
    render_shadow(testing_vec)
    
    testing_vec = np.array([0.0, 0.0, 1.0])
    render_shadow(testing_vec)
    
    testing_vec = np.array([0.0, 1.0, 0.0])
    render_shadow(testing_vec)
    
    testing_vec = np.array([0.0, -1.0, 0.0])
    render_shadow(testing_vec)
    
    testing_vec = np.array([1.0, 0.0, 0.0])
    render_shadow(testing_vec)
    
    testing_vec = np.array([-1.0, 0.0, 0.0])
    render_shadow(testing_vec)