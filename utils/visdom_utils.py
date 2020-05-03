from visdom import Visdom
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch

viz = Visdom(port=8002)
viz2 = Visdom(port=8003)

def get_current_viz(exp=0):
    if exp == 0:
        return viz
    else:
        return viz2

def visdom_plot_loss(win_name, loss, exp=0):
    loss_np = np.array(loss)

    cur_viz = get_current_viz(exp)
    cur_viz.line(win=win_name,
                 Y=loss_np,
                 opts=dict(showlegend=True, legend=[win_name]))

def guassian_light(light_tensor):
    light_tensor = light_tensor.detach().cpu()
    channel = light_tensor.size()[0]
    tensor_ret = torch.zeros(light_tensor.size())
    for i in range(channel):
        light_np = light_tensor[0].numpy() * 100.0
        light_np = gaussian_filter(light_np, sigma=2)
        tensor_ret[i] = torch.from_numpy(light_np)
        tensor_ret[i] = torch.clamp(tensor_ret[i], 0.0, 1.0)
        
    return tensor_ret

def decouple_image(mask_shadow):
    """ Decouple mask and shadow channels and merge them together """
    mask = mask_shadow[:, 0:3, :, :]
    shadow = mask_shadow[:, 3:, :, :]

    return mask, shadow

def normalize_img(imgs):
    b,c,h,w = imgs.shape
    gt_batch = b//2
    for i in range(gt_batch):
        factor = torch.max(imgs[i])
        imgs[i] = imgs[i]/factor
        imgs[gt_batch + i] = imgs[gt_batch + i]/factor
        # imgs[i] = imgs[i]/3.0
        
    imgs = torch.clamp(imgs, 0.0,1.0)
    return imgs

def visdom_show_batch(imgs, win_name=None, exp=0, nrow=2, normalize=True):
    cur_viz = get_current_viz(exp)
    if normalize:
        imgs = normalize_img(imgs)
    
    if win_name is None:
        cur_viz.images(imgs, win="batch visualize",nrow=nrow)
    else:
        cur_viz.images(imgs, win=win_name, opts=dict(title=win_name),nrow=nrow)
    
def visdom_show_light(imgs, win_name=None, exp=0, nrow=2):
    cur_viz = get_current_viz(exp)
    #     imgs = torch.clamp(imgs/torch.max(imgs),0.0,1.0)    
    
    imgs = normalize_img(imgs)
    if win_name is None:
        cur_viz.images(imgs, win="batch visualize",nrow=nrow)
    else:
        cur_viz.images(imgs, win=win_name, opts=dict(title=win_name),nrow=nrow)

def visdom_relight_results(gt_sy, sy, gt_ty, ty, gt_light, light, nov_light, valid=False, exp=0):
    cur_viz = get_current_viz(exp)

    """ Visualize three prediction results """
    prefix = ""
    if valid:
        prefix = "valid "
    else:
        prefix = "train "

    batch_size, c, h, w = gt_sy.size()
    random_batch = min(4, batch_size)
    gt_sy, sy, gt_ty, ty, gt_light, light, nov_light = gt_sy[0:random_batch,:,:,:], \
                                                       sy[0:random_batch,:,:,:], \
                                                       gt_ty[0:random_batch,:,:], \
                                                       ty[0:random_batch,:,:,:], \
                                                       gt_light[0:random_batch,:,:,:], \
                                                       light[0:random_batch,:,:,:], \
                                                       nov_light[0:random_batch,:,:,:]


    # vis_sy = torch.clamp(torch.cat((gt_sy, sy)), 0.0, 1.0)
    # vis_sy_mask, vis_sy_shadow = decouple_image(vis_sy)

    # import pdb; pdb.set_trace()
    vis_sy_mask = torch.clamp(torch.cat((gt_sy[:,0,:,:].view(random_batch,1,h,w), sy[:,0,:,:].view(random_batch,1,h,w))), 0.0, 1.0)
    vis_sy_shadow = torch.clamp(torch.cat((gt_sy[:,1,:,:].view(random_batch,1,h,w), sy[:,1,:,:].view(random_batch,1,h,w))), 0.0, 1.0)

    win_name = prefix + "source image mask vs. inference"
    cur_viz.images(vis_sy_mask, win=win_name, opts=dict(title=win_name), nrow=random_batch)

    win_name = prefix + "source image shadow vs. inference"
    cur_viz.images(vis_sy_shadow, win=win_name, opts=dict(title=win_name), nrow=random_batch)

    # vis_ty = torch.clamp(torch.cat((gt_ty, ty)), 0.0, 1.0)
    # vis_ty_mask, vis_ty_shadow = decouple_image(vis_ty)

    # import pdb;
    # pdb.set_trace()
    vis_ty_mask = torch.clamp(torch.cat((gt_ty[:,0,:,:].view(random_batch,1,h,w), ty[:,0,:,:].view(random_batch, 1, h, w))), 0.0, 1.0)
    vis_ty_shadow = torch.clamp(torch.cat((gt_ty[:,1,:,:].view(random_batch,1,h,w), ty[:,1,:,:].view(random_batch, 1, h, w))), 0.0, 1.0)
    print(torch.max( ty[:,1,:,:]))
    print(torch.min( ty[:,1,:,:]))

    win_name = prefix + "target image mask vs. inference"
    cur_viz.images(vis_ty_mask, win=win_name, opts=dict(title=win_name), nrow=random_batch)

    win_name = prefix + "target image shadow vs. inference"
    cur_viz.images(vis_ty_shadow, win=win_name, opts=dict(title=win_name),nrow=random_batch)

    win_name = prefix + "gt light vs. inference"
    light = light.view(-1, 3, 16, 32)
    
    vis_light = torch.clamp(torch.cat((gt_light, light)),0.0,1.0)
    cur_viz.images(guassian_light(vis_light), win=win_name, opts=dict(title=win_name),nrow=random_batch)

    win_name = prefix + "target light"
    cur_viz.images(guassian_light(nov_light), win=win_name, opts=dict(title=win_name),nrow=random_batch)

def visdom_log(log_info, win_name='logger',exp=0):
    cur_viz = get_current_viz(exp)

    cur_viz.text(log_info, win=win_name)