import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def tensorboard_plot_loss(win_name, loss, writer):
#     x = np.arange(1, 1 + len(loss))
    x = len(loss)
    writer.add_scalar("Loss/{}".format(win_name), loss[-1], x)
    writer.flush()

def normalize_img(imgs):
    b,c,h,w = imgs.shape
    gt_batch = b//2
    for i in range(gt_batch):
        factor = torch.max(imgs[i])
        imgs[i] = imgs[i]/factor
        imgs[gt_batch + i] = imgs[gt_batch + i]/factor

    imgs = torch.clamp(imgs, 0.0,1.0)
    return imgs

def tensorboard_show_batch(imgs, writer, win_name=None, nrow=2, normalize=True):
    if normalize:
        imgs = normalize_img(imgs)

    writer.add_images('{}'.format(win_name), imgs)
    writer.flush()

def tensorboard_log(log_info, writer, win_name='logger'):
    writer.add_text(win_name, log_info)
    writer.flush()
